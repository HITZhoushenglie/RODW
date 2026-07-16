from email import header
import os
import argparse
import datetime
import json
import logging
import random
import time
from tarfile import data_filter

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from kornia.metrics import psnr, ssim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import sys
sys.path.append('/home/ubuntu/code/RobustWide-master/')
from rein_learning.attack_layer.DefocusBlurAttack import DefocusBlurAttack
from rein_learning.attack_layer.GaussianBlurAttack import GaussianBlurAttack
from rein_learning.custom_pipe import CustomStableDiffusionInstructPix2PixPipeline
# from dataset import get_hugging_instruct_pix2pix_dataset, collate_fn
# from model import WatermarkModel
from rein_learning.model import WatermarkModel
from utils import (
    decoded_message_error_rate_batch,
    denormalize,
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- encoding: utf-8 -*-
'''
@File    :   train_new.py
@Time    :   2025/03/05 15:53:24
@Author  :   Shenglie zhou 
@Version :   1.0
@Contact :   betterWL@hotmail.com
'''
from ppo_loss import ppoLoss
from tqdm import tqdm
import textwrap
# from prompt import ImagenetAnimalPrompts, SinglePrompt, ImageRewardPrompt, HPSPrompt
# from rewards import MultiReward
from collections import defaultdict
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
# from trl import DDPOConfig
from ddpo_config import DDPOConfig
# from transformers import HfArgumentParser
from trl.trainer.utils import PerPromptStatTracker
from transformers import HfArgumentParser
# from rein_learning.ddpo import DDPOTrainer
from prompt_reward_fn import reward_fn, prompt_fn, image_outputs_logger, write_text_on_image_tensor
from peft import LoraConfig, get_peft_model 
from scriptarguments import ScriptArguments


from rein_learning.my_datasets import InjectDataset, collate_fn

logger = logging.getLogger(__name__)


class Main(): # ddpo_config, args, args_ddpo, reward_fn, prompt_fn
    def __init__(self, ddpo_config, args, reward_function, prompt_function):
        self.config = ddpo_config
        self.train_config = args
        self.prompt_fn = prompt_function
        self.reward_fn = reward_function
        if args.seed is not None:
                set_seed(args.seed)
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.args_dict = vars(args)
        output_with_time_dir = os.path.join(args.output_dir, now)
        os.makedirs(output_with_time_dir, exist_ok=True)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%m/%d/%Y %H:%M:%S")
        fhlr = logging.FileHandler(os.path.join(output_with_time_dir, "log.txt"))
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

        # number of timesteps within each trajectory to train on
        self.num_train_timesteps = int(self.config.sample_num_steps * self.config.train_timestep_fraction)

        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

        device = self.accelerator.device
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.wm_model_config = OmegaConf.load(args.wm_model_config)
        args.message_length = self.wm_model_config["wm_enc_config"]["message_length"]
        wm_model = WatermarkModel(
            **self.wm_model_config,
            device=device,
            weight_dtype=weight_dtype,
        )

        # self.DefocusBlur = DefocusBlurAttack()
        # self.GaussianBlur = GaussianBlurAttack()
        # self.DefocusBlur.to(device)
        # self.GaussianBlur.to(device)
        # self.attacks = []

        # params_to_optimize = list(p for p in wm_model.parameters() if p.requires_grad)

        self.sd_pipeline = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
            "/home/ubuntu/code/RobustWide-master/timbrooks/instruct-pix2pix",
            torch_dtype=weight_dtype,
            local_files_only=True
        ).to(device)

        # 1 load lora weight and text encoder parameters
        self.sd_pipeline.load_lora_weights(
                "/home/ubuntu/code/RobustWide-master/TexForce/lora_weights/sd15_refl",
                weight_name="pytorch_lora_weights.bin",
                revision=None,
            )

        self.sd_pipeline.freeze_params()

        # set lora config
        trainable_params = []
        # lora_config = LoraConfig(
        #     r=8,  
        #     lora_alpha=16, 
        #     target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  
        #     lora_dropout=0.1,
        #     bias="none",
        #     task_type="TEXT_ENCODER"
        # )
        lora_config = LoraConfig(
            r=self.train_config.text_lora_r,
            lora_alpha=self.train_config.text_lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )

        self.sd_pipeline.text_encoder.add_adapter(lora_config)
        
        # load parameter
        # To avoid accelerate unscaling problems in FP16.
        for param in self.sd_pipeline.text_encoder.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                trainable_params.append(param)
        
        # params_to_optimize = []
        # params_to_optimize.extend([p for p in wm_model.parameters() if p.requires_grad])
        # params_to_optimize.extend([p for p in self.sd_pipeline.text_encoder.parameters() if p.requires_grad])
        params_to_optimize = [p for p in list(wm_model.parameters()) + trainable_params if p.requires_grad]
        wm_model.train()
        self.sd_pipeline.text_encoder.train()
        # self.sd_pipeline.freeze_params()
        # self.sd_pipeline.text_encoder.train()
        # self.sd_pipeline.unet.train()
        # self.sd_pipeline.vae.train()

        # rewards func
        self.reward_fn.to(self.accelerator.device, dtype=weight_dtype)

        # train_dataset = get_hugging_instruct_pix2pix_dataset(args.train_data_dir, args.image_size, accelerator)
        with self.accelerator.main_process_first():
            train_dataset = InjectDataset(args.train_data_dir, args.image_size)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=64,            
            pin_memory=True,
        )

        opt = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate,)
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=opt,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        # 2 use tokenizer and generate neg_prompt 
        self.neg_prompt_embed = self.sd_pipeline.text_encoder(
            self.sd_pipeline.tokenizer(
                [""] if self.config.negative_prompts is None else self.config.negative_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
        )[0]

        wm_model, opt, train_dataloader, lr_scheduler = self.accelerator.prepare(
            wm_model, opt, train_dataloader, lr_scheduler
        )
        wm_model = wm_model
        unwrapped_wm_model = self.accelerator.unwrap_model(wm_model)

        # NOTE: for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses more memory
        # self.autocast = self.sd_pipeline.autocast or self.accelerator.autocast
        self.autocast = self.accelerator.autocast

        # Start Train!
        self.train(args, output_with_time_dir, device, weight_dtype, train_dataloader, opt, lr_scheduler, wm_model, unwrapped_wm_model)
    
    def save_all(self, g_model, save_dir):
        unwrapped_model = self.accelerator.unwrap_model(g_model)
        self.accelerator.save(unwrapped_model.state_dict(), os.path.join(save_dir, "wm_model.ckpt"))
        with open(os.path.join(save_dir, "train_config.json"), "w") as f:
            json.dump(self.args_dict, f, indent=2)
        OmegaConf.save(self.wm_model_config, os.path.join(save_dir, "wm_model_config.yaml"))

    def train(self, args, output_with_time_dir, device, weight_dtype, train_dataloader, opt, lr_scheduler,  wm_model, unwrapped_wm_model):
        
        # set multi-turn editing prompt
        # edit_prompt_path = os.path.join(args.train_data_dir, "muti_edit_prompt_200.txt")
        # prompt_texts = []
        # try:
        #     with open(edit_prompt_path, 'r', encoding='utf-8') as file:
        #         for line in file:
        #             items = line.strip().split(',')
        #             items = [item.strip() for item in items if item.strip()]
        #             prompt_texts.append(items)
        #     num = len(prompt_texts)
        #     # edit_nums = len(prompt_texts[0]) - 0 # Default editing times = 4, 3, 2, 1, 0 -> 1, 2, 3, 4, 5
        #     edit_nums = len(prompt_texts[0]) - 3
        #     print(num, edit_nums)
        # except FileNotFoundError:
        #     print(f"错误：文件 {edit_prompt_path} 未找到")
        #     return []
        # except Exception as e:
        #     print(f"读取文件时发生错误: {e}")
        #     return []
        
        step = 0
        global_step = 0
        finished_flag = False
        # torch.cuda.reset_peak_memory_stats()
        # start_time = time.time()
        # memory_list = []
        while True:
            for data in train_dataloader:
                step += 1
                with self.accelerator.accumulate(wm_model):
                    message = torch.randint(0, 2, (args.batch_size, args.message_length)).to(
                        device=device, dtype=torch.float32
                    )
                    # {"image": images, "prompt": prompt, "refac_prompt": refac_prompt}
                   
                    image, prompt_ori, prompt_refac = data["image"], data["prompt"], data["refac_prompt"]

                    # wm_image = wm_model.encoder(image, message)
                    wm_image = unwrapped_wm_model.encoder(image, message)
                    
                    image_latents = self.sd_pipeline.vae.encode(image.to(dtype=weight_dtype)).latent_dist.mode() # z_ori z_wm
                    wm_image_latents = self.sd_pipeline.vae.encode(wm_image.to(dtype=weight_dtype)).latent_dist.mode()

                    # decoded_message_before_edit = wm_model.decoder(wm_image.to(dtype=torch.float32))
                    decoded_message_before_edit = unwrapped_wm_model.decoder(wm_image.to(dtype=torch.float32))

                    # 3 generated latents and log_prob for each timestep # [b t c h w]
                    # Diversification strategy based on visual language samples guidance_scale: 10 5 128x128(3) 1
                    cycle = (step - 1) // 20000  
                    if cycle % 2 == 0:
                        generated_image, image_latents_, latents,  log_prob = self.sd_pipeline(
                        prompt_ori, image=wm_image, num_images_per_prompt=1, num_inference_steps=20,
                        guidance_scale=10, image_guidance_scale=1.5, last_grad_steps=args.last_grad_steps,
                        output_type="pt",
                    )
                    else:
                        generated_image, image_latents_, latents,  log_prob = self.sd_pipeline(
                            prompt_refac, image=wm_image, num_images_per_prompt=1, num_inference_steps=20,
                            guidance_scale=10, image_guidance_scale=1.5, last_grad_steps=args.last_grad_steps,
                            output_type="pt",
                        )
                        
                    # 3 multi-turn editing
                    # edited_image = []
                    # image_latents_1 = []
                    # latents_ = []
                    # log_prob_ = []

                    # for i in range(args.batch_size):
                    #     current_img = wm_image[i:i+1]  # 保持 4D 张量 [1, C, H, W]
                    #     generated_image, image_latents_, latents, log_prob = self.sd_pipeline(
                    #         prompt_texts[i][0], image=current_img, num_images_per_prompt=1, num_inference_steps=20,
                    #         guidance_scale=10, image_guidance_scale=1.5, last_grad_steps=args.last_grad_steps,
                    #         output_type="pt",
                    #     )

                    #     for idy in range(1, edit_nums):
                    #         generated_image, image_latents_, latents, log_prob = self.sd_pipeline(
                    #             prompt_texts[i][idy], image=generated_image, num_images_per_prompt=1, num_inference_steps=20,
                    #             guidance_scale=10, image_guidance_scale=1.5, last_grad_steps=args.last_grad_steps,
                    #             output_type="pt",
                    #         )

                    #     edited_image.append(generated_image)
                    #     image_latents_1.append(image_latents_)
                        
                    #     latents_.append(latents)     
                    #     log_prob_.append(log_prob)   

                    # generated_image = torch.cat(edited_image, dim=0)      
                    # image_latents_ = torch.cat(image_latents_1, dim=0)        

                    # latents = []
                    # for a, b in zip(latents_[0], latents_[1]):
                    #     latents.append(torch.cat([a, b], dim=0)) 

                    # log_prob = []
                    # for a, b in zip(log_prob_[0], log_prob_[1]):
                    #     log_prob.append(torch.cat([a, b], dim=0))  

                    # 3 generated latents and log_prob for each timestep # [b t c h w]
                    # Reinforcement learning optimization strategy (only LLMs edit instruction are used.)
                    # generated_image, image_latents_, latents,  log_prob = self.sd_pipeline(
                    #     prompt_ori, image=wm_image, num_images_per_prompt=1, num_inference_steps=20,
                    #     guidance_scale=10, image_guidance_scale=1.5, last_grad_steps=args.last_grad_steps,
                    #     output_type="pt",
                    # )

                    # 3 generated latents and log_prob for each timestep # [b t c h w]
                    # only VLMs captions are used.
                    # generated_image, image_latents_, latents,  log_prob = self.sd_pipeline(
                    #         prompt_refac, image=wm_image, num_images_per_prompt=1, num_inference_steps=20,
                    #         guidance_scale=10, image_guidance_scale=1.5, last_grad_steps=args.last_grad_steps,
                    #         output_type="pt",
                    #     )

                    # 4 attack edited image 
                    # self.attacks.append(self.DefocusBlur)
                    # self.attacks.append(self.GaussianBlur)
                    # random_attack = random.choice(self.attacks)
                    # generated_image = random_attack(generated_image)
                    
                    # 5 generated sample and prompt data
                    samples, prompt_image_data = self._generate_samples(generated_image, image_latents_, latents, log_prob, batch_size=self.train_config.batch_size,)
                    
                    # decoded_message_after_edit = wm_model.decoder(generated_image.to(dtype=torch.float32))
                    decoded_message_after_edit = unwrapped_wm_model.decoder(generated_image.to(dtype=torch.float32))

                    # 6 Calculate the total loss
                    enc_pixel_loss = F.mse_loss(image.float(), wm_image.float())
                    enc_latent_loss = F.mse_loss(image_latents.float(), wm_image_latents.float())
                    dec_loss_before_edit = F.mse_loss(message, decoded_message_before_edit)
                    dec_loss_after_edit = F.mse_loss(message, decoded_message_after_edit)
                    enc_loss = enc_pixel_loss + args.enc_latent_weight * enc_latent_loss
                    dec_loss = dec_loss_before_edit + args.decoder_weight * dec_loss_after_edit
                    # dec_loss = args.decoder_weight * dec_loss_after_edit

                    # 7 calculate opt loss and backward to update parameters
                    all_loss, opt_loss, rewards = self.step(args, opt, lr_scheduler, enc_loss, dec_loss, samples, prompt_image_data)
                    # loss = enc_loss + dec_loss + args.opt_rein_weight * opt_loss
                    # loss = enc_loss + dec_loss

                    # 8 caculate GPUs mememory
                    # peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    # memory_list.append(peak)

                if self.accelerator.sync_gradients:
                    global_step += 1
                    if self.accelerator.is_main_process:
                        if global_step % args.log_steps == 0:
                            psnr_value = psnr(denormalize(wm_image.detach()), denormalize(image), 1)
                            ssim_value = torch.mean(ssim(denormalize(wm_image.detach()), denormalize(image), window_size=5))
                            error_rate_after_edit = decoded_message_error_rate_batch(
                                message, decoded_message_after_edit
                            )
                            error_rate_before_edit = decoded_message_error_rate_batch(
                                message, decoded_message_before_edit
                            )
                            # "opt_loss": opt_loss.detach().item(), "dec_loss_before_edit": dec_loss_before_edit.detach().item(),
                            log_dict = {
                                "step": step,
                                "global_step": global_step,
                                "lr": lr_scheduler.get_last_lr()[0],
                                "enc_pixel_loss": enc_pixel_loss.detach().item(),
                                "enc_latent_loss": enc_latent_loss.detach().item(),
                                "dec_loss_before_edit": dec_loss_before_edit.detach().item(),
                                "dec_loss_after_edit": dec_loss_after_edit.detach().item(),
                                "opt_loss": opt_loss.detach().item(),
                                "all_loss": all_loss.detach().item(),
                                "psnr": psnr_value.item(),
                                "ssim": ssim_value.item(),
                                "rewards_0": rewards[0].item(),
                                "rewards_1": rewards[1].item(),
                                "error_rate_before_edit": error_rate_before_edit,
                                "error_rate": error_rate_after_edit,
                            }
                            logger.info(log_dict)

                        if global_step % args.save_steps == 0:
                            save_step_dir = os.path.join(output_with_time_dir, f"step{global_step}")
                            os.makedirs(save_step_dir, exist_ok=True)
                            logger.info("save models!")
                            self.save_all(wm_model, save_step_dir)

                # test train time
                # if global_step >= args.single_epoch_train_steps:
                #     finished_flag = True
                #     break

                if global_step >= args.max_train_steps:
                    finished_flag = True
                    break

            if finished_flag:
                break
        
        # avg_memory = sum(memory_list) / len(memory_list)
        # single_epoch_train_time = time.time() - start_time
        # throughput_train = (args.single_epoch_train_steps * args.batch_size) / single_epoch_train_time
        # print(f"Avg Peak Memory: {avg_memory:.2f} GB")
        # print(f"Total time for {args.single_epoch_train_steps} steps: {single_epoch_train_time / 3600:.2f} hours")
        # print(f"Average time per step: {single_epoch_train_time / args.single_epoch_train_steps * 1000:.2f} ms")
        # print(f"Training Throughput: {throughput_train:.2f} img/s")

        if self.accelerator.is_main_process:
            self.save_all(wm_model, output_with_time_dir)

        self.accelerator.end_training()

    def compute_rewards(self, prompt_image_pairs, is_async=False):
        if not is_async:
            rewards = []
            for images, prompts, prompt_metadata in prompt_image_pairs:
                # calculate rewards one-by-one to avoid OOM
                tmp_rewards = []
                tmp_reward_metadata = []
                for i in range(len(images)):
                    reward, reward_metadata = self.reward_fn(images[[i]], [prompts[i]], prompt_metadata)
                    tmp_rewards.append(reward)
                    tmp_reward_metadata.append(reward_metadata)

                rewards.append(
                    (
                        torch.as_tensor(tmp_rewards, device=self.accelerator.device),
                        reward_metadata,
                    )
                )
        else:
            rewards = self.executor.map(lambda x: self.reward_fn(*x), prompt_image_pairs)
            rewards = [
                (torch.as_tensor(reward.result(), device=self.accelerator.device), reward_metadata.result())
                for reward, reward_metadata in rewards
            ]

        return zip(*rewards)
    
    def step(self, args, opt, lr_scheduler, enc_loss, dec_loss, samples, prompt_image_data):
        # self.logger.info(f"Finished sampling and start to compute rewards.")
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}
        rewards, rewards_metadata = self.compute_rewards( # False
            prompt_image_data, is_async=self.config.async_reward_computation
        )

        for i, image_data in enumerate(prompt_image_data):
            image_data.extend([rewards[i], rewards_metadata[i]])
        rewards = torch.cat(rewards)
        rewards = self.accelerator.gather(rewards).cpu().numpy()

        if self.config.per_prompt_stat_tracking: # False
            # gather the prompts across processes
            prompt_ids = self.accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = self.sd_pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = self.stat_tracker.update(prompts, rewards)
        else:

            # print(f"rewards = {rewards}, mean = {rewards.mean()}, std = {rewards.std()}")
            # print(f"mean = {rewards - rewards.mean()}, std = {rewards.std() + 1e-8}")
            # advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # print(f"rewards = {rewards}, mean = {rewards.mean()}, std = {rewards.std() + 1e-4}, advantages = {advantages}")

        # ungather advantages;  keep the entries corresponding to the samples on this process
        samples["advantages"] = (
            torch.as_tensor(advantages)
            .reshape(self.accelerator.num_processes, -1)[self.accelerator.process_index]
            .to(self.accelerator.device)
        )

        # shape  latents:[2 4 32 32] next_latents:[2 4 32 32] log_probs:[2]
        original_keys = samples.keys()
        original_values = samples.values()

        # print(type(original_values), len(original_values))
        # original_values_ = list(original_values)
        # for v in original_values_:
        #     print(f"original_values={v.shape}") # shape '[-1, 2, 4, 16, 16]' is invalid for input of size 3072

        # rebatch them as user defined train_batch_size is different from sample_batch_size
        reshaped_values = [v.reshape(-1, self.config.train_batch_size, *v.shape[1:]) for v in original_values]

        # Transpose the list of original values
        transposed_values = zip(*reshaped_values)

        # Create new dictionaries for each row of transposed values
        samples_batched = [dict(zip(original_keys, row_values)) for row_values in transposed_values]

        # caculate opt loss  samples_batched:[1]
        loss, opt_loss = self._caculate_batched_samples_loss(args, samples_batched, opt, lr_scheduler, enc_loss, dec_loss)
        return loss, opt_loss, rewards
    
    @torch.no_grad()
    def _generate_samples(self, generated_image, image_latents_, latents, log_prob, batch_size):
        """
        Generate samples from the model

        Args:
            iterations (int): Number of iterations to generate samples for
            batch_size (int): Batch size to use for sampling

        Returns:
            samples (List[Dict[str, torch.Tensor]]), prompt_image_pairs (List[List[Any]])
        """
        # ori code
        samples = []
        prompt_image_pairs = []
        self.sd_pipeline.unet.eval()
        with self.autocast():
            images = generated_image
            image_latents = image_latents_ # [6 4 32 32]
            latents = latents
            log_probs = log_prob
        
        prompts, prompt_metadata = zip(*[self.prompt_fn() for _ in range(batch_size)])

        neg_prompt_ids = self.sd_pipeline.tokenizer(
            [""] if self.config.negative_prompts is None else self.config.negative_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.sd_pipeline.tokenizer.model_max_length,
        ).input_ids.to(self.accelerator.device)
        neg_prompt_ids = neg_prompt_ids.repeat(batch_size, 1)
        sample_neg_prompt_embeds = self.sd_pipeline.text_encoder(neg_prompt_ids)[0]

        prompt_ids = self.sd_pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.sd_pipeline.tokenizer.model_max_length,
        ).input_ids.to(self.accelerator.device)
        prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

        # timesteps shape: [2 20]
        latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, ...)
        log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
        timesteps = self.sd_pipeline.scheduler.timesteps.repeat(batch_size, 1)  # (batch_size, num_steps)
        samples.append(
            {
                "prompt_ids": prompt_ids, # [2 77]
                "neg_prompt_ids": neg_prompt_ids, # [2 77]
                "prompt_embeds": prompt_embeds, # [2 77 768]
                "timesteps": timesteps, # [2 20]
                "image_latents": image_latents, # [2 4 32 32]
                "latents": latents[:, :-1],  # [2 20 4 32 32]
                "next_latents": latents[:, 1:], # [2 20 4 32 32]
                "log_probs": log_probs, # [2 20]
                "negative_prompt_embeds": sample_neg_prompt_embeds, # [2 77 768]
            }
        )
        prompt_image_pairs.append([images.to(self.accelerator.device), prompts, prompt_metadata])
        return samples, prompt_image_pairs

    def calculate_loss(self, j, latents, timesteps, image_latents, next_latents, log_probs, advantages, embeds):
        """
        Calculate the loss for a batch of an unpacked sample

        Args:
            latents (torch.Tensor):
                The latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            timesteps (torch.Tensor):
                The timesteps sampled from the diffusion model, shape: [batch_size]
            next_latents (torch.Tensor):
                The next latents sampled from the diffusion model, shape: [batch_size, num_channels_latents, height, width]
            log_probs (torch.Tensor):
                The log probabilities of the latents, shape: [batch_size]
            advantages (torch.Tensor):
                The advantages of the latents, shape: [batch_size]
            embeds (torch.Tensor): 
                The embeddings of the prompts, shape: [2*batch_size or batch_size, ...]
                Note: the "or" is because if train_cfg is True, the expectation is that negative prompts are concatenated to the embeds

        Returns:
            loss (torch.Tensor), approx_kl (torch.Tensor), clipfrac (torch.Tensor)
            (all of these are of shape (1,))
        """
        # latents:[2 4 32 32] image_latents:[2 4 32 32] timesteps: [] next_latents: [2 4 32 32] log_probs: [2] advantages:[2] embeds:[4 77 768]
        latents_ = torch.cat([latents] * 3)
        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
        # noise_latent: [6 8 32 32 ] t: [] prompt_embeds: [6 77 768]
        scaled_latent_model_input = torch.cat([latents_, image_latents], dim=1)
        if j < self.config.sample_num_steps - 3:
            with torch.no_grad():
                noise_pred = self.sd_pipeline.unet(
                    scaled_latent_model_input,
                    timesteps,
                    embeds,
                )[0]
        else:
            noise_pred = self.sd_pipeline.unet(
                scaled_latent_model_input,
                timesteps,
                embeds,
            )[0]

        sigma = self.sd_pipeline.scheduler.sigmas[j]
        noise_pred = latents_ - sigma * noise_pred

        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
            noise_pred_uncond
            + self.config.sample_guidance_scale * (noise_pred_text - noise_pred_image)
            + self.config.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        )

        noise_pred = (noise_pred - latents) / (-sigma)

        # compute the log prob of next_latents given latents under the current model
        _, log_prob = self.sd_pipeline.scheduler_step(
            noise_pred,
            timesteps,
            latents,
            prev_sample=next_latents,
        )
        
        # print(noise_pred.shape, next_latents.shape, advantages.shape, embeds.shape, log_prob.shape)
        # "latents": latents[:, :-1],  # [2 20 4 32 32]
        # "next_latents": latents[:, 1:], # [2 20 4 32 32]
        #  latents:[2 4 32 32] timesteps: [] next_latents: [2 4 32 32] log_probs: [2] advantages:[2] embeds:[6 77 768] train_adv_clip_max: 10
        advantages = torch.clamp(
            advantages,
            -self.config.train_adv_clip_max,
            self.config.train_adv_clip_max,
        )
        
        # print(f"log_prob = {log_prob}, log_probs = {log_probs}")
        # Add cropping and minimum operation: prevent over update
        with torch.no_grad():
            delta_log = torch.clamp(log_prob - log_probs, min=-50.0, max=50.0)  # limit range
        ratio = torch.exp(delta_log) # new_policy / old_policy
        
        # ratio = torch.exp(log_prob - log_probs)
        # train_clip_range: float = 1e-4
        loss = self.loss(advantages, self.config.train_clip_range, ratio)
        # print(f"ratio = {ratio}, log_prob = {log_prob}, log_probs= {log_probs}, advantages = {advantages}")
        # approx_kl = 0.5 * torch.mean((log_prob - log_probs) ** 2)

        # When delta-log is small, use Taylor approximation, and when it is large, use ratio method
        safe_mask = (torch.abs(delta_log) < 5.0).float()
        approx_kl = 0.5 * torch.mean(safe_mask * delta_log ** 2) + \
                    torch.mean((1 - safe_mask) * (torch.exp(delta_log) - delta_log - 1))

        clipfrac = torch.mean((torch.abs(ratio - 1.0) > self.config.train_clip_range).float())

        return loss, approx_kl, clipfrac

    def loss(
        self,
        advantages: torch.Tensor,
        clip_range: float,
        ratio: torch.Tensor,
    ):
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * torch.clamp(
            ratio,
            1.0 - clip_range,
            1.0 + clip_range,
        )
        # clip_range = 0.0001 ratio = 0.0
        # print(f"advantages = {advantages}, ratio = {ratio}, clip_range = {clip_range}")
        # print(f"unclipped_loss = {unclipped_loss}, clipped_loss = {clipped_loss}")

        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))

    def _caculate_batched_samples_loss(self, args, batched_samples, opt, lr_scheduler, enc_loss, dec_loss):
        loss_ = 0.0
        for _i, sample in enumerate(batched_samples):
            for j in range(self.num_train_timesteps):
                if self.train_config.text_lora_r > 0:
                    embeds = self.sd_pipeline.text_encoder(sample["prompt_ids"])[0]
                    neg_embeds = self.sd_pipeline.text_encoder(sample["neg_prompt_ids"])[0]
                else:
                    embeds = sample["prompt_embeds"]
                    neg_embeds = sample["negative_prompt_embeds"]

                if self.config.train_cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes embeds:[2 77 768]
                    # embeds = torch.cat([neg_embeds, embeds])
                    embeds = torch.cat([embeds, neg_embeds, neg_embeds])
                with self.accelerator.accumulate([self.sd_pipeline.unet, self.sd_pipeline.text_encoder]):
                    #  latents:[1 4 32 32] timesteps: [1 20] next_latents: [1 4 32 32] log_probs: [1] advantages:[1] embeds:[2 77 768]
                    # loss, _, _ = self.calculate_loss(
                    #     sample["log_probs"][:, len(sample["log_probs"])-1],
                    #     sample["log_probs"][:, len(sample["log_probs"])-2],
                    #     sample["advantages"],
                    # )
                    # latents:[2 20 4 32 32] timesteps: [2 20] next_latents: [2 20 4 32 32] log_probs: [2 20] advantages:[2] embeds:[4 77 768]
                    loss, approx_kl, _ = self.calculate_loss(j,
                            sample["latents"][:, j],
                            sample["timesteps"][0, j],
                            sample["image_latents"],
                            sample["next_latents"][:, j],
                            sample["log_probs"][:, j],
                            sample["advantages"],
                            embeds,
                        )
                # print(f"loss = {loss}, approx_kl={approx_kl}") loss_ = loss + approx_kl
                loss_ = loss + approx_kl
                # loss_ += loss 
        opt_loss = loss_ / (len(batched_samples * self.num_train_timesteps))
        loss = enc_loss + dec_loss + args.opt_rein_weight * opt_loss

        # 7 update parameters
        self.accelerator.backward(loss)
        opt.step()
        lr_scheduler.step()
        opt.zero_grad()

        # return loss / (len(batched_samples * self.num_train_timesteps))
        # return loss / len(batched_samples) 
        return loss, opt_loss

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    args, ddpo_config = parser.parse_args_into_dataclasses()
    Main(ddpo_config, args, reward_fn(args), prompt_fn(args))
