from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import warnings
import numpy as np
import PIL
import torch

from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import logging
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class DDPOPipelineOutput:
    """
    Output class for the diffusers pipeline to be finetuned with the DDPO trainer

    Args:
        images (`torch.Tensor`):
            The generated images.
        latents (`List[torch.Tensor]`):
            The latents used to generate the images.
        log_probs (`List[torch.Tensor]`):
            The log probabilities of the latents.

    """
    images: torch.Tensor
    latents: torch.Tensor
    log_probs: torch.Tensor


class CustomStableDiffusionInstructPix2PixPipeline(StableDiffusionInstructPix2PixPipeline):
    r"""
    Pipeline for pixel-level image editing by following text instructions. Based on Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]
        - *LoRA*: [`loaders.LoraLoaderMixin.load_lora_weights`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]
    # EulerAncestralDiscreteScheduler
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder = None,
        requires_safety_checker: bool = True,
    ):
        super(CustomStableDiffusionInstructPix2PixPipeline, self).__init__(
            vae, text_encoder, tokenizer, unet, scheduler,
            safety_checker, feature_extractor, image_encoder, requires_safety_checker
        )

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def freeze_params(self):
        # Freeze all parameters in vae, unet and text_encoder
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def randn_tensor(
        shape: Union[Tuple, List],
        generator: Optional[torch.Generator] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
        layout: Optional["torch.layout"] = None,
    ):
        """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
        passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
        is always created on the CPU.
        """
        # device on which tensor is created defaults to device
        rand_device = device
        batch_size = shape[0]

        layout = layout or torch.strided
        device = device or torch.device("cpu")

        if generator is not None:
            gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
            if gen_device_type != device.type and gen_device_type == "cpu":
                rand_device = "cpu"
                if device != "mps":
                    logger.info(
                        f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                        f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                        f" slighly speed up this function by passing a generator that was created on the {device} device."
                    )
            elif gen_device_type != device.type and gen_device_type == "cuda":
                raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

        # make sure generator list of length 1 is treated like a non-list
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, list):
            shape = (1,) + shape[1:]
            latents = [
                torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
                for i in range(batch_size)
            ]
            latents = torch.cat(latents, dim=0).to(device)
        else:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

        return latents

    def scheduler_step(
    self,
    model_output: torch.FloatTensor,  # 预测的噪声 (noise_pred)
    timestep: int,                    # 当前时间步 (t)
    sample: torch.FloatTensor,        # 当前潜在变量 (latents)
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
    ) :
        """
        针对KarrasDiffusionSchedulers（以DDPMScheduler为例）的scheduler_step实现
        返回包含前一样本和log_prob的结构
        """
        scheduler = self.scheduler  # 假设self.scheduler是DDPMScheduler实例
        
        # 1. 获取alpha累积乘积
        timestep = timestep.cpu().type(torch.long)
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        # alpha_prod_t_prev = (
        #     scheduler.alphas_cumprod[timestep - 1]
        #     if timestep > 0
        #     else scheduler.one
        # )
        alpha_prod_t_prev = scheduler.alphas_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0, device=alpha_prod_t.device, dtype=alpha_prod_t.dtype)

        
        # 2. 根据预测类型计算x0和预测噪声
        if scheduler.config.prediction_type == "epsilon":
            # pred_original_sample = sample - sigma * model_output
            pred_original_sample = (sample - (1 - alpha_prod_t)**0.5 * model_output) / alpha_prod_t**0.5
            pred_epsilon = model_output
        elif scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / (1 - alpha_prod_t)**0.5
        elif scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t**0.5 * sample - (1 - alpha_prod_t)**0.5 * model_output
            pred_epsilon = alpha_prod_t**0.5 * model_output + (1 - alpha_prod_t)**0.5 * sample
        else:
            raise ValueError(f"Unsupported prediction type: {scheduler.config.prediction_type}")
        
        # 3. 计算均值（确定性部分）
        mean = alpha_prod_t_prev**0.5 * pred_original_sample + (1 - alpha_prod_t_prev)**0.5 * pred_epsilon
        
        # 4. 计算方差（根据调度器配置）
        beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * beta_t
        variance = torch.clamp(variance, min=1e-20)  # 防止数值问题
        
        # 处理不同方差类型（重点！）
        # print(scheduler.config)
        # if scheduler.config.variance_type == "fixed_small":
        #     variance = variance
        # elif scheduler.config.variance_type == "fixed_large":
        #     variance = beta_t
        # else:
        #     raise ValueError(f"Unsupported variance type: {scheduler.config.variance_type}")
        
        # 5. 生成噪声样本（如果未提供）
        if prev_sample is None:
            # 生成随机噪声（与DDIM的关键区别：使用sqrt(variance)替代eta）
             # 6. 添加随机噪声
            noise = torch.randn_like(sample, device=model_output.device, dtype=model_output.dtype)
            # print(generator)
            # noise = self.randn_tensor(
            #     model_output.shape,
            #     dtype=model_output.dtype,
            #     device=model_output.device,
            #     generator=generator,
            # )
            prev_sample = mean + variance**0.5 * noise
        # prev_sample.type(sample.dtype)
        # 6. 计算对数概率（核心公式） 
        log_prob = (
            -((prev_sample.detach() - mean)**2) / (2 * variance)  # 高斯概率密度核心项
            - torch.log(variance**0.5)                            # 标准差对数项
            - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))   # 归一化常数
        ).mean(dim=tuple(range(1, prev_sample.ndim)))             # 对空间维度取平均
        return prev_sample.type(sample.dtype), log_prob

    """
    # def scheduler_step(
    # self,
    # model_output: torch.FloatTensor,
    # timestep: int,
    # sample: torch.FloatTensor,
    # generator=None,
    # ):
    #     # calculate \alpha and \beta
    #     # timestep_tensor = torch.tensor(timestep, dtype=torch.long, device=self.scheduler.alphas_cumprod.device)
    #     timestep_tensor = torch.tensor(timestep, dtype=torch.long, device=sample.device)
    #     # alpha_prod_t = self.scheduler.alphas_cumprod[timestep_tensor]
    #     alpha_prod_t = timestep_tensor
    #     beta_prod_t = 1 - alpha_prod_t

    #     # predict the original sample sample [2 4 32 32] pred_epsilon [2 4 32 32]
    #     pred_epsilon = model_output
    #     # print(model_output.shape, model_output.device)
    #     # print(timestep, timestep.device)
    #     # print(sample.shape, sample.device)
    #     pred_original_sample = (sample - beta_prod_t ** (0.5) * pred_epsilon) / alpha_prod_t ** (0.5)

    #     # calculate variance
    #     prev_timestep = timestep - 1
    #     if prev_timestep < 0:
    #         prev_timestep = 0
    #     # Move Prev_timestep to the same device as self.scheduler.alphas_cumprod
    #     prev_timestep = torch.tensor(prev_timestep, dtype=torch.long, device=self.scheduler.alphas_cumprod.device)
    #     alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep]
    #     beta_prod_t_prev = 1 - alpha_prod_t_prev
    #     variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    #     std_dev_t = variance ** 0.5

    #     # calculate the sample mean of the previous time step
    #     prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + (1 - alpha_prod_t_prev - variance) ** 0.5 * pred_epsilon

    #     # generate samples from the previous time step
    #     variance_noise = torch.randn_like(sample, dtype=sample.dtype, device=sample.device)
    #     #     variance_noise = torch.randn(sample.shape, dtype=sample.dtype, device=sample.device, generator=generator)
    #     prev_sample = prev_sample_mean + std_dev_t * variance_noise

    #     # calculate logarithmic probability
    #     log_prob = (
    #         -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
    #         - torch.log(std_dev_t)
    #         - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi)))
    #     )
    #     log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    #     return prev_sample, log_prob
    """
    
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        last_grad_steps: int = 0,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, or tensor representing an image batch which will be repainted according to `prompt`. Can also
                accpet image latents as `image`, if passing latents directly, it will not be encoded again.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
              (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. This pipeline requires a value of at least `1`.
            image_guidance_scale (`float`, *optional*, defaults to 1.5):
                Image guidance scale is to push the generated image towards the inital image `image`. Image guidance
                scale is enabled by setting `image_guidance_scale > 1`. Higher image guidance scale encourages to
                generate images that are closely linked to the source image `image`, usually at the expense of lower
                image quality. This pipeline requires a value of at least `1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            last_grad_steps:
                The number of steps to keep gradients during training.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Check inputs
        self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance. # 10 1.5
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # 2. Encode input prompt
        with torch.no_grad():
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )

        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Image latents
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            do_classifier_free_guidance,
            generator,
        )

        # covert latent into fft space
        # image_latents_ = torch.fft.fft2(image_latents, dim=(-2, -1)).to(device)
        # image_latents = image_latents_.real
        # split_size = image.shape[0]
        # fft_shifted_imag = image_latents_[:split_size,:,:,:].imag  
        
        # print(image_latents.shape, image_latents.device) # [6 4 32 32]
        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # print(latents.shape, latents.device) # [2 4 32 32]
        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        all_latents = [latents]
        all_log_probs = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                # print(latent_model_input.shape, latent_model_input.device) #[6 4 32 32]
                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # print(latent_model_input.shape, image_latents.shape)
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)
                # print(scaled_latent_model_input.shape, scaled_latent_model_input.device) 
                # predict the noise residual  # scaled_latent_model_input: [6 8 32 32] t:[] prompt_embeds:[6 77 768]
                if i < num_inference_steps - last_grad_steps:
                    with torch.no_grad():
                        noise_pred = self.unet(
                            scaled_latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False
                        )[0]
                else:
                    noise_pred = self.unet(
                        scaled_latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False
                    )[0]

                # print(noise_pred.shape, noise_pred.device) #[6 4 32 32]
                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. So we need to compute the
                # predicted_original_sample here if we are using a karras style scheduler.
                if scheduler_is_in_sigma_space: # EulerAncestralDiscreteScheduler
                    step_index = (self.scheduler.timesteps == t).nonzero()[0].item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred
                # print(noise_pred.shape, noise_pred.device) #[6 4 32 32]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )
                # print(noise_pred.shape, noise_pred.device) #[2 4 32 32]
                # Hack:
                # For karras style schedulers the model does classifer free guidance using the
                # predicted_original_sample instead of the noise_pred. But the scheduler.step function
                # expects the noise_pred and computes the predicted_original_sample internally. So we
                # need to overwrite the noise_pred here such that the value of the computed
                # predicted_original_sample is correct.
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - latents) / (-sigma)

                # compute the previous noisy sample x_t -> x_t-1
                _, log_prob = self.scheduler_step(noise_pred, t, latents, generator=generator)
                # ori denoise pre_latents
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                # print(latents.shape, log_prob)
                # print(noise_pred.shape, noise_pred.device)
                # print(t, t.device)
                # print(latents.shape, latents.device)
                # _, log_prob = self.scheduler_step(noise_pred, t, latents, generator=generator)
                # latents = prev_sample

                all_latents.append(latents)
                all_log_probs.append(log_prob)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            # inv latent space
            # fft_modified = torch.complex(latents.real, fft_shifted_imag).to(device)
            # latents = torch.fft.ifft2(fft_modified).real
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            # print(image.shape, image.device) #[2 3 256 256]
        else:
            image = latents
        # print(all_latents[len(timesteps)-1].shape, image.device) #[2 4 32 32]
        # print(all_log_probs[len(timesteps)-1], image.device) #[2]
        # return image, DDPOPipelineOutput(image, all_latents[len(timesteps)-1], all_log_probs[len(timesteps)-1])
        return image, image_latents, all_latents, all_log_probs