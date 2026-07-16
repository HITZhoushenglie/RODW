from xmlrpc.client import INVALID_XMLRPC
import time
import json
import os
import random
import numpy
import torch
from torchvision import transforms
import argparse
from PIL import Image
from omegaconf import OmegaConf
from kornia.metrics import psnr, ssim

import sys
sys.path.append('/home/ubuntu/code/RobustWide-master/')

from rein_learning.model import WatermarkModel
from torch.utils.data import DataLoader
# import sys
# sys.path.append('/home/ubuntu/code/Robust-Wide-master/')
# from rein_learning.my_datasets import InjectDataset, collate_fn


from utils import (
    normalize,
    denormalize,
    decoded_message_error_rate,
    save_image_for_tensor,
)


import datetime
import csv


class TimeStampManager:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            # init timesteps
            cls.start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # cls.log_dir = os.path.join("./inference_results", cls.start_time)
            
            # ori + rein + blip
            cls.log_dir = os.path.join("./inference_results/robustness/rodw", cls.start_time)
            
            # ori + rein  
            # cls.log_dir = os.path.join("./inference_results/Ablation/openimage/ori+rein", cls.start_time)

            # vlm+rlo
            # cls.log_dir = os.path.join("./inference_results/robustness/vlm+rlo", cls.start_time)

            # llm+rlo
            # cls.log_dir = os.path.join("./inference_results/robustness/llm+rlo", cls.start_time)

            # test image resolution
            # cls.log_dir = os.path.join("./inference_results/robustness/128x128", cls.start_time)

            # test image resolution
            # cls.log_dir = os.path.join("./inference_results/robustness/192x192", cls.start_time)

            # 256x256 128bits
            # cls.log_dir = os.path.join("./inference_results/robustness/256x256/128bits", cls.start_time)

            cls.log_dir_img = os.path.join(cls.log_dir, "./image")
            cls.log_dir_wm = os.path.join(cls.log_dir, "./wm_image")
            cls.log_dir_edit = os.path.join(cls.log_dir, "./edit_image")
            cls.log_dir_res = os.path.join(cls.log_dir, "./res_image")

            os.makedirs(cls.log_dir, exist_ok=True)
            os.makedirs(cls.log_dir_img, exist_ok=True)
            os.makedirs(cls.log_dir_wm, exist_ok=True)
            os.makedirs(cls.log_dir_edit, exist_ok=True)
            os.makedirs(cls.log_dir_res, exist_ok=True)
        return cls._instance


def get_log_path(filename):
    manager = TimeStampManager() 
    image_name = os.path.join(manager.log_dir_img, f"{filename}_orig.png")
    wm_image_name = os.path.join(manager.log_dir_wm, f"{filename}_wm.png")
    edit_image_name = os.path.join(manager.log_dir_edit, f"{filename}_edit.png")
    res_image_name = os.path.join(manager.log_dir_res, f"{filename}_res.png")

    return image_name, wm_image_name, edit_image_name, res_image_name

def get_csv_path(filename):
    manager = TimeStampManager()
    return os.path.join(manager.log_dir, f"{filename}.csv")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_wm_model(ckpt_dir, wm_model_config_path=None):
    if wm_model_config_path is None:
        wm_model_config_path = os.path.join(os.path.join(ckpt_dir, "wm_model_config.yaml"))
    wm_model_config = OmegaConf.load(wm_model_config_path)
    message_length = wm_model_config["wm_enc_config"]["message_length"]
    model = WatermarkModel(**wm_model_config)
    model_ckpt = torch.load(os.path.join(ckpt_dir, "wm_model.ckpt"), map_location='cpu')
    # model.load_state_dict(model_ckpt)
    model.load_state_dict(model_ckpt, strict=False) 
    model.eval()
    return model, message_length

@torch.no_grad()
def main(current_step, ckpt_dir, instance_image_file, edit_prompt_path, image_size, batch_size, wm_data_dir, cur_avg_psnr, cur_avg_ssim, cur_avg_ber, save_best, device):
    size = 256
    transform_list = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    image_transforms = transforms.Compose(transform_list)
    wm_model, message_length = load_wm_model(ckpt_dir=ckpt_dir)
    wm_model = wm_model.to(device)
    # os.makedirs(wm_data_dir, exist_ok=True)

    # not change
    from custom_pipe import CustomStableDiffusionInstructPix2PixPipeline
    weight_dtype = torch.float32
    pipe = CustomStableDiffusionInstructPix2PixPipeline.from_pretrained(
        "./timbrooks/instruct-pix2pix",
        torch_dtype=weight_dtype,
        local_files_only=True
    ).to(device)

    image_name = os.listdir(instance_image_file)
    image_name.sort(key=lambda x: int(x[:-4]))
    psnr_value, ssim_value, ber_ = 0.0, 0.0, 0.0
    
    # 1 get original prompt
    # test_file_path = "/home/ubuntu/code/Robust-Wide-master/data/edit_prompt_test_1200.json"
    # prompt_texts = []
    # with open(test_file_path) as json_file:
    #     data = json.load(open(test_file_path))
    #     for p in data:
    #         prompt_texts.append(p['prompt'])
    # num = len(prompt_texts)

    # 2 get original prompt
    # Insp2p
    test_file_path = edit_prompt_path
    # test_file_path = "/home/ubuntu/datasets/robust_wide/data/muti_edit_prompt_200.txt"
    # OpenImage
    # test_file_path = "/home/ubuntu/datasets/robust_wide/data/OpenImage/edit_prpmpt_test_200.txt"
    prompt_texts = []
    try:
        with open(test_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                items = line.strip().split(',')
                items = [item.strip() for item in items if item.strip()]
                prompt_texts.append(items)
        num = len(prompt_texts)
        # ori + rein + blip
        # edit_nums = len(prompt_texts[0]) - 0 # Default editing times = 4, 3, 2, 1, 0 -> 1, 2, 3, 4, 5
        # ori + rein 
        edit_nums = len(prompt_texts[0]) - 4
        print(num, edit_nums)
        
    except FileNotFoundError:
        print(f"错误：文件 {test_file_path} 未找到")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []

    global_step = 0
    
    # init time
    encode_time, decode_time, edit_time = 0.0, 0.0, 0.0

    for idx in range(num):
        global_step += 1
        image_real_name = os.path.join(instance_image_file, image_name[idx])
        message = torch.randint(0, 2, size=(1, message_length)).float().to(device)
        instance_image = Image.open(image_real_name).convert("RGB")
        image = image_transforms(instance_image).unsqueeze(0).to(device)
        
        # add watermark
        start_encode_time = time.time()
        wm_image = wm_model.encoder(image, message)
        end_encode_time = time.time()
        diff_encode_time = end_encode_time - start_encode_time
        # decoded_message = wm_model.decoder(wm_image)
        
        start_edit_time = time.time()
        # one edit
        generated_image, _, _, _ = pipe(
                prompt_texts[idx][0], image=wm_image, num_images_per_prompt=1, num_inference_steps=20,
                guidance_scale=10, image_guidance_scale=1.5, last_grad_steps=3,
                output_type="pt",
            )

        # muti edit
        for idy in range(1, edit_nums):
            generated_image, _, _, _ = pipe(
                prompt_texts[idx][idy], image=generated_image, num_images_per_prompt=1, num_inference_steps=20,
                guidance_scale=10, image_guidance_scale=1.5, last_grad_steps=3,
                output_type="pt",
            )

        end_edit_time = time.time()
        diff_edit_time = end_edit_time - start_edit_time
        print('\nEditing time:', diff_edit_time, 's', '\n (Note that please execute multiple times to get the average time)\n')

        # extract watermark
        start_decode_time = time.time()
        decoded_message = wm_model.decoder(generated_image)
        end_decode_time = time.time()
        diff_decode_time = end_decode_time - start_decode_time
        print('\nDecode time:', diff_decode_time, 's', '\n (Note that please execute multiple times to get the average time)\n')


        # img_name = instance_image_file.split("/")[-1].split(".")[0]
        residual = wm_image - image
        residual_abs = torch.abs(residual)
        residual_abs_max = torch.max(residual_abs).item()
        residual_abs_min = torch.min(residual_abs).item()
        residual_image = normalize((residual_abs - residual_abs_min) / (residual_abs_max - residual_abs_min))
        ber = decoded_message_error_rate(message[0], decoded_message[0])

        # save best results
        # if save_best:
        ori_image_name, wm_image_name, edit_image_name, res_image_name = get_log_path(f"{global_step}")
        save_image_for_tensor(image[0], ori_image_name)
        save_image_for_tensor(wm_image[0], wm_image_name)
        save_image_for_tensor(generated_image[0], edit_image_name)
        save_image_for_tensor(residual_image[0], res_image_name)

        # caculate mean
        psnr_value += psnr(denormalize(wm_image), denormalize(image), 1)
        ssim_value += torch.mean(ssim(denormalize(wm_image), denormalize(image), window_size=5))
        ber_ += ber

        # caculate time
        encode_time += diff_encode_time 
        edit_time += diff_edit_time
        decode_time += diff_decode_time

    psnr_value = psnr_value / num
    ssim_value = ssim_value / num
    ber = ber_ / num
    # caculate time
    encode_time += diff_encode_time 
    edit_time += diff_edit_time
    decode_time += diff_decode_time
    # print(f"psnr: {psnr_value}, ssim: {ssim_value}, ber: {ber}")
    print(f"psnr_avg: {psnr_value}, ssim_avg: {ssim_value}, ber_avg: {ber}, encode_time_avg: {encode_time}, edit_time_avg: {edit_time}, decode_time_avg: {decode_time}")

    # ori + rein + blip
    # csv_file = get_csv_path(f"rodw_insp2p_result_edit_2_all")
    # csv_file_best = get_csv_path(f"rodw_insp2p_result_edit_2_best")

    csv_file = get_csv_path(f"rodw_insp2p_result_edit_2_all")
    csv_file_best = get_csv_path(f"rodw_insp2p_result_edit_2_best")
    

    # ori + rein
    # csv_file = get_csv_path(f"ori_rein_insp2p_result_edit_1_all")
    # csv_file_best = get_csv_path(f"ori_rein_insp2p_result_edit_1_best")

    # vlm+rlo
    # csv_file = get_csv_path(f"vlm_rlo_insp2p_result_edit_2_all")
    # csv_file_best = get_csv_path(f"vlm_rlo_insp2p_result_edit_2_best")

    # llm+rlo
    # csv_file = get_csv_path(f"llm_rlo_insp2p_result_edit_2_all")
    # csv_file_best = get_csv_path(f"llm_rlo_insp2p_result_edit_2_best")

    # 128x128
    # csv_file = get_csv_path(f"128x128_insp2p_result_edit_2_all")
    # csv_file_best = get_csv_path(f"128x128_insp2p_result_edit_2_best")

    # 192x192
    # csv_file = get_csv_path(f"192x192_insp2p_result_edit_3_all")
    # csv_file_best = get_csv_path(f"192x192_insp2p_result_edit_3_best")

    #256x256 128bits 
    # csv_file = get_csv_path(f"128_256x256_insp2p_result_edit_3_all")
    # csv_file_best = get_csv_path(f"128_256x256_insp2p_result_edit_3_best")

    file_exists = os.path.exists(csv_file)
    file_exists_best = os.path.exists(csv_file_best)
    
    # # save all value
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists or os.stat(csv_file).st_size == 0:
            # writer.writerow(["current_step", "psnr", "ssim", "ber"])
            writer.writerow(["current_step", "psnr", "ssim", "ber", "encode_time", "edit_time", "decode_time"])
                        
        # write into csv
        # writer.writerow([current_step, psnr_value.item(), ssim_value.item(), ber])
        writer.writerow([current_step, psnr_value.item(), ssim_value.item(), ber, encode_time, edit_time, decode_time])

    # search best value
    with open(csv_file_best, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists_best or os.stat(csv_file).st_size == 0:
            writer.writerow(["current_step", "psnr_best", "ssim_best", "ber_best", "encode_time", "edit_time", "decode_time"])
            writer.writerow([current_step, cur_avg_psnr, cur_avg_ssim, cur_avg_ber, encode_time, edit_time, decode_time])

    if psnr_value > cur_avg_psnr and ssim_value > cur_avg_ssim and ber < cur_avg_ber:
        cur_avg_psnr = psnr_value
        cur_avg_ssim = ssim_value
        cur_avg_ber = ber 
        save_best = True
        with open(csv_file_best, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_step, cur_avg_psnr.item(), cur_avg_ssim.item(), cur_avg_ber, encode_time, edit_time, decode_time])

        return cur_avg_psnr, cur_avg_ssim, cur_avg_ber, save_best
    else:
        return cur_avg_psnr, cur_avg_ssim, cur_avg_ber, save_best


if __name__ == "__main__":
    # set_seed(2025)
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--image_file', type=str)
    parser.add_argument('--edit_prompt_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--cur_avg_psnr", default=37.0, type=float)
    parser.add_argument("--cur_avg_ssim", default=0.95, type=float)
    parser.add_argument("--cur_avg_ber", default=0.5, type=float)
    parser.add_argument("--save_best", default=False, type=bool)
    args = parser.parse_args()
    
    # test all result
    # all_step = 200000
    # init_step = 40000

    # test best result (llm+rlo 1-round)
    # all_step = 121000
    # init_step = 120000

    # test best result (llm+rlo 1-round)
    # all_step = 131000
    # init_step = 130000

    # test best result (dsg+rlo 2-round)
    # all_step = 128000
    # init_step = 127000

    # test rodw
    all_step = 149000
    init_step = 148000

    for current_step in range(init_step, all_step, 1000):
        model_ckpt_dir =  args.ckpt_dir + f"/step{current_step}" 
        print(model_ckpt_dir)
        args.cur_avg_psnr, args.cur_avg_ssim, args.cur_avg_ber, args.save_best = main(
            current_step,
            ckpt_dir=model_ckpt_dir, 
            instance_image_file=args.image_file,
            edit_prompt_path=args.edit_prompt_path,
            image_size=args.image_size,
            batch_size=args.batch_size,
            wm_data_dir=args.output_dir, 
            cur_avg_psnr=args.cur_avg_psnr,
            cur_avg_ssim=args.cur_avg_ssim,
            cur_avg_ber=args.cur_avg_ber,
            save_best=args.save_best,
            device='cuda:4',
        )
