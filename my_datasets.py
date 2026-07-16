import json
from glob import glob
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def collate_fn(examples):
    image = torch.stack([example["image"] for example in examples])
    image = image.to(memory_format=torch.contiguous_format).float()
    prompt = [example["prompt"] for example in examples]
    refac_prompt = [example["refac_prompt"] for example in examples]
    return {"image": image, "prompt": prompt, "refac_prompt": refac_prompt}

class InjectDataset(Dataset):
    def __init__(self, instance_data_root, image_size) -> None:
        super().__init__()
        self.image_size = image_size
        
        # load insp2p datasets
        self.img_path = os.path.join(instance_data_root, "instructpix2pix_20000")

        # load OpenImage datasets
        # self.img_path = os.path.join(instance_data_root, "openimage_ori_20000")

        # load multi image 
        # self.img_path = os.path.join(instance_data_root, "muti_edited_200")
        
        # prompt path
        self.edit_prompt_path = os.path.join(instance_data_root, "edit_prompt_20000.txt")
        self.refac_edit_prompt_path = os.path.join(instance_data_root, "refactor_edit_prompt_20000.txt")

        # multi-turn editing prompt
        # self.edit_prompt_path = os.path.join(instance_data_root, "muti_edit_prompt_200.txt")
        # self.refac_edit_prompt_path = os.path.join(instance_data_root, "muti_edit_prompt_200.txt")

        self.all_img_name = os.listdir(self.img_path)
        self.all_img_name.sort(key=lambda x: int(x[:-4]))

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
            ]
        )

        # transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),

        # get edit prompt
        self.edit_prompts = []
        with open(self.edit_prompt_path, "r") as fileHandler:
            lines = fileHandler.readlines()
        for line in lines:
            line = line.replace("\n", "")
            self.edit_prompts.append(line)

        # get refac prompt 
        self.refac_edit_prompts = []
        with open(self.refac_edit_prompt_path, "r") as fileHandler:
            lines = fileHandler.readlines()
        for line in lines:
            line = line.replace("\n", "")
            self.refac_edit_prompts.append(line)

    def __len__(self):
        return len(self.all_img_name)

    def __getitem__(self, index):
        # get img_data
        img_name = self.all_img_name[index]
        real_img = Image.open(os.path.join(self.img_path, img_name))
        real_img = real_img.convert("RGB").resize((self.image_size, self.image_size))
        real_img = np.array(real_img).transpose(2, 0, 1)
        real_img = torch.tensor(real_img)
        real_img = 2 * (real_img / 255) - 1
        images = self.train_transforms(real_img)

        # get edit_prompt
        prompt = self.edit_prompts[index]
        # get refac prompt
        refac_prompt = self.refac_edit_prompts[index]
        return {"image": images, "prompt": prompt, "refac_prompt": refac_prompt}