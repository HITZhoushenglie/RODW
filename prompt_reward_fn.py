import textwrap
from collections import defaultdict

import torch
from rein_learning.rewards import MultiReward
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from prompt import HPSPrompt, ImageRewardPrompt, ImagenetAnimalPrompts, SinglePrompt


def reward_fn(args):
    reward_model = MultiReward(args.reward_list)
    return reward_model 


def prompt_fn(args):
    if args.prompt == 'animal':
        prompts = ImagenetAnimalPrompts() 
    elif args.prompt == 'single':
        prompts = SinglePrompt('None')
    elif args.prompt == 'imagereward':
        prompts = ImageRewardPrompt()
    elif args.prompt == 'hps':
        prompts = HPSPrompt(phase='train')
    
    def fn(index=None):
        return prompts[index], None
    
    return fn


def write_text_on_image_tensor(image_tensor, text, position=(10, 10), font_size=20, color=(255, 0, 0)):
    # Convert the PyTorch tensor to a PIL image
    transform_to_pil = transforms.ToPILImage()
    img = transform_to_pil(image_tensor)

    txt_blk_height = 100
    new_img = Image.new("RGB", (img.width, img.height + txt_blk_height), "white")
    new_img.paste(img, (0, 0))
    
    # Draw text on the image
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default(font_size)

    # Wrap the text
    wrapped_text = textwrap.fill(text, width=50)

    # Calculate the position for the text
    text_x = 10
    text_y = img.height 

    # Add text to the image line by line
    for line in wrapped_text.split('\n'):
        draw.text((text_x, text_y), line, font=font, fill=color)
        text_y += 20 

    # draw.text((10, 10 + img.height), text, font=font, fill=color)
    
    # Convert the PIL image back to a PyTorch tensor
    transform_to_tensor = transforms.ToTensor()
    image_tensor_with_text = transform_to_tensor(new_img)
    
    return image_tensor_with_text

def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = defaultdict(list) 
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()

        image = write_text_on_image_tensor(image, f'{reward:.2f} | {prompt}')

        # result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()
        if 'images' in result:
            result['images'] = torch.cat([result['images'], image.unsqueeze(0).float()], dim=0)
            # result['prompts'].append(prompt)
        else:
            result['images'] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )