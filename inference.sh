# Script Name: rein_inference.sh
# Description: generate the watermarked image
# Author: Shenglie Zhou
# Date: 2025-8-25 / 2026-1-27

# model
# insp2p+rein+blip: /home/ubuntu/data0/model/RODW/train_results/2025-06-07T00-35-50

# datasets
# insp2p: /home/ubuntu/datasets/robust_wide/data/muti_edited_200
# insp2p: /home/ubuntu/data0/datasets/instructpix2pix/data/instructpix2pix_1200

# test prompt
# insp2p: /home/ubuntu/datasets/robust_wide/data/muti_edit_prompt_200.txt
# insp2p: /home/ubuntu/data0/datasets/instructpix2pix/data/edit_prompt_1200.txt

python rein_learning/rein_inference.py \
  --ckpt_dir '/home/ubuntu/data0/model/RODW/train_results/2025-06-07T00-35-50' \
  --image_file '/home/ubuntu/data0/datasets/instructpix2pix/data/instructpix2pix_1200' \
  --edit_prompt_path '/home/ubuntu/data0/datasets/instructpix2pix/data/edit_prompt_1200.txt' \
  --output_dir '/home/ubuntu/code/RobustWide-master/inference_results' \
  --image_size 256 \
  --batch_size 1
