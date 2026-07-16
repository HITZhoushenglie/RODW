# Script Name: rein_inference.sh
# Description: generate the watermarked image
# Author: Shenglie Zhou
# Date: 2025-8-25 / 2026-1-27

# 128 x 128
# ori + rein + blip 
# insp2p+rein+blip: /home/ubuntu/code/RobustWide-master/train_results/2026-03-23T06-11-38

# 192 x 192
# ori + rein + blip 
# insp2p+dsg+rlo: /home/ubuntu/code/RobustWide-master/train_results/2026-04-11T13-04-06

# 256 x 256 128 bits
# insp2p+ori+rein: /home/ubuntu/code/RobustWide-master/train_results/2026-04-11T12-37-42

# 256 x 256
# ori + rein (llm + rlo)
# insp2p+ori+rein: /home/ubuntu/data0/model/RODW/train_results/2025-08-26T01-07-32
# openimage+ori+rein: /home/ubuntu/data0/model/RODW/train_results/2025-08-20T01-40-27

# ori + rein (vlm + rlo)
# insp2p+ori+rein: /home/ubuntu/data0/model/RODW/train_results/single_style/vlm+rlo/2026-02-14T23-20-56

# ori + rein + blip 
# insp2p+rein+blip: /home/ubuntu/data0/model/RODW/train_results/2025-06-07T00-35-50
# openimage+rein+blip: /home/ubuntu/data0/model/RODW/train_results/2025-07-18T13-07-43

# datasets
# insp2p: /home/ubuntu/datasets/robust_wide/data/muti_edited_200
# openiamge: /home/ubuntu/datasets/robust_wide/data/OpenImage/openimage_200
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