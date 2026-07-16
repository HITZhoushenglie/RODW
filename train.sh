# Script Name: train.sh
# Description: launch the training process
# Author: Shenglie Zhou
# Date: 2025-7-18

OUTPUT_DIR="/home/ubuntu/code/RobustWide-master/train_results"

# 128x128
# OUTPUT_DIR="/home/ubuntu/data0/model/RODW/train_results/resolution/rodw"
WM_MODEL_CONFIG="/home/ubuntu/code/RobustWide-master/rein_learning/config.yaml"

# insp2p 
DATA_DIR="/home/ubuntu/datasets/robust_wide/data"

# openimage
# DATA_DIR="/home/ubuntu/datasets/robust_wide/data/OpenImage"

# Computational Complexity Datasets
# DATA_DIR="/home/ubuntu/datasets/robust_wide/data"

#######################DEFAULT_SETTING#######################
IMAGE_SIZE=192
LEARNING_RATE=1e-3
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
SINGLE_EPOCH_TRAIN_STEPS=200000
MAX_TRAIN_STEPS=200000
LR_WARMUP_STEPS=400
DECODER_WEIGHT=0.1
ENC_LATENT_LOSS=0.001
OPT_REIN_LOSS=0.1
LAST_GRAD_STEPS=3
#######################BEST_SETTING############################

# accelerate config

accelerate launch --main_process_port=6666 rein_learning/train_new.py \
  --train_data_dir $DATA_DIR \
  --wm_model_config $WM_MODEL_CONFIG \
  --output_dir $OUTPUT_DIR \
  --image_size $IMAGE_SIZE \
  --batch_size $BATCH_SIZE \
  --single_epoch_train_steps $SINGLE_EPOCH_TRAIN_STEPS \
  --max_train_steps $MAX_TRAIN_STEPS \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler "cosine" \
  --lr_warmup_steps $LR_WARMUP_STEPS \
  --log_steps 10 \
  --save_steps 1000 \
  --decoder_weight $DECODER_WEIGHT \
  --last_grad_steps $LAST_GRAD_STEPS \
  --enc_latent_weight $ENC_LATENT_LOSS \
  --opt_rein_weight $OPT_REIN_LOSS\
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS