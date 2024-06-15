#!/bin/sh

# export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# export MODEL_NAME="./models/wolf_plushie_sd"
export MODEL_NAME="./models/dog6_sd"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export ACCELERATE_PATH="/net/tscratch/people/plglukaszst/envs/lora/bin/accelerate"

# export OUTPUT_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/models/wolf_plushie_sd"
# export OUTPUT_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/models/dog6_sd"
export OUTPUT_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/models/backpack_sd"
# export INSTANCE_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/data/dreambooth/dataset/wolf_plushie"
# export INSTANCE_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/data/dreambooth/dataset/dog6"
export INSTANCE_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/data/dreambooth/dataset/backpack"
# export PROMPT="a photo of sks stuffed animal"
# export PROMPT="a photo of sbu dog"
export PROMPT="a photo of zwz backpack"
# export VALID_PROMPT="a sks stuffed animal riding a bicycle"
# export VALID_PROMPT="a sbu dog riding a bicycle"
export VALID_PROMPT="a zwz backpack in the mountains"
export RANK=64

$ACCELERATE_PATH launch lora/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}"\
  --rank=$RANK \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=25 \
  --seed="0" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model