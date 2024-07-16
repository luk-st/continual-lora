#!/bin/sh

# Configuration
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export ACCELERATE_PATH="/net/tscratch/people/plglukaszst/envs/lora/bin/accelerate"

export RANK=64
export SEED=42
export WORKING_PATH="./models/seed_${SEED}_style"

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="${WORKING_PATH}/watercolor_painting_style_sd1"
export INSTANCE_DIR="./data/styledrop/watercolor_painting_style"
export PROMPT="a cat in skn style"
export VALID_PROMPT="a woman in skn style"

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
  --max_train_steps=250 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=75 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model

wait

export MODEL_NAME="${WORKING_PATH}/watercolor_painting_style_sd1"
export OUTPUT_DIR="${WORKING_PATH}/oil_painting_style_sd2"
export INSTANCE_DIR="./data/styledrop/oil_painting_style"
export PROMPT="a village in zwz style"
export VALID_PROMPT="a dog in zwz style"

$ACCELERATE_PATH launch lora/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=$RANK \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=75 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model

wait

export MODEL_NAME="${WORKING_PATH}/oil_painting_style_sd2"
export OUTPUT_DIR="${WORKING_PATH}/flat_cartoon_illustration_style_sd3"
export INSTANCE_DIR="./data/styledrop/flat_cartoon_illustration"
export PROMPT="a woman working on a laptop in fci style"
export VALID_PROMPT="a cat sitting on table in fci style"

$ACCELERATE_PATH launch lora/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=$RANK \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=75 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model

wait

export MODEL_NAME="${WORKING_PATH}/flat_cartoon_illustration_style_sd3"
export OUTPUT_DIR="${WORKING_PATH}/abstract_rainbow_colored_style_sd4"
export INSTANCE_DIR="./data/styledrop/abstract_rainbow_colored_flowing_smoke_wave"
export PROMPT="a wave in rcf style"
export VALID_PROMPT="a house in rcf style"

$ACCELERATE_PATH launch lora/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=$RANK \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=75 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model

wait

export MODEL_NAME="${WORKING_PATH}/abstract_rainbow_colored_style_sd4"
export OUTPUT_DIR="${WORKING_PATH}/sticker_style_sd5"
export INSTANCE_DIR="./data/styledrop/sticker"
export PROMPT="a Christmas tree in cts style"
export VALID_PROMPT="a duck in cts style"

$ACCELERATE_PATH launch lora/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=$RANK \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=75 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model
