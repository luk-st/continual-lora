#!/bin/sh

export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export ACCELERATE_PATH="/net/tscratch/people/plglukaszst/envs/lora/bin/accelerate"
export RANK=64
export SEED=42

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="./models/seed_${SEED}/wolf_plushie_sd1"
export INSTANCE_DIR="./data/dreambooth/dataset/wolf_plushie"
export PROMPT="a photo of sks stuffed animal"
export VALID_PROMPT="a sks stuffed animal riding a bicycle"

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
  --max_train_steps=500 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=125 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model

wait

export MODEL_NAME="./models/seed_${SEED}/wolf_plushie_sd1"
export OUTPUT_DIR="./models/seed_${SEED}/backpack_sd2"
export INSTANCE_DIR="./data/dreambooth/dataset/backpack"
export PROMPT="a photo of zwz backpack"
export VALID_PROMPT="a zwz backpack in the mountains"

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
  --max_train_steps=500 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=125 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model

wait

export MODEL_NAME="./seed_${SEED}/models/backpack_sd2"
export OUTPUT_DIR="./seed_${SEED}/models/dog6_sd3"
export INSTANCE_DIR="./data/dreambooth/dataset/dog6"
export PROMPT="a photo of sbu dog"
export VALID_PROMPT="a sbu dog riding a bicycle"

wait

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
  --max_train_steps=500 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=125 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model

wait

export MODEL_NAME="./seed_${SEED}/models/dog6_sd3"
export OUTPUT_DIR="./seed_${SEED}/models/candle_sd4_2"
export INSTANCE_DIR="./data/dreambooth/dataset/candle"
export PROMPT="a photo of uwu candle"
export VALID_PROMPT="a uwu candle in the in the gym"

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
  --max_train_steps=500 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=125 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model

wait

export MODEL_NAME="./seed_${SEED}/models/candle_sd4"
export OUTPUT_DIR="./seed_${SEED}/models/cat2_sd5"
export INSTANCE_DIR="./data/dreambooth/dataset/cat2"
export PROMPT="a photo of pdw cat"
export VALID_PROMPT="a pdw cat in the prison"
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
  --max_train_steps=500 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=125 \
  --seed=$SEED \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --save_whole_model