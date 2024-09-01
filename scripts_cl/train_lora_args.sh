#!/bin/sh

export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"
export ACCELERATE_PATH="accelerate"
export RANK=64
export SEED=$(($1))
export WORK_DIR=$2
export EXPERIMENT_NAME=$3
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"

echo "Seed: $SEED, Work dir: $WORK_DIR"
echo "Model name: $MODEL_NAME"
echo "Experiment name: $EXPERIMENT_NAME"

shift 3

for arg in "$@"
do
  IFS="," read -r index train_prompt valid_prompt dataset_dir <<< "$arg"

  echo "TASK $index"
  echo "Train prompt: $train_prompt Validation prompt: $valid_prompt, Dataset dir: $dataset_dir, Model name: $MODEL_NAME"

  export OUTPUT_DIR="${WORK_DIR}/${index}"
  export INSTANCE_DIR=$dataset_dir
  export PROMPT=$train_prompt
  export VALID_PROMPT=$valid_prompt
  export MODEL_NAME=$MODEL_NAME

  $ACCELERATE_PATH launch lora/train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --pretrained_vae_model_name_or_path=$VAE_PATH \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="${PROMPT}"\
    --rank=$RANK \
    --resolution=1024 \
    --train_batch_size=1 \
    --learning_rate=5e-5 \
    --report_to="wandb" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --validation_prompt="${VALID_PROMPT}" \
    --validation_epochs=250 \
    --seed=$SEED \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --use_8bit_adam \
    --experiment_name=$EXPERIMENT_NAME \

  export MODEL_NAME=$OUTPUT_DIR
  wait
done

