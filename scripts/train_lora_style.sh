
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export ACCELERATE_PATH="/net/tscratch/people/plglukaszst/envs/lora/bin/accelerate"

# for style
export OUTPUT_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/models/game_style"
export INSTANCE_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/data/styledrop/game"
export PROMPT="a woman in szn style"
export VALID_PROMPT="a cat in szn style"


$ACCELERATE_PATH launch lora/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=64 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=50 \
  --seed="0" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --push_to_hub