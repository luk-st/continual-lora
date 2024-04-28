export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export ACCELERATE_PATH="/net/tscratch/people/plglukaszst/envs/lora/bin/accelerate"

# for subject
export LORA_PATH="lukasz-staniszewski/dog6_subject"
export INSTANCE_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/data/dreambooth/dataset/dog6"
export PROMPT="a sbu dog"

# for style
export LORA_PATH2="lukasz-staniszewski/game_style"
export INSTANCE_DIR2="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/data/styledrop/game"
export PROMPT2="a woman in szn style"

# general
export OUTPUT_DIR="/net/tscratch/people/plglukaszst/projects/ziplora-analysis/models/ziplora-dog6-game"
export VALID_PROMPT="a sbu dog in szn style"


$ACCELERATE_PATH launch train_dreambooth_ziplora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --lora_name_or_path=$LORA_PATH \
  --instance_prompt="${PROMPT}" \
  --instance_data_dir=$INSTANCE_DIR \
  --lora_name_or_path_2=$LORA_PATH2 \
  --instance_prompt_2="${PROMPT2}" \
  --instance_data_dir_2=$INSTANCE_DIR2 \
  --resolution=1024 \
  --init_merger_value=0.8 \
  --init_merger_value_2=1.2 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --similarity_lambda=0.01 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=100 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=10 \
  --seed="0" \
  --report_to="wandb" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --quick_release
