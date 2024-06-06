
# export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export MODEL_NAME="kzaleskaa/vector-graphics"
export ACCELERATE_PATH="/net/tscratch/people/plgkzaleska/envs/lora/bin/accelerate"

# for style
export OUTPUT_DIR="/net/tscratch/people/plgkzaleska/ziplora-analysis/models/3d_rendering"
export INSTANCE_DIR="/net/tscratch/people/plgkzaleska/ziplora-analysis/data/styledrop/3d_rendering"
export PROMPT="a woman in skn style"
export VALID_PROMPT="a cat in skn style"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"


$ACCELERATE_PATH launch lora/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --instance_prompt="a woman in skn style" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="a dog in skn style" \
  --validation_epochs=25 \
  --seed="0" \
  --push_to_hub