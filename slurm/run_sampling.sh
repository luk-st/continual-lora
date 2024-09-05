#!/bin/bash
#SBATCH -A plgdiffusion-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -t 40:00:00
#SBATCH --ntasks 4
#SBATCH --gres gpu:4
#SBATCH --mem 480G
#SBATCH --cpus-per-task=72
#SBATCH --nodes 1
#SBATCH -o slurm_out/sample_mag_max_light_order0_seed0.log
#SBATCH --job-name=sample_mag_max_light_order0_seed0

module load openmpi/4.1.1-gcc-11.2.0
module load Miniconda3/23.3.1-0

eval "$(conda shell.bash hook)"
conda activate lora

cd /storage1/lukasz/projects/lora-cl

export PYTHONPATH=$PWD

accelerate launch --num-processes 4 ./eval/sample_images.py 

python3 scripts_cl/train_object_order.py --models_path ./models/mag_max_light/seed_0_object/seed_0_order --method_name mag_max_light --task_type object