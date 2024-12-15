#!/bin/bash -l
#SBATCH -A plgdiffusion-gpu-gh200
#SBATCH -p plgrid-gpu-gh200
#SBATCH -t 40:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:4
#SBATCH --mem 420G
#SBATCH --cpus-per-task=24
#SBATCH --nodes 1
#SBATCH -o slurm_out/sample_mag_max_light_order0_seed0.log
#SBATCH --job-name=sample_mag_max_light_order0_seed0

module load ML-bundle/24.06a

cd /net/scratch/hscra/plgrid/plglukaszst/projects/lora-cl
source ./venv/bin/activate

export PYTHONPATH=$PWD

accelerate launch --num-processes 4 ./eval/sample_images.py --models_path ./models/mag_max_light/seed_5_object/seed_0_order --method_name mag_max_light --task_type object