#!/bin/bash
#SBATCH -A plggenerativepw2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 4:30:00
#SBATCH --ntasks 4
#SBATCH --gres gpu:4
#SBATCH --mem 480G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH -o slurm_out/sample_mag_max_light_order0_seed5.log
#SBATCH --job-name=sample_mag_max_light_order0_seed5

# module load GCC/12.3.0
# module load NVHPC/24.5-CUDA-12.4.0
# module load OpenMPI/5.0.3

module load Miniconda3/23.3.1-0
eval "$(conda shell.bash hook)"
conda activate lora

# cd /net/scratch/hscra/plgrid/plglukaszst/projects/lora-cl
# source ./venv/bin/activate

export PYTHONPATH=$PWD
accelerate launch --num-processes 4 ./eval/sample_images.py --models_path ./models/mag_max_light/seed_0_object/seed_0_order --method_name mag_max_light --task_type object