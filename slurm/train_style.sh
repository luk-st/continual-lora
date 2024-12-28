#!/bin/bash
#SBATCH -A plgzzsn2024-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH -o slurm_out/st_mm_ord0_s5.log
#SBATCH --job-name=st_mm_ord0_s5

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "$(conda shell.bash hook)"

conda activate /net/tscratch/people/plgkzaleska/envs/lora

cd /net/tscratch/people/plgkzaleska/ziplora-analysis

export PYTHONPATH=$PWD

/net/tscratch/people/plgkzaleska/envs/lora/bin/python3 scripts_cl/train_style_order.py --style_seed 5 --order_seed 0 --experiment_name ortho_init