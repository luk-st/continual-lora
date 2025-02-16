#!/bin/bash
#SBATCH -A $slurm_account
#SBATCH -p $slurm_partition
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1
#SBATCH -o slurm_out/obj_mai_ord42_s0.log
#SBATCH --job-name=obj_mai_ord42_s0

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "$(conda shell.bash hook)"

conda activate PASS_CONDA_ENV
export PYTHONPATH=$PWD

python3 scripts_train/train_object_order.py --order_seed 42 --object_seed 0 --experiment_name merge_and_init