#!/bin/bash

slurm_account="<PASS_SLURM_ACCOUNT>"
slurm_partition="<PASS_SLURM_PARTITION>"
conda_env="<PASS_CONDA_ENV>"

methods=("mag_max_light" "merge_and_init" "naive_cl" "ortho_init")
order_seeds=(0 5 10 42)
object_seeds=(0 5)

for method_name in "${methods[@]}"; do
    for order_seed in "${order_seeds[@]}"; do
        for object_seed in "${object_seeds[@]}"; do
            log_file="slurm_out/sample_obj_${method_name}_order${order_seed}_seed${object_seed}.log"
            job_name="sample_obj_${method_name}_order${order_seed}_seed${object_seed}"

            sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A $slurm_account
#SBATCH -p $slurm_partition
#SBATCH -t 4:30:00
#SBATCH --ntasks 4
#SBATCH --gres gpu:4
#SBATCH --mem 480G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load Miniconda3/23.3.1-0
eval "\$(conda shell.bash hook)"
conda activate $conda_env

export PYTHONPATH=\$PWD

accelerate launch --num-processes 4 ./eval/sample_images.py --models_path ./models/${method_name}/seed_${object_seed}_object/seed_${order_seed}_order --method_name ${method_name} --task_type object
EOT

        done
    done
done
