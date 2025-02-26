#!/bin/bash

cd_path="<PASS_REPO_PATH>"
slurm_account="<PASS_SLURM_ACCOUNT>"
slurm_partition="<PASS_SLURM_PARTITION>"
conda_env="<PASS_CONDA_ENV>"
order_seeds=(0 5 10 42)
style_seeds=(0 5)

for order_seed in "${order_seeds[@]}"; do
    for style_seed in "${style_seeds[@]}"; do
        log_file="slurm_out/style_mai_ord${order_seed}_s${style_seed}.log"
        job_name="style_mai_ord${order_seed}_s${style_seed}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A $slurm_account
#SBATCH -p $slurm_partition
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate $conda_env

cd $cd_path

export PYTHONPATH=\$PWD

python3 scripts_train/train_style_order.py --order_seed $order_seed --style_seed $style_seed --experiment_name merge_and_init
EOT

    done
done

for order_seed in "${order_seeds[@]}"; do
    for style_seed in "${style_seeds[@]}"; do
        log_file="slurm_out/style_mai_ord${order_seed}_s${style_seed}.log"
        job_name="style_mai_ord${order_seed}_s${style_seed}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A $slurm_account
#SBATCH -p $slurm_partition
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate $conda_env

cd $cd_path

export PYTHONPATH=\$PWD

python3 scripts_train/train_style_order.py --order_seed $order_seed --style_seed $style_seed --experiment_name mag_max_light
EOT

    done
done

for order_seed in "${order_seeds[@]}"; do
    for style_seed in "${style_seeds[@]}"; do
        log_file="slurm_out/style_naive_ord${order_seed}_s${style_seed}.log"
        job_name="style_naive_ord${order_seed}_s${style_seed}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A $slurm_account
#SBATCH -p $slurm_partition
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate $conda_env

cd $cd_path

export PYTHONPATH=\$PWD

python3 scripts_train/train_style_order.py --order_seed $order_seed --style_seed $style_seed --experiment_name naive_cl
EOT

    done
done

for order_seed in "${order_seeds[@]}"; do
    for style_seed in "${style_seeds[@]}"; do
        log_file="slurm_out/style_orth_ord${order_seed}_s${style_seed}.log"
        job_name="style_orth_ord${order_seed}_s${style_seed}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A $slurm_account
#SBATCH -p $slurm_partition
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate $conda_env

cd $cd_path

export PYTHONPATH=\$PWD

python3 scripts_train/train_style_order.py --order_seed $order_seed --style_seed $style_seed --experiment_name ortho_init
EOT

    done
done