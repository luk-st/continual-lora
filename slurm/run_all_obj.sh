#!/bin/bash

cd_path="<PASS_REPO_PATH>"
order_seeds=(0 5 10 42)
object_seeds=(0 5)

for order_seed in "${order_seeds[@]}"; do
    for object_seed in "${object_seeds[@]}"; do
        log_file="slurm_out/obj_mai_ord${order_seed}_s${object_seed}.log"
        job_name="obj_mai_ord${order_seed}_s${object_seed}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A plggenerativepw2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate lora

cd $cd_path

export PYTHONPATH=\$PWD

python3 scripts_cl/train_object_order.py --order_seed $order_seed --object_seed $object_seed --experiment_name merge_and_init
EOT

    done
done

for order_seed in "${order_seeds[@]}"; do
    for object_seed in "${object_seeds[@]}"; do
        log_file="slurm_out/obj_mm_ord${order_seed}_s${object_seed}.log"
        job_name="obj_mm_ord${order_seed}_s${object_seed}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A plggenerativepw2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate lora

cd $cd_path

export PYTHONPATH=\$PWD

python3 scripts_cl/train_object_order.py --order_seed $order_seed --object_seed $object_seed --experiment_name mag_max_light
EOT

    done
done

for order_seed in "${order_seeds[@]}"; do
    for object_seed in "${object_seeds[@]}"; do
        log_file="slurm_out/obj_naive_ord${order_seed}_s${object_seed}.log"
        job_name="obj_naive_ord${order_seed}_s${object_seed}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A plggenerativepw2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate lora

cd $cd_path

export PYTHONPATH=\$PWD

python3 scripts_cl/train_object_order.py --order_seed $order_seed --object_seed $object_seed --experiment_name naive_cl
EOT

    done
done

for order_seed in "${order_seeds[@]}"; do
    for object_seed in "${object_seeds[@]}"; do
        log_file="slurm_out/obj_naive_ord${order_seed}_s${object_seed}.log"
        job_name="obj_naive_ord${order_seed}_s${object_seed}"

        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A plggenerativepw2-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 12:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load GCC/11.2.0
module load Miniconda3/23.3.1-0

eval "\$(conda shell.bash hook)"

conda activate lora

cd $cd_path

export PYTHONPATH=\$PWD

python3 scripts_cl/train_object_order.py --order_seed $order_seed --object_seed $object_seed --experiment_name ortho_init
EOT

    done
done