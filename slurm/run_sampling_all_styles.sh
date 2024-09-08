#!/bin/bash

methods=("mag_max_light" "merge_and_init" "naive_cl" "ortho_init")
order_seeds=(0 5 10 42)
style_seeds=(0 5)

for method_name in "${methods[@]}"; do
    for order_seed in "${order_seeds[@]}"; do
        for style_seed in "${style_seeds[@]}"; do
            log_file="slurm_out/sample_style_${method_name}_order${order_seed}_seed${style_seed}.log"
            job_name="sample_style_${method_name}_order${order_seed}_seed${style_seed}"

            sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A plgdiffusion-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 4:30:00
#SBATCH --ntasks 4
#SBATCH --gres gpu:4
#SBATCH --mem 480G
#SBATCH --cpus-per-task=16
#SBATCH --nodes 1

module load Miniconda3/23.3.1-0
eval "\$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgkzaleska/envs/lora

export PYTHONPATH=\$PWD

accelerate launch --num-processes 4 ./eval/sample_images.py --models_path ./models/${method_name}/seed_${style_seed}_style/seed_${order_seed}_order --method_name ${method_name} --task_type style
EOT

        done
    done
done
