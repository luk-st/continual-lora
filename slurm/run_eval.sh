#!/bin/bash

methods=("mag_max_light" "merge_and_init" "naive_cl" "ortho_init")

for method_name in "${methods[@]}"; do
    log_file="slurm_out/fix_eval_style_${method_name}.log"
    job_name="eval_style_${method_name}"

    sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH -A plgdiffusion-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH -t 2:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 140G
#SBATCH --cpus-per-task=8
#SBATCH --nodes 1

module load Miniconda3/23.3.1-0
eval "\$(conda shell.bash hook)"
conda activate /net/tscratch/people/plgkzaleska/envs/lora

export PYTHONPATH=\$PWD

python3 ./eval/eval_method.py --samples_path results/style/${method_name} --method_name ${method_name} --task_type style --models_dir models/${method_name}/
EOT

done
