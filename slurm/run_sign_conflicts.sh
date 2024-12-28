#!/bin/bash

methods=("mag_max_light" "merge_and_init" "ortho_init" "naive_cl")
order_seeds=(0 5 10 42)
style_seeds=(0 5)
export PYTHONPATH=\$PWD

for method_name in "${methods[@]}"; do
    for order_seed in "${order_seeds[@]}"; do
        for style_seed in "${style_seeds[@]}"; do
            python3.11 ./eval/sign_conflicts_A_B_vectors.py --models_path models/${method_name}/seed_${style_seed}_object/seed_${order_seed}_order --method_name ${method_name} --task_type object
        done
    done
done