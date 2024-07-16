import os

import torch

from eval.helpers import get_cl_lora_alignment_metrics, sample_cl_models, save_pickle
from eval.metrics import calculate_cl_metrics
from eval.plots import (
    convert_metrics_to_arrays,
    plot_incremental_performance_heatmap,
    plot_incremental_performance_plot,
)

RESULTS_DIR_PATH = "./results/seed_0_style"
N_TASKS = 5
DATASETS = [
    f"./data/styledrop/{ds_name}"
    for ds_name in [
        "watercolor_painting_style",
        "oil_painting_style",
        "flat_cartoon_illustration",
        "abstract_rainbow_colored_flowing_smoke_wave",
        "sticker",
    ]
]
STYLES_MODELS = [
    f"./models/seed_0_style/{model_name}"
    for model_name in [
        "watercolor_painting_style_sd1",
        "oil_painting_style_sd2",
        "flat_cartoon_illustration_style_sd3",
        "abstract_rainbow_colored_style_sd4",
        "sticker_style_sd5",
    ]
]
STYLES_TOKENS = ["skn style", "zwz style", "fci style", "rcf style", "cts style"]

PROMPT_TEMPLATES = [
    "a wizard in {}",
    "a policeman in {}",
    "a santa hat in {}",
    "a clock in {}",
    "a mirror in {}",
]


def get_style_metrics():
    os.makedirs(RESULTS_DIR_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks_outs = [
        sample_cl_models(
            STYLES_MODELS,
            task_number=task_idx,
            tasks_tokens=STYLES_TOKENS,
            prompts_templates=PROMPT_TEMPLATES,
            n_tasks=N_TASKS,
            device=device,
        )
        for task_idx in range(N_TASKS + 1)
    ]
    save_pickle(tasks_outs, f"{RESULTS_DIR_PATH}/samples.pkl")

    metrics = get_cl_lora_alignment_metrics(
        models_tasks_outputs=tasks_outs,
        gt_datasets_paths=DATASETS,
        n_tasks=N_TASKS,
        is_style=True,
    )
    save_pickle(metrics, f"{RESULTS_DIR_PATH}/metrics.pkl")

    clip_array, dino_array = convert_metrics_to_arrays(
        metrics_names=["clip", "dino"], tasks_metrics_dict=metrics, n_tasks=N_TASKS
    )

    plot_incremental_performance_heatmap(
        clip_array=clip_array,
        dino_array=dino_array,
        name=f"heatmap",
        save_dir=RESULTS_DIR_PATH,
    )
    plot_incremental_performance_plot(
        clip_array=clip_array.T,
        dino_array=dino_array.T,
        name=f"plot",
        save_dir=RESULTS_DIR_PATH,
    )

    dino_avg_accuracy, dino_avg_forgetting = calculate_cl_metrics(dino_array.T[1:, :])
    print(
        f"DINO AVG_ACC={dino_avg_accuracy}",
        f"DINO AVG_FORGETTING={dino_avg_forgetting}",
    )
    clip_avg_accuracy, clip_avg_forgetting = calculate_cl_metrics(clip_array.T[1:, :])
    print(
        f"CLIP AVG_ACC={clip_avg_accuracy}",
        f"CLIP AVG_FORGETTING={clip_avg_forgetting}",
    )


if __name__ == "__main__":
    get_style_metrics()
