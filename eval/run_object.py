import os

import torch

from eval.helpers import get_cl_lora_alignment_metrics, sample_cl_models, save_pickle
from eval.metrics import calculate_cl_metrics
from eval.plots import (
    convert_metrics_to_arrays,
    plot_incremental_performance_heatmap,
    plot_incremental_performance_plot,
)

RESULTS_DIR_PATH = "./results/seed_33_object"
N_TASKS = 5
DATASETS = [
    f"./data/dreambooth/dataset/{ds_name}"
    for ds_name in [
        "wolf_plushie",
        "backpack",
        "dog6",
        "candle",
        "cat2",
    ]
]
OBJECTS_MODELS = [
    f"./models/seed_33_object/{model_name}"
    for model_name in [
        "wolf_plushie_sd1",
        "backpack_sd2",
        "dog6_sd3",
        "candle_sd4",
        "cat2_sd5",
    ]
]
OBJECTS_TOKENS = [
    "sks stuffed animal",
    "zwz backpack",
    "sbu dog",
    "uwu candle",
    "pdw cat",
]


PROMPT_TEMPLATES = [
    "a {} in a purple wizard outfit",
    "a {} in a police outfit",
    "a {} wearing a santa hat",
    "a {} in a jail",
    "a {} looking into a mirror",
]


def get_object_metrics():
    os.makedirs(RESULTS_DIR_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tasks_outs = [
        sample_cl_models(
            OBJECTS_MODELS,
            task_number=task_idx,
            tasks_tokens=OBJECTS_TOKENS,
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
        n_tasks=N_TASKS,
        name=f"heatmap",
        save_dir=RESULTS_DIR_PATH,
    )
    plot_incremental_performance_plot(
        clip_array=clip_array.T,
        dino_array=dino_array.T,
        n_tasks=N_TASKS,
        name=f"plot",
        save_dir=RESULTS_DIR_PATH,
    )

    dino_avg_accuracy, dino_avg_forgetting = calculate_cl_metrics(dino_array.T[1:, :])
    clip_avg_accuracy, clip_avg_forgetting = calculate_cl_metrics(clip_array.T[1:, :])
    with open(f"{RESULTS_DIR_PATH}/out_cl.txt", "w") as file:
        dino_str = f"DINO AVG_ACC={dino_avg_accuracy}\nDINO AVG_FORGETTING={dino_avg_forgetting}\n"
        clip_str = f"CLIP AVG_ACC={clip_avg_accuracy}\nCLIP AVG_FORGETTING={clip_avg_forgetting}\n"

        file.write(dino_str)
        print(dino_str)

        file.write(clip_str)
        print(clip_str)


if __name__ == "__main__":
    get_object_metrics()
