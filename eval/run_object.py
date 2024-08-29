import json
import os

import torch

from eval.helpers import (average_on_seeds, convert_metrics_to_arrays,
                          get_cl_lora_alignment_metrics, load_pickle,
                          sample_cl_models, save_pickle)
from eval.metrics import calculate_cl_metrics
from eval.plots import (plot_incremental_performance_heatmap,
                        plot_incremental_performance_plot)

SEEDS = [0, 11, 33, 42]
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
FINAL_RESULTS_PATH = "./results/object_avaraged"
METRICS = ["clip", "dino"]


def _get_results_dir_path(seed):
    return f"./results/seed_{seed}_object"


def _get_object_models(seed):
    return [
        f"./models/seed_{seed}_object/{model_name}"
        for model_name in [
            "wolf_plushie_sd1",
            "backpack_sd2",
            "dog6_sd3",
            "candle_sd4",
            "cat2_sd5",
        ]
    ]


def save_config():
    config = {
        "SEEDS": SEEDS,
        "N_TASKS": N_TASKS,
        "DATASETS": DATASETS,
        "PROMPT": [
            {"OBJECTS_TOKENS": OBJECTS_TOKENS},
            {"PROMPT_TEMPLATES": PROMPT_TEMPLATES},
        ],
        "FINAL_RESULTS_PATH": FINAL_RESULTS_PATH,
        "METRICS": METRICS,
    }
    with open(f"{FINAL_RESULTS_PATH}/config.json", "w") as file:
        json.dump(config, file)


def get_object_metrics():
    os.makedirs(FINAL_RESULTS_PATH, exist_ok=True)
    save_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds_outs = []
    for seed in SEEDS:
        results_dir_path = _get_results_dir_path(seed=seed)
        models_paths = _get_object_models(seed=seed)
        os.makedirs(results_dir_path, exist_ok=True)
        seed_outs = [
            sample_cl_models(
                models_paths,
                task_number=task_idx,
                tasks_tokens=OBJECTS_TOKENS,
                prompts_templates=PROMPT_TEMPLATES,
                n_tasks=N_TASKS,
                samples_per_prompt=4,
                device=device,
            )
            for task_idx in range(N_TASKS + 1)
        ]
        save_pickle(seed_outs, f"{results_dir_path}/samples.pkl")
        # seed_outs = load_pickle(f"{results_dir_path}/samples.pkl")
        seeds_outs.append(seed_outs)

    seeds_metrics = []
    for idx, seed in enumerate(SEEDS):
        results_dir_path = _get_results_dir_path(seed=seed)
        per_seed_metrics = get_cl_lora_alignment_metrics(
            models_tasks_outputs=seeds_outs[idx],
            gt_datasets_paths=DATASETS,
            n_tasks=N_TASKS,
            is_style=False,
        )
        save_pickle(per_seed_metrics, f"{results_dir_path}/metrics.pkl")
        # per_seed_metrics = load_pickle(f"{results_dir_path}/metrics.pkl")
        seeds_metrics.append(per_seed_metrics)

    seeds_arrays = [
        convert_metrics_to_arrays(
            metrics_names=METRICS,
            tasks_metrics_dict=seed_metrics,
            n_tasks=N_TASKS,
        )
        for seed_metrics in seeds_metrics
    ]
    final_array = average_on_seeds(seeds_metrics=seeds_arrays, metrics_names=METRICS)
    save_pickle(final_array, f"{FINAL_RESULTS_PATH}/final_metrics.pkl")
    # final_array=load_pickle(f"{FINAL_RESULTS_PATH}/final_metrics.pkl")

    plot_incremental_performance_heatmap(
        arrays=[final_array["clip"], final_array["dino"]],
        names=["CLIP", "DINO"],
        n_tasks=N_TASKS,
        name=f"heatmap",
        save_dir=FINAL_RESULTS_PATH,
    )
    plot_incremental_performance_plot(
        arrays=[final_array["clip"].T, final_array["dino"].T],
        names=["CLIP", "DINO"],
        n_tasks=N_TASKS,
        name=f"plot",
        save_dir=FINAL_RESULTS_PATH,
    )

    dino_avg_accuracy, dino_avg_forgetting = calculate_cl_metrics(
        final_array["dino"].T[1:, :]
    )
    clip_avg_accuracy, clip_avg_forgetting = calculate_cl_metrics(
        final_array["clip"].T[1:, :]
    )
    with open(f"{FINAL_RESULTS_PATH}/out_cl.txt", "w") as file:
        dino_str = f"DINO AVG_ACC={dino_avg_accuracy}\nDINO AVG_FORGETTING={dino_avg_forgetting}\n"
        clip_str = f"CLIP AVG_ACC={clip_avg_accuracy}\nCLIP AVG_FORGETTING={clip_avg_forgetting}\n"

        file.write(dino_str)
        print(dino_str)

        file.write(clip_str)
        print(clip_str)


if __name__ == "__main__":
    get_object_metrics()
