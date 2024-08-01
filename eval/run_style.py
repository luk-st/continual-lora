import os

import torch

from eval.helpers import (
    average_on_seeds,
    convert_metrics_to_arrays,
    get_cl_lora_alignment_metrics,
    load_pickle,
    sample_cl_models,
    save_pickle,
)
from eval.metrics import calculate_cl_metrics
from eval.plots import (
    plot_incremental_performance_heatmap,
    plot_incremental_performance_plot,
)

SEEDS = [0, 11, 33, 42]
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
STYLES_TOKENS = ["skn style", "zwz style", "fci style", "rcf style", "cts style"]
PROMPT_TEMPLATES = [
    "a wizard in {}",
    "a policeman in {}",
    "a santa hat in {}",
    "a clock in {}",
    "a mirror in {}",
]
FINAL_RESULTS_PATH = "./results/style_avaraged"
METRICS = ["csd", "dino"]


def _get_results_dir_path(seed):
    return f"./results/seed_{seed}_style"


def _get_style_models(seed):
    return [
        f"./models/seed_{seed}_style/{model_name}"
        for model_name in [
            "watercolor_painting_style_sd1",
            "oil_painting_style_sd2",
            "flat_cartoon_illustration_style_sd3",
            "abstract_rainbow_colored_style_sd4",
            "sticker_style_sd5",
        ]
    ]


def get_style_metrics():
    os.makedirs(FINAL_RESULTS_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds_outs = []
    for seed in SEEDS:
        results_dir_path = _get_results_dir_path(seed=seed)
        models_paths = _get_style_models(seed=seed)
        os.makedirs(results_dir_path, exist_ok=True)
        seed_outs = [
            sample_cl_models(
                models_paths,
                task_number=task_idx,
                tasks_tokens=STYLES_TOKENS,
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
            is_style=True,
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
    # final_array = load_pickle(f"{FINAL_RESULTS_PATH}/final_metrics.pkl")

    plot_incremental_performance_heatmap(
        clip_array=final_array['csd'],
        dino_array=final_array['dino'],
        n_tasks=N_TASKS,
        name=f"heatmap",
        save_dir=FINAL_RESULTS_PATH,
    )
    plot_incremental_performance_plot(
        clip_array=final_array['csd'].T,
        dino_array=final_array['dino'].T,
        n_tasks=N_TASKS,
        name=f"plot",
        save_dir=FINAL_RESULTS_PATH,
    )

    dino_avg_accuracy, dino_avg_forgetting = calculate_cl_metrics(final_array['dino'].T[1:, :])
    csd_avg_accuracy, csd_avg_forgetting = calculate_cl_metrics(final_array['csd'].T[1:, :])

    with open(f"{FINAL_RESULTS_PATH}/out_cl.txt", "w") as file:
        dino_str = f"DINO AVG_ACC={dino_avg_accuracy}\nDINO AVG_FORGETTING={dino_avg_forgetting}\n"
        csd_str = f"CSD AVG_ACC={csd_avg_accuracy}\nCSD AVG_FORGETTING={csd_avg_forgetting}\n"

        file.write(dino_str)
        print(dino_str)

        file.write(csd_str)
        print(csd_str)


if __name__ == "__main__":
    get_style_metrics()
