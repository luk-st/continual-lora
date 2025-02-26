import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from eval.helpers import load_pickle, save_pickle
from eval.metrics import calculate_cl_metrics, clip_image_metric, csd_metric, dino_metric
from plots.incremental_performance_heatmap import plot_incremental_performance_heatmap, plot_incremental_performance_plot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_METRICS_ROOT = Path("./results/metrics/")
STYLE_METRICS = {
    "csd": csd_metric,
    "dino": dino_metric,
}
OBJECT_METRICS = {
    "clip": clip_image_metric,
    "dino": dino_metric,
}


def reverse_metrics_dict(metrics_dict, task_type):
    final_metrics_dict = {}
    metrics_names = STYLE_METRICS if task_type == "style" else OBJECT_METRICS

    for metric_name in metrics_names:
        metric_dict = {}
        for order_and_seed, order_and_seed_array in metrics_dict.items():
            metric_dict[order_and_seed] = order_and_seed_array[metric_name]
        final_metrics_dict[metric_name] = metric_dict

    return final_metrics_dict


def convert_metrics_to_arrays(tasks_metrics_dict, n_tasks: int):
    metrics_arrays = {}

    metrics_names = tasks_metrics_dict.keys()
    for metric_name in metrics_names:
        square_array = np.full((n_tasks, n_tasks), np.nan)
        for i in range(n_tasks):
            for j in range(i + 1):
                square_array[j, i] = tasks_metrics_dict[metric_name].get((i + 1, j + 1), np.nan)
        metrics_arrays[metric_name] = square_array

    return metrics_arrays


def get_order_seed_tasks(models_dir, order_seed, training_seed, task_type):
    config_file_path = Path(
        models_dir, f"seed_{training_seed}_{task_type}", f"seed_{order_seed}_order", "config.json"
    ).resolve()
    with open(config_file_path, "r") as file:
        tasks_config = json.load(file)
    return tasks_config["tasks"]


def find_all_orders_paths(models_path):
    return [p for p in Path(models_path).iterdir() if p.is_dir()]


def find_all_seeds_paths(order_path):
    return [p for p in Path(order_path).iterdir() if p.is_dir()]


def get_after_task_dir(order_seed_path, task_idx):
    return (order_seed_path / f"after_task_{task_idx}").resolve()


def get_on_task_samples(after_task_dir, task_idx):
    return load_pickle((after_task_dir / f"on_task_{task_idx}.pkl").resolve())["samples"]


def get_task_eval_path(task_config, task_type):
    metrics_key = "path" if task_type == "object" else "metric_path"
    return Path(task_config[metrics_key]).resolve()


def calculate_metrics(model_samples, eval_path, task_type):
    metrics_map = OBJECT_METRICS if task_type == "object" else STYLE_METRICS

    return {metric_name: metrics_map[metric_name](model_samples, eval_path) for metric_name in metrics_map.keys()}


def calc_cl_metrics(order_seed_metrics_arrays, task_type, save_path):
    metrics_map = OBJECT_METRICS if task_type == "object" else STYLE_METRICS

    final_scores = {}
    with open(f"{save_path}/out_cl.txt", "w") as file:
        for metric_name in metrics_map.keys():
            final_scores[metric_name] = {
                "avg_score": [],
                "avg_forgetting": [],
            }
            for order_and_seed, order_and_seed_array in order_seed_metrics_arrays.items():
                metric_avg_score, metric_avg_forgetting = calculate_cl_metrics(
                    metric_matrix=order_and_seed_array[metric_name]
                )

                output_str = f"{metric_name.upper()} | Order: {order_and_seed[0]} | Seed: {order_and_seed[1]}\n"
                output_str += f"AVG_SCORE={metric_avg_score} | AVG_FORGETTING={metric_avg_forgetting}\n"

                final_scores[metric_name]["avg_score"].append(metric_avg_score)
                final_scores[metric_name]["avg_forgetting"].append(metric_avg_forgetting)

                file.write(output_str)
                print(output_str, end="")

            avg_score_mean = np.mean(final_scores[metric_name]["avg_score"])
            avg_score_std = np.std(final_scores[metric_name]["avg_score"])
            avg_forgetting_mean = np.mean(final_scores[metric_name]["avg_forgetting"])
            avg_forgetting_std = np.std(final_scores[metric_name]["avg_forgetting"])

            output_str = f"{metric_name.upper()} | Averaged results:\n"
            output_str += f"Mean average score={avg_score_mean} | Std average score={avg_score_std}\n"
            output_str += (
                f"Mean average forgetting={avg_forgetting_mean} | Std average forgetting={avg_forgetting_std}\n\n"
            )
            file.write(output_str)
            print(output_str, end="")


def average_metrics(metrics_dict, task_type):
    metrics_map = OBJECT_METRICS if task_type == "object" else STYLE_METRICS

    output_metrics = {}
    tasks_combinations = metrics_dict[list(metrics_dict.keys())[0]].keys()
    for metric_name in metrics_map.keys():
        metric_vals = {}
        for task_combination in tasks_combinations:
            metric_vals[task_combination] = np.mean(
                [metrics_dict[seed_metrics][task_combination][metric_name] for seed_metrics in metrics_dict.keys()]
            )
        output_metrics[metric_name] = metric_vals

    return output_metrics


def get_metrics(samples_path, method_name, task_type, models_dir):
    metrics_path = (PATH_METRICS_ROOT / task_type / method_name).resolve()
    os.makedirs(metrics_path, exist_ok=True)

    metrics_dict = {}

    orders_paths = find_all_orders_paths(samples_path)
    for order_path in tqdm(orders_paths, desc="Orders"):
        order_seed = order_path.name.split("order_")[1]

        seeds_paths = find_all_seeds_paths(order_path)
        for seed_path in tqdm(seeds_paths, desc="Seeds"):
            seed = seed_path.name.split("seed_")[1]
            seeds_results = {}

            tasks = get_order_seed_tasks(
                models_dir=models_dir, order_seed=order_seed, training_seed=seed, task_type=task_type
            )
            tasks_map = {task["index"]: task for task in tasks}

            for after_task_model_idx in tqdm(tasks_map.keys(), desc="Models"):
                after_task_dir = get_after_task_dir(seed_path, after_task_model_idx)

                for task_idx in [task_idx for task_idx in tasks_map.keys() if task_idx <= after_task_model_idx]:
                    on_task_samples = get_on_task_samples(after_task_dir, task_idx)
                    eval_path = get_task_eval_path(tasks_map[task_idx], task_type)

                    seeds_results[(after_task_model_idx, task_idx)] = calculate_metrics(
                        model_samples=on_task_samples, eval_path=eval_path, task_type=task_type
                    )

            metrics_dict[(order_seed, seed)] = seeds_results
    save_pickle(metrics_dict, Path(metrics_path, "metrics_dict.pkl"))
    # metrics_dict = load_pickle(Path(metrics_path, "metrics_dict.pkl"))

    avg_metrics = average_metrics(metrics_dict, task_type)
    save_pickle(avg_metrics, Path(metrics_path, "metrics_avg.pkl"))
    # avg_metrics = load_pickle(Path(metrics_path, "metrics_avg.pkl"))

    n_tasks = max([at_idx for at_idx, _ in avg_metrics[next(iter(avg_metrics.keys()))].keys()])
    metrics_arrays = convert_metrics_to_arrays(avg_metrics, n_tasks=n_tasks)

    for metric_name, vals_array in metrics_arrays.items():
        plot_incremental_performance_heatmap(
            array=vals_array,
            metric_name=metric_name.upper(),
            name=task_type,
            save_dir=metrics_path,
        )
        plot_incremental_performance_plot(
            array=vals_array.T,
            metric_name=metric_name.upper(),
            name=task_type,
            save_dir=metrics_path,
        )
    order_seed_arrays = {
        order_and_seed: convert_metrics_to_arrays(
            reverse_metrics_dict(metrics_dict[order_and_seed], task_type), n_tasks
        )
        for order_and_seed in metrics_dict.keys()
    }

    calc_cl_metrics(order_seed_arrays, task_type, metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_path", type=str, required=True)
    parser.add_argument(
        "--method_name", type=str, required=True, choices=["mag_max_light", "merge_and_init", "naive_cl", "ortho_init"]
    )
    parser.add_argument("--task_type", type=str, required=True, choices=["object", "style"])
    parser.add_argument("--models_dir", type=str, required=True)
    args = parser.parse_args()

    get_metrics(
        samples_path=args.samples_path,
        method_name=args.method_name,
        task_type=args.task_type,
        models_dir=args.models_dir,
    )
