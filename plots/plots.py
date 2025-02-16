from pathlib import Path
from eval.eval_method import clip_image_metric, dino_metric, load_pickle, get_task_eval_path, calculate_metrics, csd_metric
import torch
import json
import pickle
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH_METRICS_ROOT = Path("./results/metrics/")
OBJECT_METRICS = {
    "clip": clip_image_metric,
    "dino": dino_metric,
}
STYLE_METRICS = {
    "csd": csd_metric,
    "dino": dino_metric,
}
METHOD_TO_NAME = {
    "mag_max_light": "Magnitude-based Merging",
    "naive_cl": "Naïve Continual Fine-Tuning",
    "ortho_init": "Merge & Orthogonal Initialization",
    "merge_and_init": "Merge & Initialization",
}

METRIC_TO_NAME = {
    "clip": "CLIP-I",
    "dino": "DINO",
    "csd": "CSD",
}


def load_res_from_path_object(path, tasks_map):
    path = Path(path).resolve()
    with open(path) as f:
        tasks_config = json.load(f)

    new_tasks_map = []
    for task in tasks_config['tasks']:
        for idx, task_base in tasks_map.items():
            if task['prompt'] == task_base['prompt']:
                new_tasks_map.append(idx)
                break
    return new_tasks_map

def load_res_from_path_style(path, tasks_map):
    path = Path(path).resolve()
    with open(path) as f:
        tasks_config = json.load(f)

    new_tasks_map = []
    for task in tasks_config['tasks']:
        for idx, task_base in tasks_map.items():
            if task['style'] == task_base['style']:
                new_tasks_map.append(idx)
                break
    return new_tasks_map

path_results = Path("results/object/base/after_task_10").resolve()
imgs_base_object = [
    load_pickle(path_results / Path(f"on_task_{task_idx}.pkl"))["samples"] for task_idx in range(1, 11)
]
path_config = Path("data/data_object/config.json").resolve()
with open(path_config) as f:
    file = json.load(f)
    tasks = file["tasks"]

    tasks_map = {int(idx) + 1: task for idx, task in enumerate(tasks)}
    for task_k, task_v in tasks_map.items():
        task_v["index"] = task_k

results = {}
for task_idx in [task_idx for task_idx in tasks_map.keys()]:
    on_task_samples = imgs_base_object[task_idx - 1]
    eval_path = get_task_eval_path(tasks_map[task_idx], "object")
    results[task_idx] = calculate_metrics(
        model_samples=on_task_samples, eval_path=eval_path, task_type="object"
    )

task_map_1 = load_res_from_path_object("models/naive_cl/seed_0_object/seed_0_order/config.json", tasks_map)
task_map_2 = load_res_from_path_object("models/naive_cl/seed_0_object/seed_5_order/config.json", tasks_map)
task_map_3 = load_res_from_path_object("models/naive_cl/seed_0_object/seed_10_order/config.json", tasks_map)
task_map_4 = load_res_from_path_object("models/naive_cl/seed_0_object/seed_42_order/config.json", tasks_map)

results1 = {
    task_idx: results[task_map_1[task_idx]] for task_idx in range(len(task_map_1))
}
results2 = {
    task_idx: results[task_map_2[task_idx]] for task_idx in range(len(task_map_2))
}
results3 = {
    task_idx: results[task_map_3[task_idx]] for task_idx in range(len(task_map_3))
}
results4 = {
    task_idx: results[task_map_4[task_idx]] for task_idx in range(len(task_map_4))
}
results_final = {
    metric_name: {
        task_idx + 1: np.mean([results1[task_idx][metric_name], results2[task_idx][metric_name], results3[task_idx][metric_name], results4[task_idx][metric_name]])
        for task_idx in range(10)
    }
    for metric_name in OBJECT_METRICS.keys()
}

results = {}
for task_idx in [task_idx for task_idx in tasks_map.keys()]:
    on_task_samples = imgs_base_object[task_idx - 1]
    eval_path = get_task_eval_path(tasks_map[task_idx], "object")
    results[task_idx] = calculate_metrics(
        model_samples=on_task_samples, eval_path=eval_path, task_type="object"
    )

clip = [metrics_results["clip"] for metrics_results in results.values()]
dino = [metrics_results["dino"] for metrics_results in results.values()]
np.mean(clip, axis=0), np.mean(dino, axis=0)


# OBJECTS

METRIC_NAME = "dino"
METHOD_NAME = "mag_max_light"
METHOD_NAME2 = "naive_cl"
METHOD_NAME3 = "ortho_init"
METHOD_NAME4 = "merge_and_init"
TASK_IDX = 1
path = Path(f"results/metrics/object/{METHOD_NAME}/metrics_avg.pkl").resolve()
path2 = Path(f"results/metrics/object/{METHOD_NAME2}/metrics_avg.pkl").resolve()
path3 = Path(f"results/metrics/object/{METHOD_NAME3}/metrics_avg.pkl").resolve()
path4 = Path(f"results/metrics/object/{METHOD_NAME4}/metrics_avg.pkl").resolve()

with open(path, "rb") as f:
    metrics = pickle.load(f)[METRIC_NAME]
with open(path2, "rb") as f:
    metrics2 = pickle.load(f)[METRIC_NAME]
with open(path3, "rb") as f:
    metrics3 = pickle.load(f)[METRIC_NAME]
with open(path4, "rb") as f:
    metrics4 = pickle.load(f)[METRIC_NAME]

exclude_models = []
plt.style.use("ggplot")
plt.rcParams.update(
    {
        "axes.facecolor": "#f0f0f0",
        "figure.facecolor": "#ffffff",
        "grid.color": "white",
        "axes.edgecolor": "#393939",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#393939",
        "xtick.color": "#393939",
        "ytick.color": "#393939",
        "font.size": 12,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "figure.figsize": (14, 7),
    }
)

tasks = defaultdict()
tasks2 = defaultdict()
tasks3 = defaultdict()
tasks4 = defaultdict()

data = metrics
data2 = metrics2
data3 = metrics3
data4 = metrics4
data3 = metrics3


labels = ["" for _ in range(len(list(data.keys())) + 1)]
for (model_idx, task_idx) in data.keys():
    if task_idx != TASK_IDX:
        continue
    labels[model_idx] = f"#{model_idx} tasks"
    tasks[model_idx] = data[(model_idx, TASK_IDX)]

for (model_idx, task_idx) in data2.keys():
    if task_idx != TASK_IDX:
        continue
    tasks2[model_idx] = data2[(model_idx, TASK_IDX)]

for (model_idx, task_idx) in data3.keys():
    if task_idx != TASK_IDX:
        continue
    tasks3[model_idx] = data3[(model_idx, TASK_IDX)]

for (model_idx, task_idx) in data4.keys():
    if task_idx != TASK_IDX:
        continue
    tasks4[model_idx] = data4[(model_idx, TASK_IDX)]



colors = plt.cm.viridis(np.linspace(0, 1, 11))
tasks[0] = {}
labels[0] = "Base model"
tasks[0] = results_final[METRIC_NAME][TASK_IDX]
fig, ax = plt.subplots(figsize=(8, 6))

x = [model_idx for model_idx in tasks.keys() if model_idx not in [0] + exclude_models]
y = [tasks[model_idx] for model_idx in x]
ax.plot(x, y, label=METHOD_TO_NAME[METHOD_NAME], color="rebeccapurple", linewidth=3, markersize=7, linestyle="-")
y2 = [tasks2[model_idx] for model_idx in x]
ax.plot(x, y2, label=METHOD_TO_NAME[METHOD_NAME2], color="darkcyan", linewidth=3, markersize=7, linestyle="-")
y3 = [tasks3[model_idx] for model_idx in x]
ax.plot(x, y3, label=METHOD_TO_NAME[METHOD_NAME3], color="goldenrod", linewidth=3, markersize=7, linestyle="-")
y4 = [tasks4[model_idx] for model_idx in x]
ax.plot(x, y4, label=METHOD_TO_NAME[METHOD_NAME4], color="indianred", linewidth=3, markersize=7, linestyle="-")
plt.axhline(y=tasks[0], color='black', linestyle='--', linewidth=3, label="Base model performance")

ax.set_xlabel('Model after task', fontsize=24)
ax.set_ylabel(METRIC_TO_NAME[METRIC_NAME], fontsize=24)
ax.set_xticklabels(list(tasks.keys()), fontsize=20)
ax.set_xlim((TASK_IDX, 10))
ax.set_ylim(0.15, 0.7)
ax.set_yticks([0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels([0.2, 0.3, 0.4, 0.5], fontsize=20)
ax.legend(fontsize=18, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(f"models_object.pdf")
plt.close()


# STYLES

path_results = Path("results/style/base/after_task_10").resolve()
imgs_base_style = [
    load_pickle(path_results / Path(f"on_task_{task_idx}.pkl"))["samples"] for task_idx in range(1, 11)
]
path_config = Path("data/data_style/config.json").resolve()
with open(path_config) as f:
    file = json.load(f)
    tasks = file["tasks"]

    tasks_map = {int(idx) + 1: task for idx, task in enumerate(tasks)}
    for task_k, task_v in tasks_map.items():
        task_v["index"] = task_k

results_style = {}
for task_idx in [task_idx for task_idx in tasks_map.keys()]:
    on_task_samples = imgs_base_style[task_idx - 1]
    eval_path = get_task_eval_path(tasks_map[task_idx], "style")
    results_style[task_idx] = calculate_metrics(
        model_samples=on_task_samples, eval_path=eval_path, task_type="style"
    )

csd = [metrics_results["csd"] for metrics_results in results_style.values()]
dino = [metrics_results["dino"] for metrics_results in results_style.values()]
np.mean(csd, axis=0), np.mean(dino, axis=0)

task_map_1 = load_res_from_path_style("models/merge_and_init/seed_0_style/seed_0_order/config.json", tasks_map)
task_map_2 = load_res_from_path_style("models/merge_and_init/seed_0_style/seed_5_order/config.json", tasks_map)
task_map_3 = load_res_from_path_style("models/merge_and_init/seed_0_style/seed_10_order/config.json", tasks_map)
task_map_4 = load_res_from_path_style("models/merge_and_init/seed_0_style/seed_42_order/config.json", tasks_map)

results1 = {
    task_idx: results_style[task_map_1[task_idx]] for task_idx in range(len(task_map_1))
}
results2 = {
    task_idx: results_style[task_map_2[task_idx]] for task_idx in range(len(task_map_2))
}
results3 = {
    task_idx: results_style[task_map_3[task_idx]] for task_idx in range(len(task_map_3))
}
results4 = {
    task_idx: results_style[task_map_4[task_idx]] for task_idx in range(len(task_map_4))
}
results_final = {
    metric_name: {
        task_idx + 1: np.mean([results1[task_idx][metric_name], results2[task_idx][metric_name], results3[task_idx][metric_name], results4[task_idx][metric_name]])
        for task_idx in range(10)
    }
    for metric_name in STYLE_METRICS.keys()
}

METRIC_NAME = "dino"
METHOD_NAME = "mag_max_light"
METHOD_NAME2 = "naive_cl"
METHOD_NAME3 = "ortho_init"
METHOD_NAME4 = "merge_and_init"
TASK_IDX = 2


path = Path(f"results/metrics/style/{METHOD_NAME}/metrics_avg.pkl").resolve()
path2 = Path(f"results/metrics/style/{METHOD_NAME2}/metrics_avg.pkl").resolve()
path3 = Path(f"results/metrics/style/{METHOD_NAME3}/metrics_avg.pkl").resolve()
path4 = Path(f"results/metrics/style/{METHOD_NAME4}/metrics_avg.pkl").resolve()

with open(path, "rb") as f:
    metrics = pickle.load(f)[METRIC_NAME]

with open(path2, "rb") as f:
    metrics2 = pickle.load(f)[METRIC_NAME]

with open(path3, "rb") as f:
    metrics3 = pickle.load(f)[METRIC_NAME]

with open(path4, "rb") as f:
    metrics4 = pickle.load(f)[METRIC_NAME]

exclude_models = []
plt.style.use("ggplot")
plt.rcParams.update(
    {
        "axes.facecolor": "#f0f0f0",
        "figure.facecolor": "#ffffff",
        "grid.color": "white",
        "axes.edgecolor": "#393939",
        "axes.linewidth": 1.0,
        "axes.labelcolor": "#393939",
        "xtick.color": "#393939",
        "ytick.color": "#393939",
        "font.size": 12,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "figure.figsize": (14, 7),
    }
)

tasks = defaultdict()
tasks2 = defaultdict()
tasks3 = defaultdict()
tasks4 = defaultdict()

data = metrics
data2 = metrics2
data3 = metrics3
data4 = metrics4
data3 = metrics3


labels = ["" for _ in range(len(list(data.keys())) + 1)]
data[(1, 2)] = data[(1, 1)]
data2[(1, 2)] = data2[(1, 1)]
data3[(1, 2)] = data3[(1, 1)]
data4[(1, 2)] = data4[(1, 1)]
for (model_idx, task_idx) in data.keys():
    if task_idx != TASK_IDX:
        continue
    labels[model_idx] = f"#{model_idx} tasks"
    tasks[model_idx] = data[(model_idx, TASK_IDX)]

for (model_idx, task_idx) in data2.keys():
    if task_idx != TASK_IDX:
        continue
    tasks2[model_idx] = data2[(model_idx, TASK_IDX)]

for (model_idx, task_idx) in data3.keys():
    if task_idx != TASK_IDX:
        continue
    tasks3[model_idx] = data3[(model_idx, TASK_IDX)]

for (model_idx, task_idx) in data4.keys():
    if task_idx != TASK_IDX:
        continue
    tasks4[model_idx] = data4[(model_idx, TASK_IDX)]



colors = plt.cm.viridis(np.linspace(0, 1, 11))
tasks[0] = {}
labels[0] = "Base model"
tasks[0] = results_final[METRIC_NAME][TASK_IDX]
fig, ax = plt.subplots(figsize=(8, 6))
x = sorted([model_idx for model_idx in tasks.keys() if model_idx not in [0] + exclude_models])
y = [tasks[model_idx] for model_idx in x]
ax.plot(x, y, label=METHOD_TO_NAME[METHOD_NAME], color="rebeccapurple", linewidth=3, markersize=7, linestyle="-")
y2 = [tasks2[model_idx] for model_idx in x]
ax.plot(x, y2, label=METHOD_TO_NAME[METHOD_NAME2], color="darkcyan", linewidth=3, markersize=7, linestyle="-")
y3 = [tasks3[model_idx] for model_idx in x]
ax.plot(x, y3, label=METHOD_TO_NAME[METHOD_NAME3], color="goldenrod", linewidth=3, markersize=7, linestyle="-")
y4 = [tasks4[model_idx] for model_idx in x]
ax.plot(x, y4, label=METHOD_TO_NAME[METHOD_NAME4], color="indianred", linewidth=3, markersize=7, linestyle="-")
plt.axhline(y=tasks[0], color='black', linestyle='--', linewidth=3, label="Base model performance")

ax.set_xlabel('Model after task', fontsize=24)
ax.set_ylabel(METRIC_TO_NAME[METRIC_NAME], fontsize=24)
ax.set_xticklabels(sorted(list(tasks.keys()))[1:], fontsize=20)
ax.set_xlim((1, 10))
ax.set_ylim(0.1, 0.5)
ax.set_yticks([0.2, 0.3, 0.4])
ax.set_yticklabels([0.2, 0.3, 0.4], fontsize=20)
ax.legend(fontsize=18, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(f"models_style.pdf")
plt.close()