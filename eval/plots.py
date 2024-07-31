import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_incremental_performance_heatmap(
    clip_array, dino_array, n_tasks, name="", save_dir="."
):
    tasks = ["<No task>"] + [f"{idx}" for idx in range(1, n_tasks + 1)]
    colormap = "YlOrRd"

    # CLIP-I
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        clip_array,
        annot=True,
        fmt=".3f",
        cmap=colormap,
        xticklabels=tasks,
        yticklabels=tasks[1:],
        linewidths=0.9,
        linecolor="black",
    )
    plt.title("CLIP-I Alignment Score on next tasks")
    plt.xlabel("Model after task")
    plt.ylabel("Performance on task")
    if name != "":
        plt.savefig(f"{save_dir}/{name}_CLIP.svg")
    plt.show()

    # DINO
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        dino_array,
        annot=True,
        fmt=".3f",
        cmap=colormap,
        xticklabels=tasks,
        yticklabels=tasks[1:],
        linewidths=0.9,
        linecolor="black",
    )
    plt.title("DINO Alignment Score on next tasks")
    plt.xlabel("Model after task")
    plt.ylabel("Performance on task")
    if name != "":
        plt.savefig(f"{save_dir}/{name}_DINO.svg")
    plt.show()


def plot_incremental_performance_plot(
    clip_array, dino_array, n_tasks, name="", save_dir="."
):
    plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "axes.facecolor": "#f0f0f0",
            "figure.facecolor": "#ffffff",
            "grid.color": "white",
            "axes.edgecolor": "gray",
            "axes.linewidth": 1.0,
            "axes.labelcolor": "gray",
            "xtick.color": "gray",
            "ytick.color": "gray",
            "font.size": 12,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "figure.figsize": (14, 7),
        }
    )
    colors = plt.cm.viridis(np.linspace(0, 1, clip_array.shape[1] + 1))
    markers = ["o", "s", "^", "D", "P", "X", "H"]
    tasks = ["<No task>"] + [f"{idx}" for idx in range(1, n_tasks + 1)]

    plt.figure(figsize=(10, 4))
    for task_finished in range(clip_array.shape[1] + 1):
        if task_finished == 0:
            y = clip_array[0, :]
            x = np.arange(y.shape[0]) + 1
            plt.plot(
                x,
                y,
                marker=markers[task_finished],
                color=colors[task_finished],
                linestyle="--",
                linewidth=2,
                markersize=7,
                label="Base",
            )
        elif task_finished == 1:
            y = clip_array[task_finished, 0]
            x = [1]
            plt.plot(
                x,
                y,
                marker=markers[task_finished],
                color=colors[task_finished],
                linestyle="",
                markersize=7,
                label=f"Model after task {task_finished}",
            )
        else:
            y = clip_array[task_finished, :]
            x = np.arange(y.shape[0]) + 1
            plt.plot(
                x,
                y,
                marker=markers[task_finished],
                color=colors[task_finished],
                linestyle="-",
                linewidth=2,
                markersize=7,
                label=f"Model after task {task_finished}",
            )
    plt.title(
        "CLIP-I Alignment on each task\nduring each phase of task-incremental training",
        fontsize=13,
    )
    plt.xlabel("Number of task", fontsize=14, color="gray")
    plt.ylabel("CLIP-I Score", fontsize=14, color="gray")
    plt.xticks(
        ticks=np.arange(1, len(tasks)), labels=tasks[1:], fontsize=12, color="gray"
    )
    plt.yticks(fontsize=12, color="gray")
    plt.legend(
        title="Model type",
        fontsize=10,
        title_fontsize="12",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.grid(True, linestyle="-", linewidth=0.7, color="white", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if name != "":
        plt.savefig(f"{save_dir}/{name}_CLIP.svg")
    plt.show()

    plt.figure(figsize=(10, 4))
    for task_finished in range(dino_array.shape[1] + 1):
        if task_finished == 0:
            y = dino_array[0, :]
            x = np.arange(y.shape[0]) + 1
            plt.plot(
                x,
                y,
                marker=markers[task_finished],
                color=colors[task_finished],
                linestyle="--",
                linewidth=2,
                markersize=7,
                label="Base",
            )
        elif task_finished == 1:
            y = dino_array[task_finished, 0]
            x = [1]
            plt.plot(
                x,
                y,
                marker=markers[task_finished],
                color=colors[task_finished],
                linestyle="",
                markersize=7,
                label=f"Model after task {task_finished}",
            )
        else:
            y = dino_array[task_finished, :]
            x = np.arange(y.shape[0]) + 1
            plt.plot(
                x,
                y,
                marker=markers[task_finished],
                color=colors[task_finished],
                linestyle="-",
                linewidth=2,
                markersize=7,
                label=f"Model after task {task_finished}",
            )
    plt.title(
        "DINO Image Alignment on each task\nduring each phase of task-incremental training",
        fontsize=13,
    )
    plt.xlabel("Number of task", fontsize=14, color="gray")
    plt.ylabel("DINO Score", fontsize=14, color="gray")
    plt.xticks(
        ticks=np.arange(1, len(tasks)), labels=tasks[1:], fontsize=12, color="gray"
    )
    plt.yticks(fontsize=12, color="gray")
    plt.legend(
        title="Model type",
        fontsize=10,
        title_fontsize="12",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.grid(True, linestyle="-", linewidth=0.7, color="white", alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if name != "":
        plt.savefig(f"{save_dir}/{name}_DINO.svg")
    plt.show()
