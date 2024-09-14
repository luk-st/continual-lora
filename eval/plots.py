import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_incremental_performance_heatmap(
    array, metric_name, name="", save_dir="."
):
    tasks_x = [f"{idx}" for idx in range(1, array.shape[0]+1)]
    tasks_y = [f"{idx}" for idx in range(1, array.shape[1]+1)]
    colormap = "Greens"

    plt.rcParams.update(
        {
            "axes.facecolor": "#f0f0f0",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "gray",
            "axes.linewidth": 1.0,
            "axes.labelcolor": "gray",
            "xtick.color": "gray",
            "ytick.color": "gray",
        }
    )

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        array,
        annot=True,
        fmt=".3f",
        cmap=colormap,
        xticklabels=tasks_x,
        yticklabels=tasks_y,
        linewidths=0.5,
        linecolor="black",
        # vmin=0.2 if metric_name == "CSD" else 0.15, 
        # vmax=0.65 if metric_name == "CSD" else 0.5,
        # cbar_kws={"ticks": [0.3, 0.4, 0.5, 0.6] if metric_name == "CSD" else [0.2, 0.3, 0.4], "alpha": 1.0, "shrink": 0.9},
        annot_kws={"size": 13},
    )

    # plt.title(f"Task-incremental {metric_name} Alignment on each task")
    plt.xlabel("Model after task", fontsize=20)
    plt.ylabel("Performance on task", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15, rotation=0)
    plt.tight_layout(rect=[0, 0, 1, 1])
    if name != "":
        plt.savefig(f"{save_dir}/{name}_{metric_name}_heatmap.pdf")
    else:
        plt.show()

def plot_incremental_performance_plot(
    array, metric_name, name="", save_dir="."
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
    colors = plt.cm.viridis(np.linspace(0, 1, array.shape[1] + 1))
    tasks = [f"{idx}" for idx in range(1, array.shape[0]+1)]

    plt.figure(figsize=(10, 4))
    for task_finished in range(0, array.shape[1]):
        y = array[task_finished, :]
        x = np.arange(y.shape[0])
        plt.plot(
            x,
            y,
            color=colors[task_finished],
            linestyle="-",
            linewidth=2,
            markersize=7,
            label=f"Model after task {task_finished}",
        )

    plt.title(
    f"Task-incremental {metric_name} Alignment on each task",
        fontsize=13,
    )
    plt.xlabel("Number of task", fontsize=14, color="gray")
    plt.ylabel(f"{metric_name} Score", fontsize=14, color="gray")
    plt.xticks(
        ticks=np.arange(0, len(tasks)), labels=tasks, fontsize=12, color="gray"
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
        plt.savefig(f"{save_dir}/{name}_{metric_name}_plot.svg")
    else:
        plt.show()