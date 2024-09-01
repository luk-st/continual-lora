import argparse
import os
import random
import json
import subprocess

STYLE_DATASET_CONFIG = "data/data_style/config.json"

TRAIN_PROMPT_TEMPLATE = "{} image of {}"
VALID_PROMPT = "{} image of pen in the jungle"


def get_work_dir(experiment_name, style_seed, order_seed):
    return (
        f"./models/{experiment_name}/seed_{style_seed}_style/seed_{order_seed}_order"
    )


def save_serialized_lists(path, serialized_lists):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(serialized_lists))

def main(experiment_name, style_seed, order_seed):
    path_to_save_models = get_work_dir(experiment_name, style_seed, order_seed)
    os.makedirs(path_to_save_models, exist_ok=True)

    # read json file
    with open(STYLE_DATASET_CONFIG, "r") as f:
        train_styles = json.load(f)

    random.seed(order_seed)
    random.shuffle(train_styles["tasks"])

    serialized_lists = []

    for idx, task in enumerate(train_styles["tasks"]):
        task["index"] = idx
        task["train_prompt"] = TRAIN_PROMPT_TEMPLATE.format(task["style"], task["train_object"])
        task["valid_prompt"] = VALID_PROMPT.format(task["style"])
        serialized_lists.append(
            f"{task['index']},{task['train_prompt']},{task['valid_prompt']},{task['train_path']}"
        )

    with open(os.path.join(path_to_save_models, "config.json"), "w") as f:
        json.dump(train_styles, f, indent=4)

    subprocess.call(
        ["sh", "./scripts_cl/train_lora_args.sh", str(style_seed), path_to_save_models, experiment_name]
        + serialized_lists
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run token training in different order"
    )
    parser.add_argument(
        "--style_seed", type=int, required=True, help="Seed for training"
    )
    parser.add_argument(
        "--order_seed",
        type=int,
        required=True,
        help="Seed for shuffling the order of training",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        default="merge_and_init",
        help="Name of the experiment",
        choices=["merge_and_init", "mag_max_light"],
    )

    args = parser.parse_args()
    main(args.experiment_name, args.style_seed, args.order_seed)
