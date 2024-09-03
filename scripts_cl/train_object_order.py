import argparse
import json
import os
import random
import subprocess

OBJECT_DATASET_CONFIG = "data/data_object/config.json"

TRAIN_PROMPT_TEMPLATE = "a photo of {}"
VALID_PROMPT = "a {} in the jungle"

SCRIPT_PATH_LORA="./scripts_cl/train_lora_args_naive.sh"
SCRIPT_PATH_MERGE="./scripts_cl/train_lora_args_naive.sh"


DREAMBOOTH_CLASSES = {
    "backpack": ["backpack", "backpack_dog"],
    "stuffed animal": ["bear_plushie", "grey_sloth_plushie", "wolf_plushie"],
    "bowl": ["berry_bowl"],
    "can": ["can"],
    "candle": ["candle"],
    "cat": ["cat", "cat2"],
    "clock": ["clock"],
    "sneaker": ["colorful_sneaker", "shiny_sneaker"],
    "dog": ["dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8"],
    "toy": ["duck_toy", "monster_toy", "poop_emoji", "rc_car", "robot_toy"],
    "boot": ["fancy_boot"],
    "glasses": ["pink_sunglasses"],
    "cartoon": ["red_cartoon"],
    "teapot": ["teapot"],
    "vase": ["vase"],
}


def get_work_dir(experiment_name: str, object_seed: int, order_seed: int) -> str:
    return f"./models/{experiment_name}/seed_{object_seed}_object/seed_{order_seed}_order"


def main(experiment_name: str, object_seed: int, order_seed: int) -> None:
    if experiment_name in ["merge_and_init", "mag_max_light"]:
        script_path = SCRIPT_PATH_MERGE
    elif experiment_name in ["naive_cl"]:
        script_path = SCRIPT_PATH_LORA
    else:
        raise NotImplementedError(f"Unknown experiment name: {experiment_name}")

    path_to_save_models = get_work_dir(experiment_name, object_seed, order_seed)
    os.makedirs(path_to_save_models, exist_ok=True)

    # read json
    with open(OBJECT_DATASET_CONFIG, "r") as f:
        train_objects = json.load(f)

    random.seed(order_seed)
    random.shuffle(train_objects["tasks"])

    print(f"Training objects: {train_objects['tasks']}")

    serialized_lists = []

    for idx, task in enumerate(train_objects["tasks"]):
        task["index"] = idx + 1
        train_prompt = TRAIN_PROMPT_TEMPLATE.format(task["prompt"])
        valid_prompt = VALID_PROMPT.format(task["prompt"])
        task["train_prompt"] = train_prompt
        task["valid_prompt"] = valid_prompt
        serialized_lists.append(f"{task['index']},{train_prompt},{valid_prompt},{task['path']}")

    with open(os.path.join(path_to_save_models, "config.json"), "w") as f:
        json.dump(train_objects, f, indent=4)

    subprocess.call(
        [
            "sh",
            script_path,
            str(object_seed),
            path_to_save_models,
            experiment_name,
        ]
        + serialized_lists,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run token training in different order")
    parser.add_argument("--object_seed", type=int, required=True, help="Seed for training")
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
        help="Name of the experiment",
        choices=["merge_and_init", "mag_max_light", "naive_cl"],
    )

    args = parser.parse_args()
    main(args.experiment_name, args.object_seed, args.order_seed)
