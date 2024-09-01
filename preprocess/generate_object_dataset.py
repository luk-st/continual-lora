import json
import os
import random
import shutil
from typing import Dict, List

SEED = 42

NUMBER_OF_OBJECTS = 10

ORIGINAL_DATASET_PATH = "data/dreambooth/dataset"
OBJECT_DATASET_PATH = "data/data_object"

OBJECT_TOKENS = ["sks", "zwz", "sbu", "uwu", "pdw", "twt", "zsn", "ofl", "psq", "cws"]

TRAIN_PROMPT_TEMPLATE = "{} {}"

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

random.seed(SEED)


def choose_n_objects(
    objects_dict: Dict[str, List[str]], n: int = NUMBER_OF_OBJECTS
) -> Dict[str, str]:
    selected_classes = random.sample(sorted(list(objects_dict.keys())), n)
    selected_objects = {
        cls: random.choice(objects_dict[cls]) for cls in selected_classes
    }
    return selected_objects


def move_selected_objects_dirs(
    selected_objects: Dict[str, str], original_path: str, new_path: str
) -> None:
    for obj in selected_objects.values():
        original_dir = os.path.join(original_path, obj)
        new_dir = os.path.join(new_path, obj)
        shutil.copytree(original_dir, new_dir, dirs_exist_ok=True)


def create_config_file(
    selected_objects: Dict[str, str], tokens: List[str], path: str
) -> None:
    tasks = []
    for idx, cls in enumerate(selected_objects.keys()):
        prompt = TRAIN_PROMPT_TEMPLATE.format(tokens[idx], cls)
        tasks.append(
            {
                "prompt": prompt,
                "path": os.path.join(path, selected_objects[cls]),
                "class": cls,
                "object": selected_objects[cls].replace("_", " "),
                "token": tokens[idx],
            }
        )

    config = {"tasks": tasks}

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    if not os.path.exists(OBJECT_DATASET_PATH):
        os.makedirs(OBJECT_DATASET_PATH)

    selected_objects = choose_n_objects(DREAMBOOTH_CLASSES)
    move_selected_objects_dirs(
        selected_objects, ORIGINAL_DATASET_PATH, OBJECT_DATASET_PATH
    )
    create_config_file(selected_objects, OBJECT_TOKENS, OBJECT_DATASET_PATH)


if __name__ == "__main__":
    main()
