import json
import os
import random
import shutil
from typing import List, Tuple

SEED = 42

NUMBER_OF_STYLES = 10
IMAGES_PER_TRAIN_STYLE = 5

ORIGINAL_DATASET_PATH = "data/style_unlearn"
OBJECT_DATASET_PATH = "data/data_style"

random.seed(SEED)


def get_dirs_names(path: str) -> List[str]:
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def choose_n_styles(
    styles_dir: str, n: int = NUMBER_OF_STYLES
) -> Tuple[List[str], List[str], List[str]]:
    unique_objects = get_dirs_names(os.path.join(styles_dir, "Seed_Images"))

    train_objects = random.sample(unique_objects, n)
    metric_objects = [obj for obj in unique_objects if obj not in train_objects]

    styles_names = get_dirs_names(styles_dir)
    styles_names.remove("Seed_Images")
    selected_styles = random.sample(styles_names, n)

    return train_objects, metric_objects, selected_styles


def copy_images(src_dir: str, dest_dir: str, images: List[str]) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    for image in images:
        shutil.copy(os.path.join(src_dir, image), dest_dir)


def copy_styles(
    train_objects: List[str], metric_objects: List[str], selected_styles: List[str]
) -> None:
    tasks = []
    metric_objects_idx = random.choices(range(1, 21), k=len(metric_objects))

    for idx, style in enumerate(selected_styles):
        train_path = os.path.join(OBJECT_DATASET_PATH, style, "train")
        metric_path = os.path.join(OBJECT_DATASET_PATH, style, "metric")

        # Copy training images
        style_dir = os.path.join(ORIGINAL_DATASET_PATH, style, train_objects[idx])
        images = sorted(os.listdir(style_dir))
        selected_images = random.sample(images, IMAGES_PER_TRAIN_STYLE)
        copy_images(style_dir, train_path, selected_images)

        # Copy metric images
        for obj, obj_idx in zip(metric_objects, metric_objects_idx):
            img_path = os.path.join(ORIGINAL_DATASET_PATH, style, obj, f"{obj_idx}.jpg")
            output_img_path = os.path.join(metric_path, f"{obj}_{obj_idx}.jpg")
            os.makedirs(metric_path, exist_ok=True)
            shutil.copy(img_path, output_img_path)

        tasks.append(
            {
                "style": style.lower().replace("_", " "),
                "train_object": train_objects[idx].lower().replace("_", " "),
                "train_path": train_path,
                "metric_path": metric_path,
            }
        )

    with open(os.path.join(OBJECT_DATASET_PATH, "config.json"), "w") as f:
        config = {"tasks": tasks}
        json.dump(config, f, indent=4)


def main() -> None:
    train_objects, metric_objects, selected_styles = choose_n_styles(
        ORIGINAL_DATASET_PATH
    )
    copy_styles(train_objects, metric_objects, selected_styles)


if __name__ == "__main__":
    main()
