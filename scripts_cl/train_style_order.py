import argparse
import os
import random
import subprocess

DATASET_PATH = "./data/unlearn-canvas-dataset"
DATASET_SCP_PATH = "./data/train_unlearn"
PROMPT_TEMPLATE = "image of {} in the style of {}"


def get_work_dir(object_seed, order_seed):
    return f"./models/seed_{object_seed}_style/seed_{order_seed}_order"


def shuffle_lists(seed, lists):
    combined = list(zip(*lists))
    random.seed(seed)
    random.shuffle(combined)
    return combined


def save_serialized_lists(path, serialized_lists):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(serialized_lists))


def generate_train_dataset():
    random.seed(42)  # set seed to be ALWAYS the same
    dirs = os.listdir(DATASET_PATH)
    random_train_styles = random.choices(dirs, k=5)
    results = [[], [], []]

    for idx, style in enumerate(random_train_styles):
        objects = os.listdir(os.path.join(DATASET_PATH, style))
        train_object, val_object = random.choices(objects, k=2)
        imgs = os.listdir(os.path.join(DATASET_PATH, style, train_object))
        chosen_img = random.choices(imgs, k=5)

        for img in chosen_img:
            os.makedirs(
                os.path.join(DATASET_SCP_PATH, style, train_object),
                exist_ok=True,
            )
            os.system(
                f"scp {os.path.join(DATASET_PATH, style, train_object, img)} {os.path.join(DATASET_SCP_PATH, style, train_object)}"
            )

        results[0].append(
            PROMPT_TEMPLATE.format(
                train_object.replace("_", " ").lower(), style.replace("_", " ").lower()
            )
        )
        results[1].append(
            PROMPT_TEMPLATE.format(
                val_object.replace("_", " ").lower(), style.replace("_", " ").lower()
            )
        )
        results[2].append(os.path.join(DATASET_SCP_PATH, style, train_object))

    return results


def main(style_seed, order_seed):
    path_to_save_models = get_work_dir(style_seed, order_seed)
    os.makedirs(path_to_save_models, exist_ok=True)

    train_dataset = generate_train_dataset()
    train_dataset = shuffle_lists(order_seed, train_dataset)

    serialized_lists = [
        f"{idx + 1},{train_prompt},{val_prompt},{train_path}"
        for idx, (train_prompt, val_prompt, train_path) in enumerate(train_dataset)
    ]
    save_serialized_lists(f"{path_to_save_models}/order_args.txt", serialized_lists)

    subprocess.call(
        ["sh", "./scripts_cl/train_lora_args.sh", str(style_seed), path_to_save_models]
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

    args = parser.parse_args()
    main(args.style_seed, args.order_seed)
