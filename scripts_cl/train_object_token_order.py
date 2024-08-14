import argparse
import os
import random
import subprocess

TRAIN_PROMPTS = [
    "a photo of sks stuffed animal",
    "a photo of zwz backpack",
    "a photo of sbu dog",
    "a photo of uwu candle",
    "a photo of pdw cat",
]

VALID_PROMPTS = [
    "a sks stuffed animal riding a bicycle",
    "a zwz backpack in the mountains",
    "a sbu dog riding a bicycle",
    "a uwu candle in the in the gym",
    "a pdw cat in the prison",
]

DATASETS = [
    f"./data/dreambooth/dataset/{ds_name}"
    for ds_name in [
        "wolf_plushie",
        "backpack",
        "dog6",
        "candle",
        "cat2",
    ]
]


def get_work_dir(object_seed, order_seed):
    return f"./models/seed_{object_seed}_object/seed_{order_seed}_order"


def shuffle_lists(seed, lists):
    combined = list(zip(*lists))
    random.seed(seed)
    random.shuffle(combined)
    return combined

def save_serialized_lists(path, serialized_lists):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(serialized_lists))


def main(object_seed, order_seed):
    order_seed = order_seed
    path_to_save_models = get_work_dir(object_seed, order_seed)

    combined_shuffled = shuffle_lists(
        order_seed, [TRAIN_PROMPTS, VALID_PROMPTS, DATASETS]
    )

    print(f"Items shuffled with seed {order_seed}:")
    print(f"\t{combined_shuffled}")
    print(f"Path to save models: {path_to_save_models}")

    serialized_lists = [
        f"{idx + 1},{train_prompt},{valid_prompt},{dataset_dir}"
        for idx, (train_prompt, valid_prompt, dataset_dir) in enumerate(
            combined_shuffled
        )
    ]

    save_serialized_lists(os.path.join(path_to_save_models, "order_args.txt"), serialized_lists)

    subprocess.call(
        ["sh", "./scripts_cl/train_lora_args.sh", str(object_seed), path_to_save_models]
        + serialized_lists
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run token training in different order"
    )
    parser.add_argument(
        "--object_seed", type=int, required=True, help="Seed for training"
    )
    parser.add_argument(
        "--order_seed",
        type=int,
        required=True,
        help="Seed for shuffling the order of training",
    )

    args = parser.parse_args()
    main(args.object_seed, args.order_seed)
