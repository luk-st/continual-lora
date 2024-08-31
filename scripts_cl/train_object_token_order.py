import argparse
import os
import random
import subprocess


OBJECT_TOKENS = ["sks", "zwz", "sbu", "uwu", "pdw", "twt", "zsn", "ofl", "psq", "cws"]

TRAIN_PROMPT_TEMPLATE = "a photo of {} {}"

VALID_PROMPTS = [
    "a {} {} riding a bicycle",
    "a {} {} in the mountains",
    "a {} {} riding a bicycle",
    "a {} {} in the in the gym",
    "a {} {} in the prison",
    "a {} {} riding a bicycle",
    "a {} {} in the mountains",
    "a {} {} riding a bicycle",
    "a {} {} in the in the gym",
    "a {} {} in the prison",
]


def get_work_dir(object_seed, order_seed):
    return f"./models/seed_{object_seed}_object/seed_{order_seed}_order"


def read_classes_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    classes_section = {}
    section = None

    for line in lines:
        line = line.strip()

        if line == "Classes":
            section = "classes"
            continue
        elif line == "Prompts" or line == "Object Prompts" or line == "Live Subject Prompts":
            break  # exit loop when prompts section is encountered

        if section == "classes" and line and ',' in line:
            subject_name, class_name = line.split(',')
            if class_name in classes_section:
                classes_section[class_name].append(subject_name)
            else:
                classes_section[class_name] = [subject_name]

    del classes_section['class']
    return classes_section


def randomly_select_objects(classes_dict):
    random.seed(42)
    selected_classes = random.sample(sorted(list(classes_dict.keys())), 10)
    selected_objects = {cls: random.choice(classes_dict[cls]) for cls in selected_classes}
    return selected_objects



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
    path_to_save_models = get_work_dir(object_seed, order_seed)
    os.makedirs(path_to_save_models, exist_ok=True)

    train_objects = randomly_select_objects(read_classes_from_file("./data/dreambooth/dataset/prompts_and_classes.txt"))

    obj_classes = list(train_objects.keys())
    obj_names = list(train_objects.values())

    train_prompts = [
        TRAIN_PROMPT_TEMPLATE.format(OBJECT_TOKENS[idx], obj_class) for idx, obj_class in enumerate(obj_classes)
    ]
    valid_prompts = [
        promt.format(OBJECT_TOKENS[idx], obj_classes[idx]) for idx, promt in enumerate(VALID_PROMPTS)
    ]
    datasets = [f"./data/dreambooth/dataset/{obj_name}" for obj_name in obj_names]


    combined_shuffled = shuffle_lists(
        order_seed, [train_prompts, valid_prompts, datasets]
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
