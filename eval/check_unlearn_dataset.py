import os

from tqdm import tqdm


def remove_ds_store(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file == ".DS_Store":
                os.remove(os.path.join(root, file))
                print(f"Removed .DS_Store file: {os.path.join(root, file)}")


def check_unlearn_dataset(dataset_dir, metadata_file="metadata.jsonl"):
    remove_ds_store(dataset_dir)
    with open(os.path.join(dataset_dir, metadata_file), "r") as f:
        # seed images are stored twice in the metadata file
        for line in tqdm(f, desc="Checking metadata file", total=62 * 20 * 20):
            metadata = eval(line)
            style_name = metadata["file_name"]
            if not os.path.exists(os.path.join(dataset_dir, style_name)):
                print(f"Style directory does not exist: {style_name}")
                continue


if __name__ == "__main__":
    dataset_dir = "/net/tscratch/people/plgkzaleska/ziplora-analysis/data/temp/combine"
    check_unlearn_dataset(dataset_dir)
