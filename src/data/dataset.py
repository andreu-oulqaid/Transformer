import os
from datasets import load_dataset, load_from_disk

def get_dataset(data_path="./data/multi30k"):
    DATA_PATH = data_path

    if os.path.exists(DATA_PATH) and len(os.listdir(DATA_PATH)) > 0:
        print("Loading dataset from disk...")
        dataset = load_from_disk(DATA_PATH)
    else:
        print("Downloading dataset...")
        dataset = load_dataset("bentrevett/multi30k")
        dataset.save_to_disk(DATA_PATH)

    return dataset