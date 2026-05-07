from datasets import load_dataset, load_from_disk
import logging
import os


def load_wikitext2(local=False, cache_dir=""):
    local_path = os.path.join(cache_dir, "wikitext-2/test")
    if local:
        dataset = load_from_disk(local_path)
    else:
        try:
            dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split="test",
            )
        except:
            logging.error(
                "Unable to download Wikitext-2 dataset from huggingface. Falling back to local version"
            )
            dataset = load_from_disk(local_path)

    return dataset


def load_c4(train, local=False, cache_dir=""):
    if train:
        local_path = os.path.join(cache_dir, "c4/train")
        if local:
            dataset = load_from_disk(local_path)
        else:
            try:
                dataset = load_dataset(
                    "allenai/c4",
                    "default",
                    data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                    split="train[:1000]",
                    revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
                )
            except:
                logging.error(
                    "Unable to download C4 dataset from huggingface. Falling back to local version"
                )
                dataset = load_from_disk(local_path)
    else:
        local_path = os.path.join(cache_dir, "c4/val")
        if local:
            dataset = load_from_disk(local_path)
        else:
            try:
                dataset = load_dataset(
                    "allenai/c4",
                    "default",
                    data_files={
                        "validation": "en/c4-validation.00000-of-00008.json.gz"
                    },
                    split="validation[:1100]",
                    revision="607bd4c8450a42878aa9ddc051a65a055450ef87",
                )
            except:
                logging.error(
                    "Unable to download C4 dataset from huggingface. Falling back to local version"
                )
                dataset = load_from_disk(local_path)

    return dataset


def load_fineweb_edu(local=False, cache_dir=""):
    local_path = os.path.join(cache_dir, "fineweb-edu/train")
    if local:
        dataset = load_from_disk(local_path)
    else:
        try:
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                "sample-10BT",
                split="train[:1100]",
                data_files=["sample/10BT/000_00000.parquet"],
                cache_dir="../data",
            )
        except:
            logging.error(
                "Unable to download fineweb-edu dataset from huggingface. Falling back to local version"
            )
            dataset = load_from_disk(local_path)

    return dataset
