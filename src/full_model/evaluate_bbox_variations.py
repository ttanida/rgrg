from ast import literal_eval
import logging
import os
import random

from datasets import Dataset
import numpy as np
import pandas as pd
import torch
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.full_model.train_full_model import get_tokenizer
from src.path_datasets_and_weights import path_full_dataset

# specify the checkpoint you want to evaluate by setting "RUN" and "CHECKPOINT"
RUN = 46
CHECKPOINT = "checkpoint_val_loss_19.793_overall_steps_155252.pt"
NUM_IMAGES_TO_EVALUATE_PER_VARIATION = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed value for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def get_tokenized_dataset(tokenizer, raw_test_dataset):
    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_test_dataset = raw_test_dataset.map(tokenize_function)

    # tokenized dataset will consist of the columns
    #   - mimic_image_file_path (str)
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])
    #   - reference_report (str)

    return tokenized_test_dataset


def get_dataset():
    usecols = [
        "mimic_image_file_path",
        "bbox_coordinates",
        "bbox_phrases",
        "bbox_phrase_exists"
    ]

    # all of the columns below are stored as strings in the csv_file
    # however, as they are actually lists, we apply the literal_eval func to convert them to lists
    converters = {
        "bbox_coordinates": literal_eval,
        "bbox_phrases": literal_eval,
        "bbox_phrase_exists": literal_eval
    }

    dataset_as_df = pd.read_csv(os.path.join(path_full_dataset, "test.csv"), usecols=usecols, converters=converters)

    # limit test images to NUM_IMAGES_TO_EVALUATE_PER_VARIATION (most likely 1000)
    dataset_as_df = dataset_as_df[:NUM_IMAGES_TO_EVALUATE_PER_VARIATION]

    raw_test_dataset = Dataset.from_pandas(dataset_as_df)

    return raw_test_dataset


def main():
    # the datasets still contain the untokenized phrases
    raw_test_dataset = get_dataset()

    tokenizer = get_tokenizer()

    # tokenize the raw datasets
    tokenized_test_dataset, tokenized_test_2_dataset = get_tokenized_dataset(tokenizer, raw_test_dataset)

    test_transforms = get_transforms()

    test_dataset_complete = CustomDataset("test", tokenized_test_dataset, test_transforms, log)
    test_2_dataset_complete = CustomDataset("test", tokenized_test_2_dataset, test_transforms, log)

    test_loader, test_2_loader = get_data_loaders(tokenizer, test_dataset_complete, test_2_dataset_complete)

    checkpoint = torch.load(
        os.path.join(path_runs_full_model, f"run_{RUN}", "checkpoints", f"{CHECKPOINT}"),
        map_location=torch.device("cpu"),
    )

    # if there is a key error when loading checkpoint, try uncommenting down below
    checkpoint["model"]["object_detector.rpn.head.conv.weight"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.weight")
    checkpoint["model"]["object_detector.rpn.head.conv.bias"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.bias")

    model = ReportGenerationModel()
    model.load_state_dict(checkpoint["model"])
    model.to(device, non_blocking=True)
    model.eval()

    del checkpoint

    evaluate_model_on_test_set(model, test_loader, test_2_loader, tokenizer)

if __name__ == "__main__":
    main()
