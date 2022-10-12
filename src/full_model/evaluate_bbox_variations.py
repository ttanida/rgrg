from ast import literal_eval
import logging
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset
from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.train_full_model import get_tokenizer
from src.path_datasets_and_weights import path_runs_full_model

# specify the checkpoint you want to evaluate by setting "RUN" and "CHECKPOINT"
RUN = 46
CHECKPOINT = "checkpoint_val_loss_19.793_overall_steps_155252.pt"

# NUM_IMAGES_TO_EVALUATE_PER_VARIATION is fixed to 1000, since the test set we use (test-1000.csv) also has exactly 1000 images
# if you want to change NUM_IMAGES_TO_EVALUATE_PER_VARIATION, then you also have to create a test set with the same number of images
# (you can set the number of rows/images to create in the csv files by setting NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES in line 67 of create_dataset.py
# to the desired number)
NUM_IMAGES_TO_EVALUATE_PER_VARIATION = 1000
IMAGE_INPUT_SIZE = 512
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed value for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def evaluate_on_position_variations(model, test_loader, tokenizer):
    mean = 0
    stds_to_evaluate = [0.1, 0.2, 0.3, 0.4, 0.5]

    # we have 1000 images, with each image having 29 bboxes, and we need 2 values to vary the bbox position in x and y direction
    num_values_to_sample = NUM_IMAGES_TO_EVALUATE_PER_VARIATION * 29 * 2

    for std in stds_to_evaluate:
        sampled_values = np.random.normal(mean, std, size=num_values_to_sample)



    for batch in tqdm(test_loader):
        pass




def get_data_loader(tokenizer, test_dataset_complete):
    custom_collate_test = CustomCollator(
        tokenizer=tokenizer, is_val_or_test=True, pretrain_without_lm_model=False
    )

    test_loader = DataLoader(
        test_dataset_complete,
        collate_fn=custom_collate_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return test_loader


def get_transforms():
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # don't apply data augmentations to test set
    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    return test_transforms


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
        "bbox_labels",
        "bbox_phrases",
        "bbox_phrase_exists",
        "bbox_is_abnormal",
        "reference_report"
    ]

    # all of the columns below are stored as strings in the csv_file
    # however, as they are actually lists, we apply the literal_eval func to convert them to lists
    converters = {
        "bbox_coordinates": literal_eval,
        "bbox_labels": literal_eval,
        "bbox_phrases": literal_eval,
        "bbox_phrase_exists": literal_eval,
        "bbox_is_abnormal": literal_eval,
    }

    dataset_as_df = pd.read_csv(os.path.join(path_full_dataset, "test.csv"), usecols=usecols, converters=converters)

    # limit test images to NUM_IMAGES_TO_EVALUATE_PER_VARIATION (most likely 1000)
    dataset_as_df = dataset_as_df[:NUM_IMAGES_TO_EVALUATE_PER_VARIATION]

    raw_test_dataset = Dataset.from_pandas(dataset_as_df)

    return raw_test_dataset


def evaluate_on_position_variations(model, tokenizer):
    log.info("Evaluating position variations.")

    mean = 0
    stds_to_evaluate = [0.1, 0.2, 0.3, 0.4, 0.5]

    # for each of the 29 bboxes in each image, we need 2 values to vary the bbox position in x and y direction
    num_values_to_sample = NUM_IMAGES_TO_EVALUATE_PER_VARIATION * 29 * 2

    for std in stds_to_evaluate:
        log.info(f"Evaluating position variation, std: {std}")
        sampled_values = np.random.normal(mean, std, size=num_values_to_sample)



    for batch in tqdm(test_loader):
        pass


def evaluate_model_on_bbox_variations(model, tokenizer):
    evaluate_on_position_variations(model, tokenizer)
    evaluate_on_scale_variations(model, tokenizer)
    evaluate_on_aspect_ratio_variations(model, tokenizer)

    # make sure varied bboxes are clipped at 0 and image width/height.
    # pass bbox through object detector to get feature vectors for each bbox
    # (pass all 29 bboxes per image, but later remove generated senteces corresponding to empty reference sentences when computing scores)
    # object_detector = model.object_detector
    # pass those features vector to language model to generate sentence for each bbox (use language_model.generate(bbox_features))
    # language_model = model.language_model
    # pass


def main():
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

    # to decode (i.e. turn into human-readable text) the generated ids by the language model
    tokenizer = get_tokenizer()

    evaluate_model_on_bbox_variations(model, tokenizer)

    # raw_test_dataset = get_dataset()

    # # note that we don't actually need to tokenize anything (i.e. we don't need the input ids and attention mask),
    # # because we evaluate the model on it's generation capabilities for different bbox variations (for which we only need the input images)
    # # but since the custom dataset and collator are build in a way that they expect input ids and attention mask
    # # (as they were originally made for training the model),
    # # it's better to just leave it as it is instead of adding unnecessary complexity
    # tokenizer = get_tokenizer()
    # tokenized_test_dataset = get_tokenized_dataset(tokenizer, raw_test_dataset)

    # test_transforms = get_transforms()

    # test_dataset_complete = CustomDataset("test", tokenized_test_dataset, test_transforms, log)
    # test_loader = get_data_loader(tokenizer, test_dataset_complete)

if __name__ == "__main__":
    main()
