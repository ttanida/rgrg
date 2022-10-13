from ast import literal_eval
import logging
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset
import imagesize
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.train_full_model import get_tokenizer
from src.path_datasets_and_weights import path_runs_full_model

# specify the checkpoint you want to evaluate by setting "RUN" and "CHECKPOINT"
RUN = 46
CHECKPOINT = "checkpoint_val_loss_19.793_overall_steps_155252.pt"
IMAGE_INPUT_SIZE = 512
BATCH_SIZE = 4

# test csv file with only 1000 images (you can create it by setting NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES in line 67 of create_dataset.py to 1000)
path_to_partial_test_set = "/u/home/tanida/datasets/dataset-with-reference-reports-partial-1000/test-1000.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)


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


def evaluate_on_position_variations(model, test_set_as_df, tokenizer):
    def vary_bbox_coords_position(row):
        def check_coordinate(coord, dimension):
            """Make sure that new coordinate is still within the image."""
            if coord < 0:
                return 0
            elif coord > dimension:
                return dimension
            else:
                return coord

        bbox_coords_single_image = row["bbox_coordinates"]  # List[List[int]] of shape 29 x 4
        bbox_widths_heights_single_image = row["bbox_widths_heights"]  # List[List[int]] of shape 29 x 2
        relative_position_variation_bboxes = row["relative_position_variations"]  # List[List[float]] of shape 29 x 2
        image_width, image_height = row["image_width_height"]  # two integers

        # to store the new bbox coordinates after they have been varied
        varied_bbox_coords_single_image = []

        for bbox_coords, bbox_width_height, relative_position_variations in zip(bbox_coords_single_image, bbox_widths_heights_single_image, relative_position_variation_bboxes):
            x1, y1, x2, y2 = bbox_coords
            bbox_width, bbox_height = bbox_width_height
            x_rel, y_rel = relative_position_variations

            # if e.g. x_rel = 0.5 and bbox_width = 100, then x_var = 50
            x_var = int(bbox_width * x_rel)
            y_var = int(bbox_height * y_rel)

            x1 += x_var
            x2 += x_var
            y1 += y_var
            y2 += y_var

            x1 = check_coordinate(x1, image_width)
            x2 = check_coordinate(x2, image_width)
            y1 = check_coordinate(y1, image_height)
            y2 = check_coordinate(y2, image_height)

            varied_bbox_coords_single_image.append([x1, y1, x2, y2])

        return varied_bbox_coords_single_image

    log.info("Evaluating position variations.")

    num_images = len(test_set_as_df)

    mean = 0
    stds_to_evaluate = [0.1, 0.2, 0.3, 0.4, 0.5]

    for std in stds_to_evaluate:
        log.info(f"Evaluating position variation, std: {std}")

        # for each of the 29 bboxes in each image, we need 2 float values to vary the bbox position in x and y direction
        # relative to the corresponding bbox width and height
        # e.g. 0.0 denotes "no change" and 0.5 denotes "half of the bbox width/height" (depending if 0.5 was sampled for x or y direction)
        relative_position_variations = np.random.normal(mean, std, size=(num_images, 29, 2))

        test_set_as_df["relative_position_variations"] = relative_position_variations.tolist()
        test_set_as_df["bbox_coordinates_varied"] = test_set_as_df.apply(lambda row: vary_bbox_coords_position(row), axis=1)



    # for batch in tqdm(test_loader):
    #     pass


def evaluate_model_on_bbox_variations(model, test_set_as_df, tokenizer):
    evaluate_on_position_variations(model, test_set_as_df, tokenizer)
    evaluate_on_scale_variations(model, test_set_as_df, tokenizer)
    evaluate_on_aspect_ratio_variations(model, test_set_as_df, tokenizer)

    # make sure varied bboxes are clipped at 0 and image width/height.
    # pass bbox through object detector to get feature vectors for each bbox
    # (pass all 29 bboxes per image, but later remove generated senteces corresponding to empty reference sentences when computing scores)
    # object_detector = model.object_detector
    # pass those features vector to language model to generate sentence for each bbox (use language_model.generate(bbox_features))
    # language_model = model.language_model
    # pass


def get_test_set_as_df():
    def compute_bbox_widths_heights(row):
        bbox_coordinates_single_image = row["bbox_coordinates"]
        widths_heights = []
        for bbox_coords in bbox_coordinates_single_image:
            x1, y1, x2, y2 = bbox_coords
            width = x2 - x1
            height = y2 - y1
            widths_heights.append([width, height])

        return widths_heights

    def retrieve_image_widths_heights(row):
        mimic_image_file_path = row["mimic_image_file_path"]
        width, height = imagesize.get(mimic_image_file_path)
        return [width, height]

    usecols = [
        "mimic_image_file_path",
        "bbox_coordinates",
        "bbox_labels",
        "bbox_phrases",
        "bbox_phrase_exists",
    ]

    # all of the columns below are stored as strings in the csv_file
    # however, as they are actually lists, we apply the literal_eval func to convert them to lists
    converters = {
        "bbox_coordinates": literal_eval,
        "bbox_labels": literal_eval,
        "bbox_phrases": literal_eval,
        "bbox_phrase_exists": literal_eval,
    }

    test_set_as_df = pd.read_csv(path_to_partial_test_set, usecols=usecols, converters=converters)

    # add new columns that contain the bbox_widths_heights (List[List[int]] with len(outer_list)=29 and len(inner_list) = 2)
    # and image_width_height (List[int] of len 2)
    test_set_as_df["bbox_widths_heights"] = test_set_as_df.apply(lambda row: compute_bbox_widths_heights(row), axis=1)
    test_set_as_df["image_width_height"] = test_set_as_df.apply(lambda row: retrieve_image_widths_heights(row), axis=1)

    return test_set_as_df


def get_model():
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

    return model


def main():
    model = get_model()
    test_set_as_df = get_test_set_as_df()
    tokenizer = get_tokenizer()  # to decode (i.e. turn into human-readable text) the generated ids by the language model

    evaluate_model_on_bbox_variations(model, test_set_as_df, tokenizer)

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
