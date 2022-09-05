"""
This module contains all functions used to evaluate the language model.

The (main) function evaluate_language_model of this module is called by the function evaluate_model in evaluate_model.py.

evaluate_language_model returns language_model_scores which include:
    - BLEU 1-4 for all generated sentences
    - BLEU 1-4 for all generated sentences with gt = normal (i.e. the region was considered normal by the radiologist)
    - BLEU 1-4 for all generated sentences with gt = abnormal (i.e. the region was considered abnormal by the radiologist).
    - BLEU 1-4, meteor, rouge-L for all generated reports

It also calls subfunctions which:
    - save NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE (see run_configurations.py) batches of generated sentences as a txt file
    (for manual verification what the model generates)
    - save NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE (see run_configurations.py) batches of generated reports as a txt file
    (for manual verification what the model generates)
    - save NUM_IMAGES_TO_PLOT (see run_configurations.py) images to tensorboard where gt and predicted bboxes for every region are depicted,
    as well as the generated sentences (if they exist) and reference sentences for every region
"""
from collections import defaultdict
import os

import evaluate
import numpy as np
import spacy
import torch
from tqdm import tqdm

from src.full_model.run_configurations import (
    BATCH_SIZE,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
    NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
    BERTSCORE_SIMILARITY_THRESHOLD,
)

from ast import literal_eval
import logging
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset
from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.run_configurations import (
    PRETRAIN_WITHOUT_LM_MODEL,
    IMAGE_INPUT_SIZE,
    PERCENTAGE_OF_VAL_SET_TO_USE,
    NUM_WORKERS,
)

device = torch.device("cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed value for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)


def get_data_loaders(tokenizer, val_dataset):
    custom_collate_val = CustomCollator(tokenizer=tokenizer, is_val=True, pretrain_without_lm_model=PRETRAIN_WITHOUT_LM_MODEL)

    g = torch.Generator()
    g.manual_seed(seed_val)

    val_loader = DataLoader(
        val_dataset,
        collate_fn=custom_collate_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return val_loader


def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # use albumentations for Compose and transforms
    # augmentations are applied with prob=0.5
    # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
    train_transforms = A.Compose(
        [
            # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
            # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
            # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
            # INTER_AREA works best for shrinking images
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.GaussianBlur(blur_limit=(1, 1)),
            A.ColorJitter(hue=0.0, saturation=0.0),  # <- reduce
            A.Sharpen(alpha=(0.1, 0.2), lightness=0.0),  # <- reduce
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            A.GaussNoise(),  # <- reduce
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    # don't apply data augmentations to val and test set
    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms


def get_tokenized_datasets(tokenizer, raw_val_dataset):
    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])

    return tokenized_val_dataset


def get_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_datasets():
    path_dataset_full_model = "/u/home/tanida/datasets/dataset-for-full-model-original-bbox-coordinates"

    usecols = [
        "mimic_image_file_path",
        "bbox_coordinates",
        "bbox_labels",
        "bbox_phrases",
        "bbox_phrase_exists",
        "bbox_is_abnormal",
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

    datasets_as_dfs = {dataset: os.path.join(path_dataset_full_model, dataset) + ".csv" for dataset in ["valid"]}

    datasets_as_dfs = {
        dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()
    }

    # bbox_phrases is a list of str
    # replace each bbox_phrase that is empty (i.e. "") by "#"
    # this is done such that model learns to generate the "#" symbol instead of "" for empty sentences
    # this is done because generated sentences that are "" (i.e. have len = 0) will cause problems when computing e.g. Bleu scores
    for dataset_df in datasets_as_dfs.values():
        dataset_df["bbox_phrases"] = dataset_df["bbox_phrases"].apply(
            lambda bbox_phrases: [phrase if len(phrase) != 0 else "#" for phrase in bbox_phrases]
        )

    total_num_samples_val = len(datasets_as_dfs["valid"])

    # compute new number of samples for both train and val
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Val: {new_num_samples_val} images")

    # limit the datasets to those new numbers
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

    return raw_val_dataset


def get_ref_sentences_for_selected_regions(reference_sentences, selected_regions):
    """
    Args:
        reference_sentences (List[List[str]]): outer list has len batch_size, inner list has len 36 (the inner list holds all reference phrases of a single image)
        selected_regions ([batch_size x 36]): boolean array that has exactly "num_regions_selected_in_batch" True values
    """
    # array of shape [batch_size x 36]
    reference_sentences = np.asarray(reference_sentences)

    ref_sentences_for_selected_regions = reference_sentences[selected_regions]

    return ref_sentences_for_selected_regions.tolist()


def update_language_model_scores(
    language_model_scores,
    generated_sentences_for_selected_regions,
    reference_sentences_for_selected_regions,
    generated_reports,
    reference_reports,
    selected_regions,
    region_is_abnormal,
):
    def get_sents_for_normal_abnormal_selected_regions():
        selected_region_is_abnormal = region_is_abnormal[selected_regions]
        # selected_region_is_abnormal is a bool array of shape [num_regions_selected_in_batch] that specifies if a selected region is abnormal (True) or normal (False)

        gen_sents_for_selected_regions = np.asarray(generated_sentences_for_selected_regions)
        ref_sents_for_selected_regions = np.asarray(reference_sentences_for_selected_regions)

        gen_sents_for_normal_selected_regions = gen_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
        gen_sents_for_abnormal_selected_regions = gen_sents_for_selected_regions[selected_region_is_abnormal].tolist()

        ref_sents_for_normal_selected_regions = ref_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
        ref_sents_for_abnormal_selected_regions = ref_sents_for_selected_regions[selected_region_is_abnormal].tolist()

        return (
            gen_sents_for_normal_selected_regions,
            gen_sents_for_abnormal_selected_regions,
            ref_sents_for_normal_selected_regions,
            ref_sents_for_abnormal_selected_regions,
        )

    def update_language_model_scores_sentence_level():
        for score in language_model_scores["all"].values():
            score.add_batch(
                predictions=generated_sentences_for_selected_regions, references=reference_sentences_for_selected_regions
            )

        # for computing the scores for the normal and abnormal reference sentences, we have to filter the generated and reference sentences accordingly
        (
            gen_sents_for_normal_selected_regions,
            gen_sents_for_abnormal_selected_regions,
            ref_sents_for_normal_selected_regions,
            ref_sents_for_abnormal_selected_regions,
        ) = get_sents_for_normal_abnormal_selected_regions()

        if len(ref_sents_for_normal_selected_regions) != 0:
            for score in language_model_scores["normal"].values():
                score.add_batch(
                    predictions=gen_sents_for_normal_selected_regions, references=ref_sents_for_normal_selected_regions
                )

        if len(ref_sents_for_abnormal_selected_regions) != 0:
            for score in language_model_scores["abnormal"].values():
                score.add_batch(
                    predictions=gen_sents_for_abnormal_selected_regions, references=ref_sents_for_abnormal_selected_regions
                )

        return gen_sents_for_abnormal_selected_regions, ref_sents_for_abnormal_selected_regions

    def update_language_model_scores_report_level():
        for score in language_model_scores["report"].values():
            score.add_batch(
                predictions=generated_reports, references=reference_reports
            )

    gen_sents_for_abnormal_selected_regions, ref_sents_for_abnormal_selected_regions = update_language_model_scores_sentence_level()
    update_language_model_scores_report_level()

    return gen_sents_for_abnormal_selected_regions, ref_sents_for_abnormal_selected_regions


def compute_final_language_model_scores(language_model_scores):
    for subset in language_model_scores:
        temp = {}
        for metric, score in language_model_scores[subset].items():
            if metric.startswith("bleu"):
                bleu_score_type = int(metric[-1])
                result = score.compute(max_order=bleu_score_type)
                temp[f"{metric}"] = result["bleu"]
            elif metric == "meteor":
                result = score.compute()
                temp["meteor"] = result["meteor"]
            elif metric == "rouge":
                result = score.compute(rouge_types=["rougeL"], use_aggregator=True)["rougeL"]
                # index 1 ^= mid (average)
                # index 2 ^= f-score
                temp["rouge"] = float(result[1][2])

        language_model_scores[subset] = temp


def get_generated_and_reference_reports(
    generated_sentences_for_selected_regions, reference_sentences, selected_regions, sentence_tokenizer
):
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): of length "num_regions_selected_in_batch"
        reference_sentences (List[List[str]]): outer list has len batch_size, inner list has len 36 (the inner list holds all reference phrases of a single image)
        selected_regions ([batch_size x 36]): boolean array that has exactly "num_regions_selected_in_batch" True values
        sentence_tokenizer: used in get_ref_report_single_image to

    Return:
        generated_reports (List[str]): list of length batch_size containing generated reports for every image in batch
        reference_reports (List[str]): list of length batch_size containing reference reports for every image in batch
        removed_similar_generated_sentences (List[Dict[str, List]): list of length batch_size containing dicts that map from one generated sentence to a list
        of other generated sentences that were removed because they were too similar. Useful for manually verifying if removing similar generated sentences was successful
    """

    def remove_duplicate_generated_sentences(gen_report_single_image, bert_score):
        def check_gen_sent_in_sents_to_be_removed(gen_sent, similar_generated_sents_to_be_removed):
            for lists_of_gen_sents_to_be_removed in similar_generated_sents_to_be_removed.values():
                if gen_sent in lists_of_gen_sents_to_be_removed:
                    return True

            return False

        # since different (closely related) regions can have the same generated sentence, we first remove exact duplicates

        # use sentence tokenizer to separate the generated sentences
        gen_sents_single_image = sentence_tokenizer(gen_report_single_image).sents

        # convert spacy.tokens.span.Span object into str by using .text attribute
        gen_sents_single_image = [sent.text for sent in gen_sents_single_image]

        # remove exact duplicates using a dict as an ordered set
        # note that dicts are insertion ordered as of Python 3.7
        gen_sents_single_image = list(dict.fromkeys(gen_sents_single_image))

        # there can still be generated sentences that are not exact duplicates, but nonetheless very similar
        # e.g. "The cardiomediastinal silhouette is normal." and "The cardiomediastinal silhouette is unremarkable."
        # to remove these "soft" duplicates, we use bertscore

        # similar_generated_sents_to_be_removed maps from one sentence to a list of similar sentences that are to be removed
        similar_generated_sents_to_be_removed = defaultdict(list)

        for i in range(len(gen_sents_single_image)):
            gen_sent_1 = gen_sents_single_image[i]

            for j in range(i + 1, len(gen_sents_single_image)):
                if check_gen_sent_in_sents_to_be_removed(gen_sent_1, similar_generated_sents_to_be_removed):
                    break

                gen_sent_2 = gen_sents_single_image[j]
                if check_gen_sent_in_sents_to_be_removed(gen_sent_2, similar_generated_sents_to_be_removed):
                    continue

                bert_score_result = bert_score.compute(
                    lang="en", predictions=[gen_sent_1], references=[gen_sent_2], model_type="distilbert-base-uncased"
                )

                if bert_score_result["f1"][0] > BERTSCORE_SIMILARITY_THRESHOLD:
                    # remove the generated similar sentence that is shorter
                    if len(gen_sent_1) > len(gen_sent_2):
                        similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                    else:
                        similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

        gen_report_single_image = " ".join(
            sent for sent in gen_sents_single_image if not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
        )

        return gen_report_single_image, similar_generated_sents_to_be_removed

    def get_generated_reports():
        bert_score = evaluate.load("bertscore")

        generated_reports = []
        removed_similar_generated_sentences = []
        curr_index = 0

        for selected_regions_single_image in selected_regions:
            # sum up all True values for a single row in the array (corresponing to a single image)
            num_selected_regions_single_image = np.sum(selected_regions_single_image)

            # use curr_index and num_selected_regions_single_image to index all generated sentences corresponding to a single image
            gen_sents_single_image = generated_sentences_for_selected_regions[
                curr_index: curr_index + num_selected_regions_single_image
            ]

            # update curr_index for next image
            curr_index += num_selected_regions_single_image

            # concatenate generated sentences of a single image to a continuous string gen_report_single_image
            gen_report_single_image = " ".join(sent for sent in gen_sents_single_image)

            gen_report_single_image, similar_generated_sents_to_be_removed = remove_duplicate_generated_sentences(
                gen_report_single_image, bert_score
            )

            generated_reports.append(gen_report_single_image)
            removed_similar_generated_sentences.append(similar_generated_sents_to_be_removed)

        return generated_reports, removed_similar_generated_sentences

    def get_ref_report_single_image(ref_sents_single_image):
        # concatenate all non-empty ref sentences (empty ref sentences are symbolized by #)
        ref_report_single_image = " ".join(sent for sent in ref_sents_single_image if sent != "#")

        # different regions can have the same or partially the same ref sentences
        # e.g. region 1 can have ref_sentence "The lung volume is low." and regions 2 the ref_sentence "The lung volume is low. There is pneumothorax."
        # to deal with those, we first split the single str ref_report_single_image back into a list of str (where each str is a single sentence)
        # using a sentence tokenizer
        ref_sents_single_image = sentence_tokenizer(ref_report_single_image).sents

        # convert spacy.tokens.span.Span object into str by using .text attribute
        ref_sents_single_image = [sent.text for sent in ref_sents_single_image]

        # we use a dict to remove duplicate sentences and put the unique sentences back together to a single str ref_report_single_image
        # note that dicts are insertion ordered as of Python 3.7
        ref_report_single_image = " ".join(dict.fromkeys(ref_sents_single_image))

        return ref_report_single_image

    def get_reference_reports():
        reference_reports = []

        # ref_sents_single_image is a List[str] containing 36 reference sentences for 36 regions of a single image
        for ref_sents_single_image in reference_sentences:
            ref_report_single_image = get_ref_report_single_image(ref_sents_single_image)
            reference_reports.append(ref_report_single_image)

        return reference_reports

    generated_reports, removed_similar_generated_sentences = get_generated_reports()
    reference_reports = get_reference_reports()

    return generated_reports, reference_reports, removed_similar_generated_sentences


def evaluate_language_model(model, val_dl, tokenizer):
    language_model_scores = {}

    # compute bleu scores for all, normal and abnormal reference sentences as well as full reports
    for subset in ["all", "normal", "abnormal", "report"]:
        language_model_scores[subset] = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}

    # compute meteor and rouge-L scores for complete reports
    language_model_scores["report"]["meteor"] = evaluate.load("meteor")
    language_model_scores["report"]["rouge"] = evaluate.load("rouge")

    gen_and_ref_sentences_to_save_to_file = {
        "generated_sentences": [],
        "reference_sentences": [],
        "generated_abnormal_sentences": [],
        "reference_abnormal_sentences": [],
    }

    gen_and_ref_reports_to_save_to_file = {
        "generated_reports": [],
        "removed_similar_generated_sentences": [],
        "reference_reports": [],
    }

    # used in function get_generated_and_reference_reports
    sentence_tokenizer = spacy.load("en_core_web_trf")

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION):
            # since generating sentences takes some time, we limit the number of batches used to compute bleu/rouge-l/meteor
            if num_batch >= NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION:
                break

            images = batch["images"]  # shape [batch_size x 1 x 512 x 512]
            region_is_abnormal = batch["region_is_abnormal"].numpy()  # boolean array of shape [batch_size x 36]

            # List[List[str]] that holds the reference phrases. The inner list holds all reference phrases of a single image
            reference_sentences = batch["reference_sentences"]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model.generate(
                    images.to(device, non_blocking=True),
                    max_length=MAX_NUM_TOKENS_GENERATE,
                    num_beams=NUM_BEAMS,
                    early_stopping=True,
                )

            # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
            if output == -1:
                continue
            else:
                beam_search_output, selected_regions, detections, class_detected = output
                selected_regions = selected_regions.detach().cpu().numpy()

            # generated_sentences_for_selected_regions is a List[str] of length "num_regions_selected_in_batch"
            generated_sentences_for_selected_regions = tokenizer.batch_decode(
                beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # filter reference_sentences to those that correspond to the generated_sentences for the selected regions.
            # reference_sentences_for_selected_regions will therefore be a List[str] of length "num_regions_selected_in_batch"
            # (i.e. same length as generated_sentences_for_selected_regions)
            reference_sentences_for_selected_regions = get_ref_sentences_for_selected_regions(
                reference_sentences, selected_regions
            )

            (
                generated_reports,
                reference_reports,
                removed_similar_generated_sentences,
            ) = get_generated_and_reference_reports(
                generated_sentences_for_selected_regions, reference_sentences, selected_regions, sentence_tokenizer
            )

            (
                gen_sents_for_abnormal_selected_regions,
                ref_sents_for_abnormal_selected_regions,
            ) = update_language_model_scores(
                language_model_scores,
                generated_sentences_for_selected_regions,
                reference_sentences_for_selected_regions,
                generated_reports,
                reference_reports,
                selected_regions,
                region_is_abnormal,
            )

            if num_batch < NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE:
                gen_and_ref_sentences_to_save_to_file["generated_sentences"].extend(
                    generated_sentences_for_selected_regions
                )
                gen_and_ref_sentences_to_save_to_file["reference_sentences"].extend(
                    reference_sentences_for_selected_regions
                )
                gen_and_ref_sentences_to_save_to_file["generated_abnormal_sentences"].extend(
                    gen_sents_for_abnormal_selected_regions
                )
                gen_and_ref_sentences_to_save_to_file["reference_abnormal_sentences"].extend(
                    ref_sents_for_abnormal_selected_regions
                )

            if num_batch < NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE:
                gen_and_ref_reports_to_save_to_file["generated_reports"].extend(generated_reports)
                gen_and_ref_reports_to_save_to_file["reference_reports"].extend(reference_reports)
                gen_and_ref_reports_to_save_to_file["removed_similar_generated_sentences"].extend(
                    removed_similar_generated_sentences
                )

    compute_final_language_model_scores(language_model_scores)

    return language_model_scores


raw_val_dataset = get_datasets()
tokenizer = get_tokenizer()

tokenized_val_dataset = get_tokenized_datasets(tokenizer, raw_val_dataset)

val_transforms = get_transforms("val")

val_dataset_complete = CustomDataset("val", tokenized_val_dataset, val_transforms, log)

val_loader = get_data_loaders(tokenizer, val_dataset_complete)

model = ReportGenerationModel(pretrain_without_lm_model=PRETRAIN_WITHOUT_LM_MODEL)
model.load_state_dict(torch.load("/u/home/tanida/runs/full_model/run_4/weights/val_loss_6.020_epoch_1.pth", map_location=torch.device("cpu")))
model.to(device, non_blocking=True)

language_model_scores = evaluate_language_model(model, val_loader, tokenizer)

for subset in language_model_scores:
    for metric, score in language_model_scores[subset].items():
        print(f"{metric}: {score}")
