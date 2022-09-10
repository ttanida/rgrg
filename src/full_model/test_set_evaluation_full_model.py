from ast import literal_eval
import logging
import os
import random
from collections import defaultdict

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchmetrics
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset
from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.run_configurations import (
    IMAGE_INPUT_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
)
import evaluate
import spacy

PRETRAIN_WITHOUT_LM_MODEL = False
BERTSCORE_SIMILARITY_THRESHOLD = 0.9
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 30
NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE = 30
NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed value for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


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


def write_sentences_and_reports_to_file(
    gen_and_ref_sentences_to_save_to_file,
    gen_and_ref_reports_to_save_to_file
):
    def write_sentences(generated_sentences, reference_sentences, is_abnormal):
        txt_file_name = f"generated{'' if not is_abnormal else '_abnormal'}_sentences"
        txt_file_name = os.path.join("/u/home/tanida/region-guided-chest-x-ray-report-generation/src/full_model", txt_file_name)

        with open(txt_file_name, "w") as f:
            for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
                f.write(f"Generated sentence: {gen_sent}\n")
                # the hash symbol symbolizes an empty reference sentence, and thus can be replaced by '' when writing to file
                f.write(f"Reference sentence: {ref_sent if ref_sent != '#' else ''}\n\n")

    def write_reports(generated_reports, reference_reports, removed_similar_generated_sentences):
        txt_file_name = os.path.join(
            "/u/home/tanida/region-guided-chest-x-ray-report-generation/src/full_model",
            "generated_reports",
        )

        with open(txt_file_name, "w") as f:
            for gen_report, ref_report, removed_similar_gen_sents in zip(
                generated_reports, reference_reports, removed_similar_generated_sentences
            ):
                f.write(f"Generated report: {gen_report}\n\n")
                f.write(f"Reference report: {ref_report}\n\n")
                f.write("Generated sentences that were removed:\n")
                for gen_sent, list_similar_gen_sents in removed_similar_gen_sents.items():
                    f.write(f"\t{gen_sent} == {list_similar_gen_sents}\n")
                f.write("\n")
                f.write("=" * 30)
                f.write("\n\n")

    # generated_sentences is a list of str
    generated_sentences = gen_and_ref_sentences_to_save_to_file["generated_sentences"]
    generated_abnormal_sentences = gen_and_ref_sentences_to_save_to_file["generated_abnormal_sentences"]

    # reference_sentences is a list of str
    reference_sentences = gen_and_ref_sentences_to_save_to_file["reference_sentences"]
    reference_abnormal_sentences = gen_and_ref_sentences_to_save_to_file["reference_abnormal_sentences"]

    write_sentences(generated_sentences, reference_sentences, is_abnormal=False)
    write_sentences(generated_abnormal_sentences, reference_abnormal_sentences, is_abnormal=True)

    # generated_reports and reference_reports are list of str
    generated_reports = gen_and_ref_reports_to_save_to_file["generated_reports"]
    reference_reports = gen_and_ref_reports_to_save_to_file["reference_reports"]
    removed_similar_generated_sentences = gen_and_ref_reports_to_save_to_file["removed_similar_generated_sentences"]
    write_reports(generated_reports, reference_reports, removed_similar_generated_sentences)


def get_generated_sentence_for_region(
    generated_sentences_for_selected_regions, selected_regions, num_img, region_index
) -> str:
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): holds the generated sentences for all regions that were selected in the batch, i.e. of length "num_regions_selected_in_batch"
        selected_regions (Tensor[bool]): of shape [batch_size x 36], specifies for each region if it was selected to get a sentences generated (True) or not by the binary classifier for region selection.
        Ergo has exactly "num_regions_selected_in_batch" True values.
        num_img (int): specifies the image we are currently processing in the batch, its value is in the range [0, batch_size-1]
        region_index (int): specifies the region we are currently processing of a single image, its value is in the range [0, 35]

    Returns:
        str: generated sentence for region specified by num_img and region_index

    Implementation is not too easy to understand, so here is a toy example with some toy values to explain.

    generated_sentences_for_selected_regions = ["Heart is ok.", "Spine is ok."]
    selected_regions = [
        [False, False, True],
        [True, False, False]
    ]
    num_img = 0
    region_index = 2

    In this toy example, the batch_size = 2 and there are only 3 regions in total for simplicity (instead of the 36).
    The generated_sentences_for_selected_regions is of len 2, meaning num_regions_selected_in_batch = 2.
    Therefore, the selected_regions boolean tensor also has exactly 2 True values.

    (1) Flatten selected_regions:
        selected_regions_flat = [False, False, True, True, False, False]

    (2) Compute cumsum (to get an incrementation each time there is a True value):
        cum_sum_true_values = [0, 0, 1, 2, 2, 2]

    (3) Reshape cum_sum_true_values to shape of selected_regions
        cum_sum_true_values = [
            [0, 0, 1],
            [2, 2, 2]
        ]

    (4) Subtract 1 from tensor, such that 1st True value in selected_regions has the index value 0 in cum_sum_true_values,
        the 2nd True value has index value 1 and so on.
        cum_sum_true_values = [
            [-1, -1, 0],
            [1, 1, 1]
        ]

    (5) Index cum_sum_true_values with num_img and region_index to get the final index for the generated sentence list
        index = cum_sum_true_values[num_img][region_index] = cum_sum_true_values[0][2] = 0

    (6) Get generated sentence:
        generated_sentences_for_selected_regions[index] = "Heart is ok."
    """
    selected_regions_flat = selected_regions.reshape(-1)
    cum_sum_true_values = np.cumsum(selected_regions_flat)

    cum_sum_true_values = cum_sum_true_values.reshape(selected_regions.shape)
    cum_sum_true_values -= 1

    index = cum_sum_true_values[num_img][region_index]

    return generated_sentences_for_selected_regions[index]


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


def evaluate_language_model(model, test_loader, tokenizer):
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

    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False

    # used in function get_generated_and_reference_reports
    sentence_tokenizer = spacy.load("en_core_web_trf")

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(test_loader), total=NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION):
            if num_batch >= NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION:
                break

            images = batch["images"]  # shape [batch_size x 1 x 512 x 512]
            region_is_abnormal = batch["region_is_abnormal"].numpy()  # boolean array of shape [batch_size x 36]

            # List[List[str]] that holds the reference phrases. The inner list holds all reference phrases of a single image
            reference_sentences = batch["reference_sentences"]

            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model.generate(
                        images.to(device, non_blocking=True),
                        max_length=MAX_NUM_TOKENS_GENERATE,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True
                    print(f"OOM at batch number {num_batch}.\n")
                    print(f"Error message: {str(e)}\n\n")
                else:
                    raise e

            if oom:
                # free up memory
                torch.cuda.empty_cache()
                oom = False
                continue

            # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
            if output == -1:
                print(f"Empty region features before language model at batch number {num_batch}.\n\n")

                continue
            else:
                beam_search_output, selected_regions, _, _ = output
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

    write_sentences_and_reports_to_file(
        gen_and_ref_sentences_to_save_to_file,
        gen_and_ref_reports_to_save_to_file
    )

    compute_final_language_model_scores(language_model_scores)

    return language_model_scores


def print_all_scores(
    obj_detector_scores,
    region_selection_scores,
    region_abnormal_scores,
    language_model_scores,
):
    def write_obj_detector_scores(obj_detector_scores, txt_file_name):
        print(f"avg_num_detected_regions_per_image: {obj_detector_scores['avg_num_detected_regions_per_image']}")

        # replace white space by underscore for each region name (i.e. "right upper lung" -> "right_upper_lung")
        anatomical_regions = ["_".join(region.split()) for region in ANATOMICAL_REGIONS]
        avg_detections_per_region = obj_detector_scores["avg_detections_per_region"]
        avg_iou_per_region = obj_detector_scores["avg_iou_per_region"]

        for region_, avg_detections_region in zip(anatomical_regions, avg_detections_per_region):
            with open(txt_file_name, "a") as f:
                f.write(f"num_detected_{region_}: {avg_detections_region}")

        for region_, avg_iou_region in zip(anatomical_regions, avg_iou_per_region):
            with open(txt_file_name, "a") as f:
                f.write(f"iou_{region_}: {avg_iou_region}")

    def write_region_selection_scores(region_selection_scores, txt_file_name):
        for subset in region_selection_scores:
            for metric, score in region_selection_scores[subset].items():
                with open(txt_file_name, "a") as f:
                    f.write(f"region_select_{subset}_{metric}: {score}")

    def write_region_abnormal_scores(region_abnormal_scores, txt_file_name):
        for metric, score in region_abnormal_scores.items():
            with open(txt_file_name, "a") as f:
                f.write(f"region_abnormal_{metric}: {score}")

    def write_language_model_scores(language_model_scores, txt_file_name):
        for subset in language_model_scores:
            for metric, score in language_model_scores[subset].items():
                with open(txt_file_name, "a") as f:
                    f.write(f"language_model_{subset}_{metric}: {score}")

    txt_file_name = os.path.join(
        "/u/home/tanida/region-guided-chest-x-ray-report-generation/src/full_model",
        "final_scores",
    )

    write_obj_detector_scores(obj_detector_scores, txt_file_name)
    write_region_selection_scores(region_selection_scores, txt_file_name)
    write_region_abnormal_scores(region_abnormal_scores, txt_file_name)
    write_language_model_scores(language_model_scores, txt_file_name)


def update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, region_is_abnormal, class_detected):
    """
    Args:
        region_abnormal_scores (Dict)
        predicted_abnormal_regions (Tensor[bool]): shape [batch_size x 36]
        region_is_abnormal (Tensor[bool]): shape [batch_size x 36]
        class_detected (Tensor[bool]): shape [batch_size x 36]

    We only update/compute the scores for regions that were actually detected by the object detector (specified by class_detected).
    """
    detected_predicted_abnormal_regions = predicted_abnormal_regions[class_detected]
    detected_region_is_abnormal = region_is_abnormal[class_detected]

    region_abnormal_scores["precision"](detected_predicted_abnormal_regions, detected_region_is_abnormal)
    region_abnormal_scores["recall"](detected_predicted_abnormal_regions, detected_region_is_abnormal)
    region_abnormal_scores["f1"](detected_predicted_abnormal_regions, detected_region_is_abnormal)


def update_region_selection_metrics(region_selection_scores, selected_regions, region_has_sentence, region_is_abnormal):
    """
    Args:
        region_selection_scores (Dict[str, Dict])
        selected_regions (Tensor[bool]): shape [batch_size x 36]
        region_has_sentence (Tensor[bool]): shape [batch_size x 36]
        region_is_abnormal (Tensor[bool]): shape [batch_size x 36]
    """
    normal_selected_regions = selected_regions[~region_is_abnormal]
    normal_region_has_sentence = region_has_sentence[~region_is_abnormal]

    abnormal_selected_regions = selected_regions[region_is_abnormal]
    abnormal_region_has_sentence = region_has_sentence[region_is_abnormal]

    region_selection_scores["all"]["precision"](selected_regions.reshape(-1), region_has_sentence.reshape(-1))
    region_selection_scores["all"]["recall"](selected_regions.reshape(-1), region_has_sentence.reshape(-1))
    region_selection_scores["all"]["f1"](selected_regions.reshape(-1), region_has_sentence.reshape(-1))

    region_selection_scores["normal"]["precision"](normal_selected_regions, normal_region_has_sentence)
    region_selection_scores["normal"]["recall"](normal_selected_regions, normal_region_has_sentence)
    region_selection_scores["normal"]["f1"](normal_selected_regions, normal_region_has_sentence)

    region_selection_scores["abnormal"]["precision"](abnormal_selected_regions, abnormal_region_has_sentence)
    region_selection_scores["abnormal"]["recall"](abnormal_selected_regions, abnormal_region_has_sentence)
    region_selection_scores["abnormal"]["f1"](abnormal_selected_regions, abnormal_region_has_sentence)


def update_object_detector_metrics(obj_detector_scores, detections, image_targets, class_detected):
    def compute_box_area(box):
        """
        Calculate the area of a box given the 4 corner values.

        Args:
            box (Tensor[batch_size x 36 x 4])

        Returns:
            area (Tensor[batch_size x 36])
        """
        x0 = box[..., 0]
        y0 = box[..., 1]
        x1 = box[..., 2]
        y1 = box[..., 3]

        return (x1 - x0) * (y1 - y0)

    def compute_intersection_and_union_area_per_region(detections, targets, class_detected):
        # pred_boxes is of shape [batch_size x 36 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
        # they are sorted in the 2nd dimension, meaning the 1st of the 36 boxes corresponds to the 1st region/class,
        # the 2nd to the 2nd class and so on
        pred_boxes = detections["top_region_boxes"]

        # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
        # gt_boxes is of shape [batch_size x 36 x 4]
        gt_boxes = torch.stack([t["boxes"] for t in targets], dim=0)

        # below tensors are of shape [batch_size x 36]
        x0_max = torch.maximum(pred_boxes[..., 0], gt_boxes[..., 0])
        y0_max = torch.maximum(pred_boxes[..., 1], gt_boxes[..., 1])
        x1_min = torch.minimum(pred_boxes[..., 2], gt_boxes[..., 2])
        y1_min = torch.minimum(pred_boxes[..., 3], gt_boxes[..., 3])

        # intersection_boxes is of shape [batch_size x 36 x 4]
        intersection_boxes = torch.stack([x0_max, y0_max, x1_min, y1_min], dim=-1)

        # below tensors are of shape [batch_size x 36]
        intersection_area = compute_box_area(intersection_boxes)
        pred_area = compute_box_area(pred_boxes)
        gt_area = compute_box_area(gt_boxes)

        union_area = (pred_area + gt_area) - intersection_area

        # if x0_max >= x1_min or y0_max >= y1_min, then there is no intersection
        valid_intersection = torch.logical_and(x0_max < x1_min, y0_max < y1_min)

        # also there is no intersection if the class was not detected by object detector
        valid_intersection = torch.logical_and(valid_intersection, class_detected)

        # set all non-valid intersection areas to 0
        intersection_area = torch.where(
            valid_intersection,
            intersection_area,
            torch.tensor(0, dtype=intersection_area.dtype, device=intersection_area.device),
        )

        # sum up the values along the batch dimension (the values will divided by each other later to get the averages)
        intersection_area = torch.sum(intersection_area, dim=0)
        union_area = torch.sum(union_area, dim=0)

        return intersection_area, union_area

    # sum up detections for each region
    region_detected_batch = torch.sum(class_detected, dim=0)

    intersection_area_per_region_batch, union_area_per_region_batch = compute_intersection_and_union_area_per_region(detections, image_targets, class_detected)

    obj_detector_scores["sum_region_detected"] += region_detected_batch
    obj_detector_scores["sum_intersection_area_per_region"] += intersection_area_per_region_batch
    obj_detector_scores["sum_union_area_per_region"] += union_area_per_region_batch


def evaluate_model(model, test_loader, tokenizer):
    obj_detector_scores, region_selection_scores, region_abnormal_scores = get_test_losses_and_other_metrics(model, test_loader)

    language_model_scores = evaluate_language_model(model, test_loader, tokenizer)

    print_all_scores(
        obj_detector_scores,
        region_selection_scores,
        region_abnormal_scores,
        language_model_scores,
    )


def get_data_loaders(tokenizer, test_dataset_complete):
    custom_collate_test = CustomCollator(tokenizer=tokenizer, is_val=True, pretrain_without_lm_model=PRETRAIN_WITHOUT_LM_MODEL)

    test_loader = DataLoader(
        test_dataset_complete,
        collate_fn=custom_collate_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return test_loader


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


def get_tokenized_datasets(tokenizer, raw_test_dataset):
    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_test_dataset = raw_test_dataset.map(tokenize_function)

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])

    return tokenized_test_dataset


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

    datasets_as_dfs = {dataset: os.path.join(path_dataset_full_model, dataset) + ".csv" for dataset in ["test"]}

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

    raw_test_dataset = Dataset.from_pandas(datasets_as_dfs["test"])

    return raw_test_dataset


def get_test_losses_and_other_metrics(model, test_loader):
    obj_detector_scores = {}
    obj_detector_scores["sum_intersection_area_per_region"] = torch.zeros(36, device=device)
    obj_detector_scores["sum_union_area_per_region"] = torch.zeros(36, device=device)
    obj_detector_scores["sum_region_detected"] = torch.zeros(36, device=device)

    region_selection_scores = {}
    for subset in ["all", "normal", "abnormal"]:
        region_selection_scores[subset] = {
            "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
            "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
            "f1": torchmetrics.F1Score(num_classes=2, average=None).to(device),
        }

    region_abnormal_scores = {
        "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
        "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
        "f1": torchmetrics.F1Score(num_classes=2, average=None).to(device),
    }

    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False

    num_images = 0

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(test_loader)):
            images = batch["images"]
            image_targets = batch["image_targets"]
            region_has_sentence = batch["region_has_sentence"]
            region_is_abnormal = batch["region_is_abnormal"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            batch_size = images.size(0)
            num_images += batch_size

            images = images.to(device, non_blocking=True)
            image_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in image_targets]
            region_has_sentence = region_has_sentence.to(device, non_blocking=True)
            region_is_abnormal = region_is_abnormal.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(images, image_targets, input_ids, attention_mask, region_has_sentence, region_is_abnormal)
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True

                    print(f"OOM at batch number {num_batch}.\n")
                    print(f"Error message: {str(e)}\n\n")

                    raise e

            if oom:
                # free up memory
                torch.cuda.empty_cache()
                oom = False

                num_images -= batch_size

                continue

            # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
            if output == -1:
                print(f"Empty region features before language model at batch number {num_batch}.\n\n")

                num_images -= batch_size

                continue

            (
                _,
                _,
                _,
                _,
                detections,
                class_detected,  # bool tensor of shape [batch_size x 36]
                selected_regions,  # bool tensor of shape [batch_size x 36]
                predicted_abnormal_regions,  # bool tensor of shape [batch_size x 36]
            ) = output

            # update scores for object detector metrics
            update_object_detector_metrics(obj_detector_scores, detections, image_targets, class_detected)

            # update scores for region selection metrics
            update_region_selection_metrics(region_selection_scores, selected_regions, region_has_sentence, region_is_abnormal)

            # update scores for region abnormal detection metrics
            update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, region_is_abnormal, class_detected)

    # compute object detector scores
    sum_intersection = obj_detector_scores["sum_intersection_area_per_region"]
    sum_union = obj_detector_scores["sum_union_area_per_region"]
    obj_detector_scores["avg_iou_per_region"] = (sum_intersection / sum_union).tolist()

    sum_region_detected = obj_detector_scores["sum_region_detected"]
    obj_detector_scores["avg_num_detected_regions_per_image"] = torch.sum(sum_region_detected / num_images).item()
    obj_detector_scores["avg_detections_per_region"] = (sum_region_detected / num_images).tolist()

    # compute the "micro" average scores for region_selection_scores
    for subset in region_selection_scores:
        for metric, score in region_selection_scores[subset].items():
            region_selection_scores[subset][metric] = score.compute()[1].item()  # only report results for the positive class (hence [1])

    # compute the "micro" average scores for region_abnormal_scores
    for metric, score in region_abnormal_scores.items():
        region_abnormal_scores[metric] = score.compute()[1].item()

    return obj_detector_scores, region_selection_scores, region_abnormal_scores


def main():
    # the datasets still contain the untokenized phrases
    raw_test_dataset = get_datasets()

    tokenizer = get_tokenizer()

    # tokenize the raw datasets
    tokenized_test_dataset = get_tokenized_datasets(tokenizer, raw_test_dataset)

    test_transforms = get_transforms("test")

    test_dataset_complete = CustomDataset("test", tokenized_test_dataset, test_transforms, log)

    test_loader = get_data_loaders(tokenizer, test_dataset_complete)

    checkpoint = torch.load(
        "/u/home/tanida/runs/full_model/run_20/checkpoints/checkpoint_val_loss_7.256_epoch_2.pt", map_location=torch.device("cpu")
    )

    model = ReportGenerationModel(pretrain_without_lm_model=PRETRAIN_WITHOUT_LM_MODEL)
    model.load_state_dict(checkpoint["model"])
    model.to(device, non_blocking=True)
    model.eval()

    evaluate_model(model, test_loader, tokenizer)


if __name__ == "__main__":
    main()
