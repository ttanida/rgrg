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
import spacy
import torch
from torch.utils.data import DataLoader
import torchmetrics
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset
from src.full_model.evaluate_full_model.evaluate_model import (
    update_object_detector_metrics,
    update_region_abnormal_metrics,
    update_region_selection_metrics,
)
from src.full_model.evaluate_full_model.evaluate_language_model import (
    get_ref_sentences_for_selected_regions,
    get_sents_for_normal_abnormal_selected_regions,
    get_generated_reports,
    update_gen_and_ref_sentences_for_regions,
    update_num_generated_sentences_per_image,
    update_gen_sentences_with_corresponding_regions,
    compute_language_model_scores
)
from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.train_full_model import get_tokenizer
from src.path_datasets_and_weights import path_full_dataset, path_runs_full_model, path_test_set_evaluation_scores_txt_files

# specify the checkpoint you want to evaluate by setting "RUN" and "CHECKPOINT"
RUN = 38
CHECKPOINT = "checkpoint_val_loss_20.850_overall_steps_195284.pt"
BERTSCORE_SIMILARITY_THRESHOLD = 0.9
IMAGE_INPUT_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 10
NUM_BEAMS = 4
MAX_NUM_TOKENS_GENERATE = 300
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 100
NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed value for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

"""
Folder specified by path_test_set_evaluation_scores_txt_files will have these files after test set evaluation:

    - final_scores.txt
    - generated_sentences.txt
    - generated_abnormal_sentences.txt
    - generated_reports.txt
"""
final_scores_txt_file = os.path.join(path_test_set_evaluation_scores_txt_files, "final_scores.txt")


def write_all_scores_to_file(
    obj_detector_scores,
    region_selection_scores,
    region_abnormal_scores,
    language_model_scores
):
    def write_obj_detector_scores():
        with open(final_scores_txt_file, "a") as f:
            f.write(f"avg_num_detected_regions_per_image: {obj_detector_scores['avg_num_detected_regions_per_image']:.5f}\n")
            f.write(f"avg_iou: {obj_detector_scores['avg_iou']:.5f}\n")

        # replace white space by underscore for each region name (i.e. "right upper lung" -> "right_upper_lung")
        anatomical_regions = ["_".join(region.split()) for region in ANATOMICAL_REGIONS]
        avg_detections_per_region = obj_detector_scores["avg_detections_per_region"]
        avg_iou_per_region = obj_detector_scores["avg_iou_per_region"]

        for region_, avg_detections_region in zip(anatomical_regions, avg_detections_per_region):
            with open(final_scores_txt_file, "a") as f:
                f.write(f"num_detected_{region_}: {avg_detections_region:.5f}\n")

        for region_, avg_iou_region in zip(anatomical_regions, avg_iou_per_region):
            with open(final_scores_txt_file, "a") as f:
                f.write(f"iou_{region_}: {avg_iou_region:.5f}\n")

    def write_region_selection_scores():
        for subset in region_selection_scores:
            for metric, score in region_selection_scores[subset].items():
                with open(final_scores_txt_file, "a") as f:
                    f.write(f"region_select_{subset}_{metric}: {score:.5f}\n")

    def write_region_abnormal_scores():
        for metric, score in region_abnormal_scores.items():
            with open(final_scores_txt_file, "a") as f:
                f.write(f"region_abnormal_{metric}: {score:.5f}\n")

    def write_clinical_efficacy_scores(ce_score_dict):
        """
        ce_score_dict is of the structure:

        {
            precision_micro_5: ...,
            precision_micro_all: ...,
            precision_example_all: ...,
            recall_micro_5: ...,
            recall_micro_all: ...,
            recall_example_all: ...,
            f1_micro_5: ...,
            f1_micro_all: ...,
            f1_example_all: ...,
            acc_micro_5: ...,
            acc_micro_all: ...,
            acc_example_all: ...,
            condition_1 : {
                precision: ...,
                recall: ...,
                f1: ...,
                acc: ...,
            },
            condition_2 : {
                precision: ...,
                recall: ...,
                f1: ...,
                acc: ...,
            },
            ...,
            condition_14 : {
                precision: ...,
                recall: ...,
                f1: ...,
                acc: ...,
            }
        }

        where the "..." after the 4 metrics are the corresponding scores,
        and condition_* are from the 14 conditions in src/CheXbert/src/constants.py
        """
        for k, v in ce_score_dict.items():
            if k.startswith("precision") or k.startswith("recall") or k.startswith("f1") or k.startswith("acc"):
                with open(final_scores_txt_file, "a") as f:
                    f.write(f"report_CE_{k}: {v:.5f}\n")
            else:
                # k is a condition
                condition_name = "_".join(k.lower().split())
                for metric, score in ce_score_dict[k].items():
                    with open(final_scores_txt_file, "a") as f:
                        f.write(f"report_CE_{condition_name}_{metric}: {score:.5f}\n")

    def write_language_model_scores():
        """
        language_model_scores is a dict with keys:
            - all: for all generated sentences
            - normal: for all generated sentences corresponding to normal regions
            - abnormal: for all generated sentences corresponding to abnormal regions
            - report: for all generated reports
            - region: for generated sentences per region
        """
        for subset in language_model_scores:
            if subset == "region":
                for region_name in language_model_scores["region"]:
                    for metric, score in language_model_scores["region"][region_name].items():
                        # replace white space by underscore for region name (i.e. "right upper lung" -> "right_upper_lung")
                        region_name_underscored = "_".join(region_name.split())
                        with open(final_scores_txt_file, "a") as f:
                            f.write(f"region_{region_name_underscored}_{metric}: {score:.5f}\n")
            else:
                for metric, score in language_model_scores[subset].items():
                    if metric == "CE":
                        ce_score_dict = language_model_scores["report"]["CE"]
                        write_clinical_efficacy_scores(ce_score_dict)
                    else:
                        with open(final_scores_txt_file, "a") as f:
                            f.write(f"{subset}_{metric}: {score:.5f}\n")

    with open(final_scores_txt_file, "a") as f:
        f.write(f"Run: {RUN}\n")
        f.write(f"Checkpoint: {CHECKPOINT}\n")
        f.write(f"BertScore for removing similar generated sentences: {BERTSCORE_SIMILARITY_THRESHOLD}\n")
        f.write(f"Num beams: {NUM_BEAMS}\n")

    write_obj_detector_scores()
    write_region_selection_scores()
    write_region_abnormal_scores()
    write_language_model_scores()


def write_sentences_and_reports_to_file_for_test_set(
    gen_and_ref_sentences,
    gen_and_ref_reports,
    gen_sentences_with_corresponding_regions
):
    def write_sentences():
        txt_file_name = os.path.join(path_test_set_evaluation_scores_txt_files, "generated_sentences.txt")
        txt_file_name_abnormal = os.path.join(path_test_set_evaluation_scores_txt_files, "generated_abnormal_sentences.txt")

        with open(txt_file_name, "a") as f:
            for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")

        with open(txt_file_name_abnormal, "a") as f:
            for gen_sent, ref_sent in zip(generated_sentences_abnormal_regions, reference_sentences_abnormal_regions):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")

    def write_reports():
        txt_file_name = os.path.join(path_test_set_evaluation_scores_txt_files, "generated_reports.txt")

        with open(txt_file_name, "a") as f:
            for gen_report, ref_report, removed_similar_gen_sents, gen_sents_with_regions_single_report in zip(
                generated_reports,
                reference_reports,
                removed_similar_generated_sentences,
                gen_sentences_with_corresponding_regions
            ):
                f.write(f"Generated report: {gen_report}\n\n")
                f.write(f"Reference report: {ref_report}\n\n")

                f.write("Generated sentences with their regions:\n")
                for region_name, gen_sent in gen_sents_with_regions_single_report:
                    f.write(f"\t{region_name}: {gen_sent}\n")
                f.write("\n")

                f.write("Generated sentences that were removed:\n")
                for gen_sent, list_similar_gen_sents in removed_similar_gen_sents.items():
                    f.write(f"\t{gen_sent} == {list_similar_gen_sents}\n")
                f.write("\n")

                f.write("=" * 30)
                f.write("\n\n")

    num_generated_sentences_to_save = NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE * BATCH_SIZE
    num_generated_reports_to_save = NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE

    # all below are list of str
    generated_sentences = gen_and_ref_sentences["generated_sentences"][:num_generated_sentences_to_save]
    generated_sentences_abnormal_regions = gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"][:num_generated_sentences_to_save]
    reference_sentences = gen_and_ref_sentences["reference_sentences"][:num_generated_sentences_to_save]
    reference_sentences_abnormal_regions = gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"][:num_generated_sentences_to_save]

    write_sentences()

    # all below are list of str except removed_similar_generated_sentences which is a list of dict
    generated_reports = gen_and_ref_reports["generated_reports"][:num_generated_reports_to_save]
    reference_reports = gen_and_ref_reports["reference_reports"][:num_generated_reports_to_save]
    removed_similar_generated_sentences = gen_and_ref_reports["removed_similar_generated_sentences"][:num_generated_reports_to_save]

    write_reports()


def evaluate_language_model_on_test_set(model, test_loader, test_2_loader, tokenizer):
    def iterate_over_test_loader(test_loader):
        # to recover from potential OOMs
        oom = False

        # used in function get_generated_reports
        sentence_tokenizer = spacy.load("en_core_web_trf")

        with torch.no_grad():
            for num_batch, batch in tqdm(enumerate(test_loader)):

                images = batch["images"]  # shape [batch_size x 1 x 512 x 512]
                region_is_abnormal = batch["region_is_abnormal"].numpy()  # boolean array of shape [batch_size x 29]

                # List[List[str]] that holds the reference phrases. The inner list holds all reference phrases of a single image
                reference_sentences = batch["reference_sentences"]

                # List[str] that holds the reference report for the images in the batch
                reference_reports = batch["reference_reports"]

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

                        with open(final_scores_txt_file, "a") as f:
                            f.write("Generation:\n")
                            f.write(f"OOM at batch number {num_batch}.\n")
                            f.write(f"Error message: {str(e)}\n\n")
                    else:
                        raise e

                if oom:
                    # free up memory
                    torch.cuda.empty_cache()
                    oom = False
                    continue

                # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
                if output == -1:
                    with open(final_scores_txt_file, "a") as f:
                        f.write("Generation:\n")
                        f.write(f"Empty region features before language model at batch number {num_batch}.\n\n")

                    continue
                else:
                    # selected_regions is of shape [batch_size x 29] and is True for regions that should get a sentence
                    beam_search_output, selected_regions, _, _ = output
                    selected_regions = selected_regions.detach().cpu().numpy()

                # generated_sentences_for_selected_regions is a List[str] of length "num_regions_selected_in_batch"
                generated_sents_for_selected_regions = tokenizer.batch_decode(
                    beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

                # filter reference_sentences to those that correspond to the generated_sentences for the selected regions.
                # reference_sentences_for_selected_regions will therefore be a List[str] of length "num_regions_selected_in_batch"
                # (i.e. same length as generated_sentences_for_selected_regions)
                reference_sents_for_selected_regions = get_ref_sentences_for_selected_regions(
                    reference_sentences, selected_regions
                )

                (
                    gen_sents_for_normal_selected_regions,
                    gen_sents_for_abnormal_selected_regions,
                    ref_sents_for_normal_selected_regions,
                    ref_sents_for_abnormal_selected_regions,
                ) = get_sents_for_normal_abnormal_selected_regions(region_is_abnormal, selected_regions, generated_sents_for_selected_regions, reference_sents_for_selected_regions)

                generated_reports, removed_similar_generated_sentences = get_generated_reports(
                    generated_sents_for_selected_regions,
                    selected_regions,
                    sentence_tokenizer,
                    BERTSCORE_SIMILARITY_THRESHOLD
                )

                gen_and_ref_sentences["generated_sentences"].extend(generated_sents_for_selected_regions)
                gen_and_ref_sentences["generated_sentences_normal_selected_regions"].extend(gen_sents_for_normal_selected_regions)
                gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"].extend(gen_sents_for_abnormal_selected_regions)
                gen_and_ref_sentences["reference_sentences"].extend(reference_sents_for_selected_regions)
                gen_and_ref_sentences["reference_sentences_normal_selected_regions"].extend(ref_sents_for_normal_selected_regions)
                gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"].extend(ref_sents_for_abnormal_selected_regions)
                gen_and_ref_reports["generated_reports"].extend(generated_reports)
                gen_and_ref_reports["reference_reports"].extend(reference_reports)
                gen_and_ref_reports["removed_similar_generated_sentences"].extend(removed_similar_generated_sentences)

                update_gen_and_ref_sentences_for_regions(gen_and_ref_sentences, generated_sents_for_selected_regions, reference_sents_for_selected_regions, selected_regions)
                update_num_generated_sentences_per_image(gen_and_ref_sentences, selected_regions)

                if num_batch < NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE:
                    update_gen_sentences_with_corresponding_regions(gen_sentences_with_corresponding_regions, generated_sents_for_selected_regions, selected_regions)

    # whilst iterating over the validation loader, save the (all, normal, abnormal) generated and reference sentences in the respective lists
    # the list under the key "num_generated_sentences_per_image" will hold integers that represent how many sentences were generated for each image
    # this is useful to be able to get all generated and reference sentences that correspond to the same image
    # (since we append all generated and reference sentences to the "generated_sentences" and "reference_sentences" lists indiscriminately, this information would be lost otherwise)
    gen_and_ref_sentences = {
        "generated_sentences": [],
        "generated_sentences_normal_selected_regions": [],
        "generated_sentences_abnormal_selected_regions": [],
        "reference_sentences": [],
        "reference_sentences_normal_selected_regions": [],
        "reference_sentences_abnormal_selected_regions": [],
        "num_generated_sentences_per_image": []
    }

    # also save the generated and reference sentences on a per region basis
    for region_index, _ in enumerate(ANATOMICAL_REGIONS):
        gen_and_ref_sentences[region_index] = {
            "generated_sentences": [],
            "reference_sentences": []
        }

    # and of course the generated and reference reports, and additionally keep track of the generated sentences
    # that were removed because they were too similar to other generated sentences (only as a sanity-check/for writing to file)
    gen_and_ref_reports = {
        "generated_reports": [],
        "removed_similar_generated_sentences": [],
        "reference_reports": [],
    }

    # gen_sentences_with_corresponding_regions will be a list[list[tuple[str, str]]],
    # where len(outer_list) will be NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE
    # the inner list will correspond to a single generated report / single image and hold tuples of (region_name, generated_sentence),
    # i.e. each region that was selected for that single image, with the corresponding generated sentence (without any removal of similar generated sentences)
    #
    # gen_sentences_with_corresponding_regions will be used such that each generated sentences in a generated report can be directly attributed to a region
    # because this information gets lost when we concatenated generated sentences
    # this is only used to get more insights into the generated reports that are written to file
    gen_sentences_with_corresponding_regions = []

    log.info("Test loader: generating sentences/reports...")
    iterate_over_test_loader(test_loader)
    log.info("Test loader: generating sentences/reports... DONE.")

    log.info("Test loader 2: generating sentences/reports...")
    iterate_over_test_loader(test_2_loader)
    log.info("Test loader 2: generating sentences/reports... DONE.")

    write_sentences_and_reports_to_file_for_test_set(
        gen_and_ref_sentences,
        gen_and_ref_reports,
        gen_sentences_with_corresponding_regions,
    )

    with open(final_scores_txt_file, "a") as f:
        f.write(f"Num generated reports: {len(gen_and_ref_reports['generated_reports'])}\n")

    log.info("Computing language_model_scores...")
    language_model_scores = compute_language_model_scores(gen_and_ref_sentences, gen_and_ref_reports)
    log.info("Computing language_model_scores... DONE.")

    return language_model_scores


def update_object_detector_metrics_test_loader_2(obj_detector_scores, detections, image_targets, class_detected):
    def compute_box_area(box):
        """
        Calculate the area of a box given the 4 corner values.

        Args:
            box (Tensor[batch_size x 29 x 4])

        Returns:
            area (Tensor[batch_size x 29])
        """
        x0 = box[..., 0]
        y0 = box[..., 1]
        x1 = box[..., 2]
        y1 = box[..., 3]

        return (x1 - x0) * (y1 - y0)

    def get_gt_boxes_missing(targets):
        gt_boxes_missing = []

        # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image,
        # and key "labels" that contains the labels
        # the labels is a list[int] that should go from 1 to 29 (with 0 being the background class), but in test_loader_2 there will be some missing numbers
        for t in targets:
            labels_single_image = t["labels"]
            gt_boxes_missing_single_image = []
            for num in range(1, 30):
                if num not in labels_single_image:
                    gt_boxes_missing_single_image.append(True)
                else:
                    gt_boxes_missing_single_image.append(False)

            gt_boxes_missing.append(gt_boxes_missing_single_image)

        gt_boxes_missing = torch.tensor(gt_boxes_missing, device=device)

        return gt_boxes_missing

    def get_gt_boxes(targets, gt_boxes_missing):
        gt_boxes = []
        for t, gt_boxes_missing_single_image in zip(targets, gt_boxes_missing):
            curr_index_boxes_single_image = 0
            boxes_single_image = t["boxes"]
            gt_boxes_single_image = []

            for gt_boxes_missing_bool in gt_boxes_missing_single_image:
                if gt_boxes_missing_bool:
                    gt_boxes_single_image.append(torch.tensor([0, 0, 0, 0], dtype=boxes_single_image.dtype, device=device))
                else:
                    gt_boxes_single_image.append(boxes_single_image[curr_index_boxes_single_image])
                    curr_index_boxes_single_image += 1

            gt_boxes.append(torch.stack(gt_boxes_single_image))

        gt_boxes = torch.stack(gt_boxes)
        return gt_boxes

    def compute_intersection_and_union_area_per_region(detections, targets, class_detected):
        # pred_boxes is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
        # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
        # the 2nd to the 2nd class and so on
        pred_boxes = detections["top_region_boxes"]

        # since we evaluate for test_loader_2, all the images tend to have different numbers of gt bbox coordinates and labels
        # i.e. a given image has 20 gt bbox coordinates and labels, then another 23, another 18 and so on
        # create a mask of shape [batch_size x 29] that is True if a gt bbox is missing
        gt_boxes_missing = get_gt_boxes_missing(targets)

        # create the ground-truth boxes of shape [batch_size x 29 x 4]
        # replace missing ground-truth boxes by [0, 0, 0, 0]
        # since the intersection and union areas corresponding to these boxes will be set to 0 later with the gt_boxes_missing mask,
        # it does not really matter what values the missing gt boxes have (can actually be chosen arbitrarily)
        gt_boxes = get_gt_boxes(targets, gt_boxes_missing)

        # below tensors are of shape [batch_size x 29]
        x0_max = torch.maximum(pred_boxes[..., 0], gt_boxes[..., 0])
        y0_max = torch.maximum(pred_boxes[..., 1], gt_boxes[..., 1])
        x1_min = torch.minimum(pred_boxes[..., 2], gt_boxes[..., 2])
        y1_min = torch.minimum(pred_boxes[..., 3], gt_boxes[..., 3])

        # intersection_boxes is of shape [batch_size x 29 x 4]
        intersection_boxes = torch.stack([x0_max, y0_max, x1_min, y1_min], dim=-1)

        # below tensors are of shape [batch_size x 29]
        intersection_area = compute_box_area(intersection_boxes)
        pred_area = compute_box_area(pred_boxes)
        gt_area = compute_box_area(gt_boxes)

        # if x0_max >= x1_min or y0_max >= y1_min, then there is no intersection
        valid_intersection = torch.logical_and(x0_max < x1_min, y0_max < y1_min)

        # also there is no intersection if the class was not detected by object detector
        valid_intersection = torch.logical_and(valid_intersection, class_detected)

        # set all non-valid intersection areas to 0
        intersection_area[~valid_intersection] = 0

        union_area = (pred_area + gt_area) - intersection_area

        # set intersection_area and union_area to 0 if gt_boxes_missing is True for them
        intersection_area[gt_boxes_missing] = 0
        union_area[gt_boxes_missing] = 0

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


def evaluate_obj_detector_and_binary_classifiers_on_test_set(model, test_loader, test_2_loader):
    def iterate_over_test_loader(test_loader, num_images, is_test_2_loader):
        """
        We have to distinguish between test_loader and test_2_loader,
        since test_2_loader has less than 29 bbox_coordinates and bbox_labels
        (as opposed to test_loader, which always has 29).

        This is only a problem for the function update_object_detector_metrics,
        since it's written in a vectorized form that always expect 29 elements.

        So when is_test_2_loader=True, we use update_object_detector_metrics_test_loader_2,
        which works with less than 29 elements.
        """
        # to potentially recover if anything goes wrong
        oom = False

        with torch.no_grad():
            for num_batch, batch in tqdm(enumerate(test_loader)):
                images = batch["images"]
                image_targets = batch["image_targets"]
                region_has_sentence = batch["region_has_sentence"]
                region_is_abnormal = batch["region_is_abnormal"]
                input_ids = None  # not needed, since here we evaluate everything except language model
                attention_mask = None  # not needed, since here we evaluate everything except language model

                batch_size = images.size(0)
                num_images += batch_size

                images = images.to(device, non_blocking=True)
                image_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in image_targets]
                region_has_sentence = region_has_sentence.to(device, non_blocking=True)
                region_is_abnormal = region_is_abnormal.to(device, non_blocking=True)

                try:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(
                            images, image_targets, input_ids, attention_mask, region_has_sentence, region_is_abnormal
                        )
                except RuntimeError as e:  # out of memory error
                    if "out of memory" in str(e):
                        oom = True

                        with open(final_scores_txt_file, "a") as f:
                            f.write(f"OOM at batch number {num_batch}.\n")
                            f.write(f"Error message: {str(e)}\n\n")
                    else:
                        raise e

                if oom:
                    # free up memory
                    torch.cuda.empty_cache()
                    oom = False

                    num_images -= batch_size

                    continue

                (
                    _,
                    _,
                    _,
                    detections,
                    class_detected,  # bool tensor of shape [batch_size x 29]
                    selected_regions,  # bool tensor of shape [batch_size x 29]
                    predicted_abnormal_regions,  # bool tensor of shape [batch_size x 29]
                ) = output

                # update scores for object detector metrics
                if is_test_2_loader:
                    update_object_detector_metrics_test_loader_2(obj_detector_scores, detections, image_targets, class_detected)
                else:
                    update_object_detector_metrics(obj_detector_scores, detections, image_targets, class_detected)

                # update scores for region selection metrics
                update_region_selection_metrics(
                    region_selection_scores, selected_regions, region_has_sentence, region_is_abnormal
                )

                # update scores for region abnormal detection metrics
                update_region_abnormal_metrics(
                    region_abnormal_scores, predicted_abnormal_regions, region_is_abnormal, class_detected
                )

        return num_images

    """
    For the object detector, we want to compute:
      - average overall IoU
      - average IoU per region,
      - average number of detected regions per image (ideally 29.0)
      - average number each region is detected in an image (ideally 1.0 for all regions)

    To compute these metrics, we allocate several tensors:

    sum_intersection_area_per_region: for accumulating the intersection area of each region
    (will be divided by union area of each region at the end of get the IoU for each region)

    sum_union_area_per_region: for accumulating the union area of each region
    (will divide the intersection area of each region at the end of get the IoU for each region)

    sum_region_detected: for accumulating the number of times a region is detected over all images
    (this 1D array will be divided by num_images to get the average number each region is detected in an image,
    and these averages will be summed up to get the average number of detected regions in an image)
    """
    obj_detector_scores = {}
    obj_detector_scores["sum_intersection_area_per_region"] = torch.zeros(29, device=device)
    obj_detector_scores["sum_union_area_per_region"] = torch.zeros(29, device=device)
    obj_detector_scores["sum_region_detected"] = torch.zeros(29, device=device)

    """
    For the binary classifier for region selection, we want to compute the precision, recall and f1 for:
      - all regions
      - normal regions
      - abnormal regions

    Evaluation according to:
      TP: (normal/abnormal) region has sentence (gt), and is selected by classifier to get sentence (pred)
      FP: (normal/abnormal) region does not have sentence (gt), but is selected by classifier to get sentence (pred)
      TN: (normal/abnormal) region does not have sentence (gt), and is not selected by classifier to get sentence (pred)
      FN: (normal/abnormal) region has sentence (gt), but is not selected by classifier to get sentence (pred)
    """
    region_selection_scores = {}
    for subset in ["all", "normal", "abnormal"]:
        region_selection_scores[subset] = {
            "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
            "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
            "f1": torchmetrics.F1Score(num_classes=2, average=None).to(device),
        }

    """
    For the binary classifier for region normal/abnormal detection, we want to compute the precision, recall and f1 for:
      - all regions

    Evaluation according to:
      TP: region is abnormal (gt), and is predicted as abnormal by classifier (pred)
      FP: region is normal (gt), but is predicted as abnormal by classifier (pred)
      TN: region is normal (gt), and is predicted as normal by classifier (pred)
      FN: region is abnormal (gt), but is predicted as normal by classifier (pred)
    """
    region_abnormal_scores = {
        "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
        "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
        "f1": torchmetrics.F1Score(num_classes=2, average=None).to(device),
    }

    num_images = 0

    log.info("Test loader: computing scores for object detector, region selection and region abnormal module...")
    num_images = iterate_over_test_loader(test_loader, num_images, is_test_2_loader=False)
    log.info("Test loader: computing scores for object detector, region selection and region abnormal module... DONE.")

    log.info("Test loader 2: computing scores for object detector, region selection and region abnormal module...")
    num_images = iterate_over_test_loader(test_2_loader, num_images, is_test_2_loader=True)
    log.info("Test loader 2: computing scores for object detector, region selection and region abnormal module... DONE.")

    # compute object detector scores
    sum_intersection = obj_detector_scores["sum_intersection_area_per_region"]
    sum_union = obj_detector_scores["sum_union_area_per_region"]
    obj_detector_scores["avg_iou"] = (torch.sum(sum_intersection) / torch.sum(sum_union)).item()
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


def evaluate_model_on_test_set(model, test_loader, test_2_loader, tokenizer):
    obj_detector_scores, region_selection_scores, region_abnormal_scores = evaluate_obj_detector_and_binary_classifiers_on_test_set(model, test_loader, test_2_loader)

    language_model_scores = evaluate_language_model_on_test_set(model, test_loader, test_2_loader, tokenizer)

    write_all_scores_to_file(
        obj_detector_scores,
        region_selection_scores,
        region_abnormal_scores,
        language_model_scores
    )


def get_model():
    checkpoint = torch.load(
        os.path.join(path_runs_full_model, f"run_{RUN}", "checkpoints", f"{CHECKPOINT}"),
        map_location=torch.device("cpu"),
    )

    # if there is a key error when loading checkpoint, try uncommenting down below
    # since depending on the torch version, the state dicts may be different
    # checkpoint["model"]["object_detector.rpn.head.conv.weight"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.weight")
    # checkpoint["model"]["object_detector.rpn.head.conv.bias"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.bias")

    # pretrain_without_lm_model=True, since we don't need to compute the language model loss (see forward method of full model)
    # we evaluate the language model in function evaluate_language_model_on_test_set by generating sentences/reports based on input images
    model = ReportGenerationModel(pretrain_without_lm_model=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device, non_blocking=True)
    model.eval()

    del checkpoint

    return model


def get_data_loaders(tokenizer, test_dataset_complete, test_2_dataset_complete):
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

    test_2_loader = DataLoader(
        test_2_dataset_complete,
        collate_fn=custom_collate_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return test_loader, test_2_loader


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


def get_tokenized_dataset(tokenizer, raw_test_dataset, raw_test_2_dataset):
    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_test_dataset = raw_test_dataset.map(tokenize_function)
    tokenized_test_2_dataset = raw_test_2_dataset.map(tokenize_function)

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path (str)
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])
    #   - reference_report (str)

    return tokenized_test_dataset, tokenized_test_2_dataset


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

    datasets_as_dfs = {}
    datasets_as_dfs["test"] = pd.read_csv(os.path.join(path_full_dataset, "test.csv"), usecols=usecols, converters=converters)
    datasets_as_dfs["test-2"] = pd.read_csv(os.path.join(path_full_dataset, "test-2.csv"), usecols=usecols, converters=converters)

    raw_test_dataset = Dataset.from_pandas(datasets_as_dfs["test"])
    raw_test_2_dataset = Dataset.from_pandas(datasets_as_dfs["test-2"])

    return raw_test_dataset, raw_test_2_dataset


def main():
    raw_test_dataset, raw_test_2_dataset = get_dataset()

    # note that we don't actually need to tokenize anything (i.e. we don't need the input ids and attention mask),
    # because we evaluate the language model on it's generation capabilities (for which we only need the input images)
    # but since the custom dataset and collator are build in a way that they expect input ids and attention mask
    # (as they were originally made for training the model),
    # it's better to just leave it as it is instead of adding unnecessary complexity
    tokenizer = get_tokenizer()
    tokenized_test_dataset, tokenized_test_2_dataset = get_tokenized_dataset(tokenizer, raw_test_dataset, raw_test_2_dataset)

    test_transforms = get_transforms()

    test_dataset_complete = CustomDataset("test", tokenized_test_dataset, test_transforms, log)
    test_2_dataset_complete = CustomDataset("test", tokenized_test_2_dataset, test_transforms, log)

    test_loader, test_2_loader = get_data_loaders(tokenizer, test_dataset_complete, test_2_dataset_complete)

    model = get_model()

    evaluate_model_on_test_set(model, test_loader, test_2_loader, tokenizer)


if __name__ == "__main__":
    main()
