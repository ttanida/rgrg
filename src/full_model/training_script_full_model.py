from ast import literal_eval
from copy import deepcopy
import logging
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchmetrics
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.dataset_bounding_boxes.constants import ANATOMICAL_REGIONS
from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset
from src.full_model.report_generation_model import ReportGenerationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed value for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# define configurations for training run
RUN = 0
# can be useful to add additional information to run_config.txt file
RUN_COMMENT = """Train full model on small dataset"""
IMAGE_INPUT_SIZE = 512
PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.00005
PERCENTAGE_OF_VAL_SET_TO_USE = 0.004
BATCH_SIZE = 4
NUM_WORKERS = 12
EPOCHS = 100
LR = 1e-3
EVALUATE_EVERY_K_STEPS = 500  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE_LR_SCHEDULER = 20  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1
NUM_BEAMS = 4
MAX_NUM_TOKENS_GENERATE = 300
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 2  # save num_batches_of_... worth of generated sentences with their gt reference phrases to a txt file
NUM_SENTENCES_TO_GENERATE = 10


def get_title(region_set, region_indices, region_colors, class_detected_img):
    # region_set always contains 6 region names

    # get a list of 6 boolean values that specify if that region was detected
    class_detected = [class_detected_img[region_index] for region_index in region_indices]

    # add color_code to region name (e.g. "(r)" for red)
    # also add nd to the brackets if region was not detected (e.g. "(r, nd)" if red region was not detected)
    region_set = [region + f" ({color})" if cls_detect else region + f" ({color}, nd)" for region, color, cls_detect in zip(region_set, region_colors, class_detected)]

    # add a line break to the title, as to not make it too long
    return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])


def get_generated_sentence_for_region(generated_sentences_for_selected_regions, selected_regions, num_img, region_index):
    index = 0
    for num in range(num_img):
        index += torch.sum(selected_regions[num, :]).item()

    index += torch.sum(selected_regions[num_img, :region_index]).item()

    return generated_sentences_for_selected_regions[index]


def transform_sentence_to_fit_under_image(ref_sent_region):
    max_line_length = 50
    if len(ref_sent_region) < max_line_length:
        return ref_sent_region

    words = ref_sent_region.split()
    transformed_sent = ""
    current_line_length = 0
    prefix_for_alignment = "\n" + " " * 20
    for word in words:
        if len(word) + current_line_length > max_line_length:
            word = f"{prefix_for_alignment}{word}"
            current_line_length = -len(prefix_for_alignment)

        current_line_length += len(word)
        transformed_sent += word + " "

    return transformed_sent


def update_region_set_text(region_set_text, color, reference_sentences_img, generated_sentences_for_selected_regions, region_index, selected_regions, num_img):
    region_set_text += f"({color}):  \n"
    reference_sentence_region = reference_sentences_img[region_index]
    reference_sentence_region = transform_sentence_to_fit_under_image(reference_sentence_region)
    region_set_text += f"  reference: {reference_sentence_region}\n"

    box_region_selected = selected_regions[num_img][region_index]
    if not box_region_selected:
        region_set_text += "  generated: [REGION NOT SELECTED]\n\n"
    else:
        generated_sentence_region = get_generated_sentence_for_region(generated_sentences_for_selected_regions, selected_regions, num_img, region_index)
        generated_sentence_region = transform_sentence_to_fit_under_image(generated_sentence_region)
        region_set_text += f"  generated: {generated_sentence_region}\n\n"

    return region_set_text


def plot_box(box, ax, clr, linestyle, region_detected=True):
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    ax.add_artist(plt.Rectangle(xy=(x0, y0), height=h, width=w, fill=False, color=clr, linewidth=1, linestyle=linestyle))

    # add an annotation to the gt box, that the pred box does not exist (i.e. the corresponding region was not detected)
    if not region_detected:
        ax.annotate("not detected", (x0, y0), color=clr, weight="bold", fontsize=10)


def plot_detections_and_sentences_to_tensorboard(
    writer,
    overall_steps_taken,
    images,
    image_targets,
    selected_regions,
    detections,
    class_detected,
    reference_sentences,
    generated_sentences_for_selected_regions,
    num_images_to_plot=3,
):
    # pred_boxes_batch is of shape [batch_size x 36 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
    # they are sorted in the 2nd dimension, meaning the 1st of the 36 boxes corresponds to the 1st region/class,
    # the 2nd to the 2nd class and so on
    pred_boxes_batch = detections["top_region_boxes"]

    # image_targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
    # gt_boxes is of shape [batch_size x 36 x 4]
    gt_boxes_batch = torch.stack([t["boxes"] for t in image_targets], dim=0)

    # plot 6 regions at a time, as to not overload the image with boxes
    # the region_sets were chosen as to minimize overlap between the contained regions (i.e. better visibility)
    region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
    region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
    region_set_3 = ["right hilar structures", "right apical zone", "right cardiophrenic angle", "left hilar structures", "left apical zone", "left cardiophrenic angle"]
    region_set_4 = ["right hemidiaphragm", "left hemidiaphragm", "trachea", "right clavicle", "left clavicle", "aortic arch"]
    region_set_5 = ["mediastinum", "left upper abdomen", "right upper abdomen", "svc", "cavoatrial junction", "carina"]
    region_set_6 = ["right atrium", "descending aorta", "left cardiac silhouette", "upper mediastinum", "right cardiac silhouette", "abdomen"]

    regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5, region_set_6]
    region_colors = ["b", "g", "r", "c", "m", "y"]

    for num_img in range(num_images_to_plot):
        image = images[num_img].numpy().transpose(1, 2, 0)

        gt_boxes_img = gt_boxes_batch[num_img]
        pred_boxes_img = pred_boxes_batch[num_img]
        class_detected_img = class_detected[num_img].tolist()
        selected_regions = selected_regions.detach().cpu()
        reference_sentences_img = reference_sentences[num_img]

        for num_region_set, region_set in enumerate(regions_sets):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()

            plt.imshow(image, cmap="gray")
            plt.axis("off")

            region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]

            region_set_text = ""

            for region_index, color in zip(region_indices, region_colors):
                box_gt = gt_boxes_img[region_index].tolist()
                box_pred = pred_boxes_img[region_index].tolist()
                box_region_detected = class_detected_img[region_index]

                plot_box(box_gt, ax, clr=color, linestyle="solid", region_detected=box_region_detected)

                # only plot predicted box if class was actually detected
                if box_region_detected:
                    plot_box(box_pred, ax, clr=color, linestyle="dashed")

                region_set_text = update_region_set_text(
                    region_set_text, color, reference_sentences_img, generated_sentences_for_selected_regions, region_index, selected_regions, num_img
                )

            title = get_title(region_set, region_indices, region_colors, class_detected_img)
            ax.set_title(title)

            plt.xlabel(region_set_text, loc="left")

            writer.add_figure(f"img_{num_img}_region_set_{num_region_set}", fig, overall_steps_taken)


def write_all_losses_and_scores_to_tensorboard(
    writer,
    overall_steps_taken,
    train_losses_dict,
    val_losses_dict,
    obj_detector_scores,
    region_selection_scores,
    region_abnormal_scores,
    language_model_scores,
    current_lr,
):
    def write_losses(writer, overall_steps_taken, train_losses_dict, val_losses_dict):
        for loss_type in train_losses_dict:
            writer.add_scalars(
                "_loss",
                {f"{loss_type}_train": train_losses_dict[loss_type], f"{loss_type}_val": val_losses_dict[loss_type]},
                overall_steps_taken,
            )

    def write_obj_detector_scores(writer, overall_steps_taken, obj_detector_scores):
        writer.add_scalar(
            "avg_num_detected_regions_per_image",
            obj_detector_scores["avg_num_detected_regions_per_image"],
            overall_steps_taken,
        )

        # replace white space by underscore for each region name (i.e. "right upper lung" -> "right_upper_lung")
        anatomical_regions = ["_".join(region.split()) for region in ANATOMICAL_REGIONS]
        avg_detections_per_region = obj_detector_scores["avg_detections_per_region"]
        avg_iou_per_region = obj_detector_scores["avg_iou_per_region"]

        for region_, avg_detections_region in zip(anatomical_regions, avg_detections_per_region):
            writer.add_scalar(f"num_detected_{region_}", avg_detections_region, overall_steps_taken)

        for region_, avg_iou_region in zip(anatomical_regions, avg_iou_per_region):
            writer.add_scalar(f"iou_{region_}", avg_iou_region, overall_steps_taken)

    def write_region_selection_scores(writer, overall_steps_taken, region_selection_scores):
        for subset in region_selection_scores:
            for metric, score in region_selection_scores[subset].items():
                writer.add_scalar(f"region_select_{subset}_{metric}", score, overall_steps_taken)

    def write_region_abnormal_scores(writer, overall_steps_taken, region_abnormal_scores):
        for metric, score in region_abnormal_scores.items():
            writer.add_scalar(f"region_abnormal_{metric}", score, overall_steps_taken)

    def write_language_model_scores(writer, overall_steps_taken, language_model_scores):
        for subset in language_model_scores:
            for metric, score in language_model_scores[subset].items():
                writer.add_scalar(f"language_model_{subset}_{metric}", score, overall_steps_taken)

    write_losses(writer, overall_steps_taken, train_losses_dict, val_losses_dict)
    write_obj_detector_scores(writer, overall_steps_taken, obj_detector_scores)
    write_region_selection_scores(writer, overall_steps_taken, region_selection_scores)
    write_region_abnormal_scores(writer, overall_steps_taken, region_abnormal_scores)
    write_language_model_scores(writer, overall_steps_taken, language_model_scores)

    writer.add_scalar("lr", current_lr, overall_steps_taken)


def compute_final_language_model_scores(language_model_scores):
    for subset in language_model_scores:
        temp = {}
        for metric, score in language_model_scores[subset].items():
            if metric.startswith("bleu"):
                result = score.compute(max_order=int(metric[-1]))
                temp[f"{metric}"] = result["bleu"]
            else:  # bert_score
                result = score.compute(lang="en", device=device)
                avg_precision = np.array(result["precision"]).mean()
                avg_recall = np.array(result["recall"]).mean()
                avg_f1 = np.array(result["f1"]).mean()

                temp["bertscore_precision"] = avg_precision
                temp["bertscore_recall"] = avg_recall
                temp["bertscore_f1"] = avg_f1

        language_model_scores[subset] = temp


def write_sentences_to_file(gen_and_ref_sentences_to_save_to_file, generated_sentences_folder_path, overall_steps_taken):
    generated_sentences_txt_file = os.path.join(generated_sentences_folder_path, f"generated_sentences_step_{overall_steps_taken}")

    # generated_sentences is a list of str
    generated_sentences = gen_and_ref_sentences_to_save_to_file["generated_sentences"]

    # reference_sentences is a list of str
    reference_sentences = gen_and_ref_sentences_to_save_to_file["reference_sentences"]

    with open(generated_sentences_txt_file, "w") as f:
        for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
            f.write(f"Generated sentence: {gen_sent}\n")
            f.write(f"Reference sentence: {ref_sent}\n\n")


def get_sents_for_normal_abnormal_selected_regions(generated_sentences_for_selected_regions, reference_sentences_for_selected_regions, selected_regions, region_is_abnormal):
    # selected_region_is_abnormal is a bool array of shape [num_regions_selected_in_batch] that specifies if a selected region is abnormal (True) or normal (False)
    selected_region_is_abnormal = region_is_abnormal[selected_regions]
    selected_region_is_abnormal = selected_region_is_abnormal.detach().cpu().numpy()

    generated_sentences_for_selected_regions = np.asarray(generated_sentences_for_selected_regions)
    reference_sentences_for_selected_regions = np.asarray(reference_sentences_for_selected_regions)

    gen_sents_for_normal_selected_regions = generated_sentences_for_selected_regions[~selected_region_is_abnormal].tolist()
    gen_sents_for_abnormal_selected_regions = generated_sentences_for_selected_regions[selected_region_is_abnormal].tolist()

    ref_sents_for_normal_selected_regions = reference_sentences_for_selected_regions[~selected_region_is_abnormal].tolist()
    ref_sents_for_abnormal_selected_regions = reference_sentences_for_selected_regions[selected_region_is_abnormal].tolist()

    return (
        gen_sents_for_normal_selected_regions,
        gen_sents_for_abnormal_selected_regions,
        ref_sents_for_normal_selected_regions,
        ref_sents_for_abnormal_selected_regions,
    )


def update_language_model_scores(language_model_scores, generated_sentences_for_selected_regions, reference_sentences_for_selected_regions, selected_regions, region_is_abnormal):
    for score in language_model_scores["all"].values():
        score.add_batch(predictions=generated_sentences_for_selected_regions, references=reference_sentences_for_selected_regions)

    # for computing the scores for the normal and abnormal reference sentences, we have to filter the generated and reference sentences accordingly
    (
        gen_sents_for_normal_selected_regions,
        gen_sents_for_abnormal_selected_regions,
        ref_sents_for_normal_selected_regions,
        ref_sents_for_abnormal_selected_regions,
    ) = get_sents_for_normal_abnormal_selected_regions(generated_sentences_for_selected_regions, reference_sentences_for_selected_regions, selected_regions, region_is_abnormal)

    if len(ref_sents_for_normal_selected_regions) != 0:
        for score in language_model_scores["normal"].values():
            score.add_batch(predictions=gen_sents_for_normal_selected_regions, references=ref_sents_for_normal_selected_regions)

    if len(ref_sents_for_abnormal_selected_regions) != 0:
        for score in language_model_scores["abnormal"].values():
            score.add_batch(predictions=gen_sents_for_abnormal_selected_regions, references=ref_sents_for_abnormal_selected_regions)


def get_ref_sentences_for_selected_regions(reference_sentences, selected_regions):
    """
    Args:
        reference_sentences (List[List[str]]): outer list has len batch_size, inner list has len 36 (the inner list holds all reference phrases of a single image)
        selected_regions ([batch_size x 36]): boolean tensor that has exactly "num_regions_selected_in_batch" True values
    """
    # both arrays of shape [batch_size x 36]
    reference_sentences = np.asarray(reference_sentences)
    selected_regions = selected_regions.detach().cpu().numpy()

    ref_sentences_for_selected_regions = reference_sentences[selected_regions]

    return ref_sentences_for_selected_regions.tolist()


def evaluate_language_model(model, val_dl, tokenizer, writer, overall_steps_taken, generated_sentences_folder_path):
    # compute scores for all, normal and abnormal reference sentences
    subsets = ["all", "normal", "abnormal"]
    language_model_scores = {}

    for subset in subsets:
        language_model_scores[subset] = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}
        language_model_scores[subset]["bert_score"] = evaluate.load("bertscore")

    gen_and_ref_sentences_to_save_to_file = {"generated_sentences": [], "reference_sentences": []}

    # since generating sentences takes a long time (generating sentences for 36 regions takes around 8 seconds),
    # we only generate NUM_SENTENCES_TO_GENERATE sentences
    num_batches_to_process = NUM_SENTENCES_TO_GENERATE // BATCH_SIZE

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=num_batches_to_process):
            if num_batch >= num_batches_to_process:
                break

            images = batch["image"]  # shape [batch_size x 1 x 512 x 512]
            image_targets = batch["image_targets"]
            region_is_abnormal = batch["region_is_abnormal"]  # boolean tensor of shape [batch_size x 36]

            # List[List[str]] that holds the reference phrases. The inner list holds all reference phrases of a single image
            reference_sentences = batch["reference_sentences"]

            beam_search_output, selected_regions, detections, class_detected = model.generate(
                images.to(device, non_blocking=True), max_length=MAX_NUM_TOKENS_GENERATE, num_beams=NUM_BEAMS, early_stopping=True
            )

            # generated_sentences is a List[str] of length "num_regions_selected_in_batch"
            generated_sentences_for_selected_regions = tokenizer.batch_decode(beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # filter reference_sentences to those that correspond to the generated_sentences for the selected regions.
            # reference_sentences_for_selected_regions is a List[str] of length "num_regions_selected_in_batch"
            reference_sentences_for_selected_regions = get_ref_sentences_for_selected_regions(reference_sentences, selected_regions)

            if num_batch < NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE:
                gen_and_ref_sentences_to_save_to_file["generated_sentences"].extend(generated_sentences_for_selected_regions)
                gen_and_ref_sentences_to_save_to_file["reference_sentences"].extend(reference_sentences_for_selected_regions)

            update_language_model_scores(
                language_model_scores,
                generated_sentences_for_selected_regions,
                reference_sentences_for_selected_regions,
                selected_regions,
                region_is_abnormal,
            )

            if num_batch == 0:
                plot_detections_and_sentences_to_tensorboard(
                    writer,
                    overall_steps_taken,
                    images,
                    image_targets,
                    selected_regions,
                    detections,
                    class_detected,
                    reference_sentences,
                    generated_sentences_for_selected_regions,
                    num_images_to_plot=3,
                )

    write_sentences_to_file(gen_and_ref_sentences_to_save_to_file, generated_sentences_folder_path, overall_steps_taken)

    # compute final scores for language model metrics
    compute_final_language_model_scores(language_model_scores)

    return language_model_scores


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

    region_selection_scores["normal"]["precision"](normal_selected_regions, normal_region_has_sentence)
    region_selection_scores["normal"]["recall"](normal_selected_regions, normal_region_has_sentence)

    region_selection_scores["abnormal"]["precision"](abnormal_selected_regions, abnormal_region_has_sentence)
    region_selection_scores["abnormal"]["recall"](abnormal_selected_regions, abnormal_region_has_sentence)


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


def get_val_losses_and_other_metrics(model, val_dl):
    """
    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.

    Returns:
        val_losses_dict (Dict): holds different val losses of the different modules as well as the total val loss
        obj_detector_scores (Dict): holds scores of the average IoU per Region, average number of detected regions per image,
        average number each region is detected in an image
        region_selection_scores (Dict): holds precision and recall scores for all, normal and abnormal sentences
        region_abnormal_scores (Dict): holds precision and recall scores for all sentences
    """
    val_losses_dict = {
        "total_loss": 0.0,
        "obj_detector_loss": 0.0,
        "region_selection_loss": 0.0,
        "region_abnormal_loss": 0.0,
        "language_model_loss": 0.0,
    }

    num_images = 0

    """
    For the object detector, besides the obj_detector_val_loss, we also want to compute:
      - the average IoU for each region,
      - average number of detected regions per image (ideally 36.0)
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
    obj_detector_scores["sum_intersection_area_per_region"] = torch.zeros(36, device=device)
    obj_detector_scores["sum_union_area_per_region"] = torch.zeros(36, device=device)
    obj_detector_scores["sum_region_detected"] = torch.zeros(36, device=device)

    """
    For the binary classifier for region selection, we want to compute the precision and recall for:
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
    region_selection_scores["all"] = {
        "precision": torchmetrics.Precision(num_classes=2, average=None),
        "recall": torchmetrics.Recall(num_classes=2, average=None),
    }
    region_selection_scores["normal"] = {
        "precision": torchmetrics.Precision(num_classes=2, average=None),
        "recall": torchmetrics.Recall(num_classes=2, average=None),
    }
    region_selection_scores["abnormal"] = {
        "precision": torchmetrics.Precision(num_classes=2, average=None),
        "recall": torchmetrics.Recall(num_classes=2, average=None),
    }

    """
    For the binary classifier for region normal/abnormal detection, we want to compute the precision and recall for:
      - all regions

    Evaluation according to:
      TP: region is abnormal (gt), and is predicted as abnormal by classifier (pred)
      FP: region is normal (gt), but is predicted as abnormal by classifier (pred)
      TN: region is normal (gt), and is predicted as normal by classifier (pred)
      FN: region is abnormal (gt), but is predicted as normal by classifier (pred)
    """
    region_abnormal_scores = {
        "precision": torchmetrics.Precision(num_classes=2, average=None),
        "recall": torchmetrics.Recall(num_classes=2, average=None),
    }

    with torch.no_grad():
        for batch in tqdm(val_dl):
            # "image_targets" maps to a list of dicts, where each dict has the keys "boxes" and "labels" and corresponds to a single image
            # "boxes" maps to a tensor of shape [36 x 4] and "labels" maps to a tensor of shape [36]
            # note that the "labels" tensor is always sorted, i.e. it is of the form [1, 2, 3, ..., 36] (starting at 1, since 0 is background)
            images = batch["images"]
            image_targets = batch["image_targets"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            region_has_sentence = batch["region_has_sentence"]
            region_is_abnormal = batch["region_is_abnormal"]

            batch_size = images.size(0)
            num_images += batch_size

            # put all tensors on the GPU
            images = images.to(device, non_blocking=True)
            image_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in image_targets]
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            region_has_sentence = region_has_sentence.to(device, non_blocking=True)
            region_is_abnormal = region_is_abnormal.to(device, non_blocking=True)

            # detections is a dict with keys "top_region_boxes" and "top_scores"
            # "top_region_boxes" maps to a tensor of shape [batch_size x 36 x 4]
            # "top_scores" maps to a tensor of shape [batch_size x 36]
            #
            # class_detected is a tensor of shape [batch_size x 36]
            # selected_regions is a tensor of shape [batch_size x 36]
            # predicted_abnormal_regions is a tensor of shape [batch_size x 36]
            (
                obj_detector_loss_dict,
                classifier_loss_region_selection,
                classifier_loss_region_abnormal,
                language_model_loss,
                detections,
                class_detected,
                selected_regions,
                predicted_abnormal_regions,
            ) = model(log, images, image_targets, input_ids, attention_mask, region_has_sentence, region_is_abnormal)

            # sum up all 4 losses from the object detector
            obj_detector_losses = sum(loss for loss in obj_detector_loss_dict.values())

            # sum up the rest of the losses
            total_loss = obj_detector_losses + classifier_loss_region_selection + classifier_loss_region_abnormal + language_model_loss

            list_of_losses = [
                total_loss,
                obj_detector_losses,
                classifier_loss_region_selection,
                classifier_loss_region_abnormal,
                language_model_loss,
            ]

            # dicts are insertion ordered since Python 3.7
            for loss_type, loss in zip(val_losses_dict, list_of_losses):
                val_losses_dict[loss_type] += loss.item() * batch_size

            # update scores for object detector metrics
            update_object_detector_metrics(obj_detector_scores, detections, image_targets, class_detected)

            # update scores for region selection metrics
            update_region_selection_metrics(region_selection_scores, selected_regions, region_has_sentence, region_is_abnormal)

            # update scores for region abnormal detection metrics
            update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, region_is_abnormal, class_detected)

    # normalize the losses
    for loss_type in val_losses_dict:
        val_losses_dict[loss_type] /= len(val_dl)

    # average object detector scores
    sum_intersection = obj_detector_scores["sum_intersection_area_per_region"]
    sum_union = obj_detector_scores["sum_union_area_per_region"]
    obj_detector_scores["avg_iou_per_region"] = (sum_intersection / sum_union).tolist()

    sum_region_detected = obj_detector_scores["sum_region_detected"]
    obj_detector_scores["avg_num_detected_regions_per_image"] = torch.sum(sum_region_detected / num_images).item()
    obj_detector_scores["avg_detections_per_region"] = (sum_region_detected / num_images).tolist()

    # compute the "micro" average scores for region_selection_scores
    for subset in region_selection_scores:
        for metric, score in region_selection_scores[subset].values():
            region_selection_scores[subset][metric] = score.compute()[1].item()  # only report results for the positive class (hence [1])

    # compute the "micro" average scores for region_abnormal_scores
    for metric, score in region_abnormal_scores.values():
        region_abnormal_scores[metric] = score.compute()[1].item()

    return val_losses_dict, obj_detector_scores, region_selection_scores, region_abnormal_scores


def log_stats_to_console(
    train_loss,
    val_loss,
    epoch,
):
    log.info(f"Epoch: {epoch}:")
    log.info(f"\tTrain loss: {train_loss:.3f}")
    log.info(f"\tVal loss: {val_loss:.3f}")


def evaluate_model(model, train_losses_dict, val_dl, lr_scheduler, writer, tokenizer, run_params, is_epoch_end, generated_sentences_folder_path):
    # set the model to evaluation mode
    model.eval()

    overall_steps_taken = run_params["overall_steps_taken"]

    # normalize all train losses by steps_taken
    for loss_type in train_losses_dict:
        train_losses_dict[loss_type] /= run_params["steps_taken"]

    (
        val_losses_dict,
        obj_detector_scores,
        region_selection_scores,
        region_abnormal_scores,
    ) = get_val_losses_and_other_metrics(model, val_dl)

    language_model_scores = evaluate_language_model(model, val_dl, tokenizer, writer, overall_steps_taken, generated_sentences_folder_path)

    current_lr = lr_scheduler.get_last_lr()

    write_all_losses_and_scores_to_tensorboard(
        writer,
        overall_steps_taken,
        train_losses_dict,
        val_losses_dict,
        obj_detector_scores,
        region_selection_scores,
        region_abnormal_scores,
        language_model_scores,
        current_lr,
    )

    train_total_loss = train_losses_dict["total_loss"]
    total_val_loss = val_losses_dict["total_loss"]

    # decrease lr by 1e-1 if total_val_loss has not decreased after certain number of evaluations
    lr_scheduler.step(total_val_loss)

    if total_val_loss < run_params["lowest_val_loss"]:
        run_params["lowest_val_loss"] = total_val_loss
        run_params["best_epoch"] = run_params["epoch"]
        run_params["best_model_save_path"] = os.path.join(
            run_params["weights_folder_path"],
            f"val_loss_{run_params['lowest_val_loss']:.3f}_epoch_{run_params['best_epoch']}.pth",
        )
        run_params["best_model_state"] = deepcopy(model.state_dict())

    if is_epoch_end:
        log_stats_to_console(train_total_loss, total_val_loss, run_params["epoch"])


def train_model(model, train_dl, val_dl, optimizer, lr_scheduler, epochs, weights_folder_path, tokenizer, generated_sentences_folder_path, writer):
    """
    Train a model on train set and evaluate on validation set.
    Saves best model w.r.t. val loss.

    Parameters
    ----------
    model: nn.Module
        The input model to be trained.
    train_dl: torch.utils.data.Dataloder
        The train dataloader to train on.
    val_dl: torch.utils.data.Dataloder
        The val dataloader to validate on.
    optimizer: Optimizer
        The model's optimizer.
    lr_scheduler: torch.optim.lr_scheduler
        The learning rate scheduler to use.
    epochs: int
        Number of epochs to train for.
    weights_folder_path: str
        Path to folder where best weights will be saved.
    writer: torch.utils.tensorboard.SummaryWriter
        Writer for logging values to tensorboard.

    Returns
    -------
    None, but saves model with the overall lowest val loss at the end of every epoch.
    """
    run_params = {}
    run_params["epochs"] = epochs
    run_params["weights_folder_path"] = weights_folder_path
    run_params["lowest_val_loss"] = np.inf
    run_params["best_epoch"] = None  # the epoch with the lowest val loss overall
    run_params["best_model_state"] = None  # the best_model_state is the one where the val loss is the lowest overall
    run_params["best_model_save_path"] = None
    run_params["overall_steps_taken"] = 0  # for logging to tensorboard

    for epoch in range(epochs):
        run_params["epoch"] = epoch
        log.info(f"\nTraining epoch {epoch}!\n")

        train_losses_dict = {
            "total_loss": 0.0,
            "obj_detector_loss": 0.0,
            "region_selection_loss": 0.0,
            "region_abnormal_loss": 0.0,
            "language_model_loss": 0.0,
        }

        run_params["steps_taken"] = 0  # to know when to evaluate model during epoch and to normalize losses

        for num_batch, batch in tqdm(enumerate(train_dl)):
            images = batch["images"]
            image_targets = batch["image_targets"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            region_has_sentence = batch["region_has_sentence"]
            region_is_abnormal = batch["region_is_abnormal"]

            batch_size = images.size(0)

            # put all tensors on the GPU
            images = images.to(device, non_blocking=True)
            image_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in image_targets]
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            region_has_sentence = region_has_sentence.to(device, non_blocking=True)
            region_is_abnormal = region_is_abnormal.to(device, non_blocking=True)

            (
                obj_detector_loss_dict,
                classifier_loss_region_selection,
                classifier_loss_region_abnormal,
                language_model_loss,
            ) = model(epoch, images, image_targets, input_ids, attention_mask, region_has_sentence, region_is_abnormal)

            # sum up all 4 losses from the object detector
            obj_detector_losses = sum(loss for loss in obj_detector_loss_dict.values())

            # sum up the rest of the losses
            total_loss = obj_detector_losses + classifier_loss_region_selection + classifier_loss_region_abnormal + language_model_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            list_of_losses = [
                total_loss,
                obj_detector_losses,
                classifier_loss_region_selection,
                classifier_loss_region_abnormal,
                language_model_loss,
            ]

            # dicts are insertion ordered since Python 3.7
            for loss_type, loss in zip(train_losses_dict, list_of_losses):
                train_losses_dict[loss_type] += loss.item() * batch_size

            run_params["steps_taken"] += 1
            run_params["overall_steps_taken"] += 1

            is_epoch_end = True if (num_batch + 1) == len(train_dl) else False

            # evaluate every k steps and also at the end of an epoch
            if run_params["steps_taken"] >= EVALUATE_EVERY_K_STEPS or is_epoch_end:

                log.info(f"\nEvaluating at step {run_params['overall_steps_taken']}!\n")

                evaluate_model(model, train_losses_dict, val_dl, lr_scheduler, writer, tokenizer, run_params, is_epoch_end, generated_sentences_folder_path)

                log.info(f"\nMetrics evaluated at step {run_params['overall_steps_taken']}!\n")

                # reset values for the next evaluation
                for loss_type in train_losses_dict:
                    train_losses_dict[loss_type] = 0.0
                run_params["steps_taken"] = 0

                # set the model back to training
                model.train()

        # save the current best model weights at the end of each epoch
        torch.save(run_params["best_model_state"], run_params["best_model_save_path"])

    log.info("\nFinished training!")
    log.info(f"Lowest overall val loss: {run_params['lowest_val_loss']:.3f} at epoch {run_params['best_epoch']}")
    return None


def get_data_loaders(tokenizer, train_dataset, val_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    custom_collate_train = CustomCollator(tokenizer=tokenizer, is_val=False)
    custom_collate_val = CustomCollator(tokenizer=tokenizer, is_val=True)

    g = torch.Generator()
    g.manual_seed(seed_val)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=custom_collate_train,
        batch_size=BATCH_SIZE,
        shuffle=False,  # TODO: set shuffle back to True
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=custom_collate_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset_bounding_boxes
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
            A.ColorJitter(hue=0.0, saturation=0.0),
            A.Sharpen(alpha=(0.1, 0.2), lightness=0.0),
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            A.GaussNoise(),
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


def get_tokenized_datasets(tokenizer, raw_train_dataset, raw_val_dataset):
    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_train_dataset = raw_train_dataset.map(tokenize_function)
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

    return tokenized_train_dataset, tokenized_val_dataset


def get_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_datasets(config_file_path):
    path_dataset_object_detector = "/u/home/tanida/datasets/dataset-for-full-model-original-bbox-coordinates"

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

    datasets_as_dfs = {dataset: os.path.join(path_dataset_object_detector, dataset) + ".csv" for dataset in ["train", "valid", "test"]}

    datasets_as_dfs = {dataset: pd.read_csv(csv_file_path, usecols=usecols, converters=converters) for dataset, csv_file_path in datasets_as_dfs.items()}

    # bbox_phrases is a list of str
    # replace each bbox_phrase that is empty (i.e. "") by "#"
    # this is done such that model learns to generate the "#" symbol instead of "" for empty sentences
    # this is done because generated sentences that are "" (i.e. have len = 0) will cause problems when computing e.g. Bleu scores
    for dataset_df in datasets_as_dfs.values():
        dataset_df["bbox_phrases"] = dataset_df["bbox_phrases"].apply(lambda bbox_phrases: [phrase if len(phrase) != 0 else "#" for phrase in bbox_phrases])

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])

    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Train: {new_num_samples_train} images")
    log.info(f"Val: {new_num_samples_val} images")

    with open(config_file_path, "a") as f:
        f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
        f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

    # limit the datasets to those new numbers
    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    raw_train_dataset = Dataset.from_pandas(datasets_as_dfs["train"])
    raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

    return raw_train_dataset, raw_val_dataset


def create_run_folder():
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path_parent_dir = "/u/home/tanida/runs/full_model"

    run_folder_path = os.path.join(run_folder_path_parent_dir, f"run_{RUN}")
    weights_folder_path = os.path.join(run_folder_path, "weights")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")
    generated_sentences_folder_path = os.path.join(run_folder_path, "generated_sentences")

    if os.path.exists(run_folder_path):
        log.error(f"Folder to save run {RUN} already exists at {run_folder_path}.")
        log.error("Delete the folder or change the run number.")
        return None

    os.mkdir(run_folder_path)
    os.mkdir(weights_folder_path)
    os.mkdir(tensorboard_folder_path)
    os.mkdir(generated_sentences_folder_path)

    log.info(f"Run {RUN} folder created at {run_folder_path}.")

    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    config_parameters = {
        "COMMENT": RUN_COMMENT,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "PERCENTAGE_OF_TRAIN_SET_TO_USE": PERCENTAGE_OF_TRAIN_SET_TO_USE,
        "PERCENTAGE_OF_VAL_SET_TO_USE": PERCENTAGE_OF_VAL_SET_TO_USE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "EVALUATE_EVERY_K_STEPS": EVALUATE_EVERY_K_STEPS,
        "PATIENCE_LR_SCHEDULER": PATIENCE_LR_SCHEDULER,
        "NUM_BEAMS": NUM_BEAMS,
        "MAX_NUM_TOKENS_GENERATE": MAX_NUM_TOKENS_GENERATE,
        "NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE": NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
        "NUM_SENTENCES_TO_GENERATE": NUM_SENTENCES_TO_GENERATE,
    }

    with open(config_file_path, "w") as f:
        f.write(f"RUN {RUN}:\n")
        for param_name, param_value in config_parameters.items():
            f.write(f"\t{param_name}: {param_value}\n")

    return weights_folder_path, tensorboard_folder_path, config_file_path, generated_sentences_folder_path


def main():
    (
        weights_folder_path,
        tensorboard_folder_path,
        config_file_path,
        generated_sentences_folder_path,
    ) = create_run_folder()

    # the datasets still contain the untokenized phrases
    raw_train_dataset, raw_val_dataset = get_datasets(config_file_path)

    tokenizer = get_tokenizer()

    # tokenize the raw datasets
    tokenized_train_dataset, tokenized_val_dataset = get_tokenized_datasets(tokenizer, raw_train_dataset, raw_val_dataset)

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    train_dataset_complete = CustomDataset("train", tokenized_train_dataset, train_transforms)
    val_dataset_complete = CustomDataset("val", tokenized_val_dataset, val_transforms)

    train_loader, val_loader = get_data_loaders(tokenizer, train_dataset_complete, val_dataset_complete)

    model = ReportGenerationModel()
    model.to(device, non_blocking=True)
    model.train()

    opt = AdamW(model.parameters(), lr=LR)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", patience=PATIENCE_LR_SCHEDULER)
    writer = SummaryWriter(log_dir=tensorboard_folder_path)

    log.info("\nStarting training!\n")

    train_model(
        model=model,
        train_dl=train_loader,
        val_dl=train_loader,  # TODO: change back to val_loader
        optimizer=opt,
        lr_scheduler=lr_scheduler,
        epochs=EPOCHS,
        weights_folder_path=weights_folder_path,
        tokenizer=tokenizer,
        generated_sentences_folder_path=generated_sentences_folder_path,
        writer=writer,
    )


if __name__ == "__main__":
    main()
