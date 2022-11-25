"""
This module contains all functions used to evaluate the full model.

The (main) function evaluate_model of this module is called by the function train_model in train_full_model.py
every K steps and also at the end of every epoch.

The K is specified by the EVALUATE_EVERY_K_STEPS variable in run_configurations.py

evaluate_model and its sub-functions evaluate among other things:

    - total val loss as well as the val losses of each individual module (i.e. model component)
    - object detector:
        - average IoU of region (ideally 1.0 for every region)
        - average num detected regions per image (ideally 29.0)
        - average num each region is detected in an image (ideally 1.0 for every region)
    - binary classifier region selection:
        - precision, recall, f1 for all regions, regions that have gt = normal (i.e. the region was considered normal by the radiologist),
        regions that have gt = abnormal (i.e. the region was considered abnormal by the radiologist)
    - binary classifier region abnormal detection:
        - precision, recall, f1 for all regions
    - language model (is evaluated in separate evaluate_language_model.py module):
        - see doc string of evaluate_language_model.py for information on metrics
"""

import os

import torch
import torchmetrics
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.evaluate_full_model.evaluate_language_model import evaluate_language_model
from src.full_model.run_configurations import PRETRAIN_WITHOUT_LM_MODEL, WEIGHT_OBJECT_DETECTOR_LOSS, WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS, WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS, WEIGHT_LANGUAGE_MODEL_LOSS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_all_losses_and_scores_to_tensorboard(
    writer,
    overall_steps_taken,
    train_losses_dict,
    val_losses_dict,
    obj_detector_scores,
    region_selection_scores,
    region_abnormal_scores,
    language_model_scores,
    current_lr
):
    def write_losses():
        for loss_type in train_losses_dict:
            writer.add_scalars(
                "_loss",
                {f"{loss_type}_train": train_losses_dict[loss_type], f"{loss_type}_val": val_losses_dict[loss_type]},
                overall_steps_taken,
            )

    def write_obj_detector_scores():
        writer.add_scalar(
            "object_detector/avg_num_detected_regions_per_image",
            obj_detector_scores["avg_num_detected_regions_per_image"],
            overall_steps_taken,
        )

        writer.add_scalar("object_detector/iou/avg_iou", obj_detector_scores["avg_iou"], overall_steps_taken)

        # replace white space by underscore for each region name (i.e. "right upper lung" -> "right_upper_lung")
        anatomical_regions = ["_".join(region.split()) for region in ANATOMICAL_REGIONS]
        avg_detections_per_region = obj_detector_scores["avg_detections_per_region"]
        avg_iou_per_region = obj_detector_scores["avg_iou_per_region"]

        for region_, avg_detections_region in zip(anatomical_regions, avg_detections_per_region):
            writer.add_scalar(f"object_detector/num_detected/{region_}", avg_detections_region, overall_steps_taken)

        for region_, avg_iou_region in zip(anatomical_regions, avg_iou_per_region):
            writer.add_scalar(f"object_detector/iou/{region_}", avg_iou_region, overall_steps_taken)

    def write_region_selection_scores():
        for subset in region_selection_scores:
            for metric, score in region_selection_scores[subset].items():
                writer.add_scalar(f"region_select/{subset}/{metric}", score, overall_steps_taken)

    def write_region_abnormal_scores():
        for metric, score in region_abnormal_scores.items():
            writer.add_scalar(f"region_abnormal/{metric}", score, overall_steps_taken)

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
                writer.add_scalar(f"language_model/report/CE/{k}", v, overall_steps_taken)
            else:
                # k is a condition
                condition_name = "_".join(k.lower().split())
                for metric, score in ce_score_dict[k].items():
                    writer.add_scalar(f"language_model/report/CE/{condition_name}/{metric}", score, overall_steps_taken)

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
                        writer.add_scalar(f"language_model/region/{region_name_underscored}/{metric}", score, overall_steps_taken)
            else:
                for metric, score in language_model_scores[subset].items():
                    if metric == "CE":
                        ce_score_dict = language_model_scores["report"]["CE"]
                        write_clinical_efficacy_scores(ce_score_dict)
                    else:
                        writer.add_scalar(f"language_model/{subset}/{metric}", score, overall_steps_taken)

    write_losses()
    write_obj_detector_scores()
    write_region_selection_scores()
    write_region_abnormal_scores()

    if not PRETRAIN_WITHOUT_LM_MODEL and overall_steps_taken > 100000:
        write_language_model_scores()

    writer.add_scalar("lr", current_lr, overall_steps_taken)


def update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, region_is_abnormal, class_detected):
    """
    Args:
        region_abnormal_scores (Dict)
        predicted_abnormal_regions (Tensor[bool]): shape [batch_size x 29]
        region_is_abnormal (Tensor[bool]): shape [batch_size x 29]
        class_detected (Tensor[bool]): shape [batch_size x 29]

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
        selected_regions (Tensor[bool]): shape [batch_size x 29]
        region_has_sentence (Tensor[bool]): shape [batch_size x 29]
        region_is_abnormal (Tensor[bool]): shape [batch_size x 29]
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
            box (Tensor[batch_size x 29 x 4])

        Returns:
            area (Tensor[batch_size x 29])
        """
        x0 = box[..., 0]
        y0 = box[..., 1]
        x1 = box[..., 2]
        y1 = box[..., 3]

        return (x1 - x0) * (y1 - y0)

    def compute_intersection_and_union_area_per_region(detections, targets, class_detected):
        # pred_boxes is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
        # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
        # the 2nd to the 2nd class and so on
        pred_boxes = detections["top_region_boxes"]

        # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
        # gt_boxes is of shape [batch_size x 29 x 4]
        gt_boxes = torch.stack([t["boxes"] for t in targets], dim=0)

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


def get_val_losses_and_evaluate_obj_detector_and_binary_classifiers(model, val_dl, log_file, epoch):
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
    }

    if not PRETRAIN_WITHOUT_LM_MODEL:
        val_losses_dict["language_model_loss"] = 0.0

    """
    For the object detector, besides the obj_detector_val_loss, we also want to compute:
      - the average IoU for each region,
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
            # specifying average=None computes the metric for each class (i.e. negative and positive) separately
            # we then report the score of the positive class by indexing [1] once we've computed the final scores
            # this is equivalent to using average="binary" in sklearn.metric (with pos_label=1)
            #
            # note: using average="micro" is not correct, since it considers the negative and positive classes
            # to be separate classes (even in the binary case). If e.g. pred = True and ground-truth = False,
            # then it will be considered a FP for the positive class, but also a FN for the negative class,
            # which does not make any sense for the binary case and leads to incorrect scores
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
        # specifying average=None computes the metric for each class (i.e. negative and positive) separately
        # we then report the score of the positive class by indexing [1] once we've computed the final scores
        # this is equivalent to using average="binary" in sklearn.metric (with pos_label=1)
        "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
        "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
        "f1": torchmetrics.F1Score(num_classes=2, average=None).to(device),
    }

    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False

    num_images = 0

    # for normalizing the val losses
    steps_taken = 0

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl)):
            images = batch["images"]
            image_targets = batch["image_targets"]
            region_has_sentence = batch["region_has_sentence"]
            region_is_abnormal = batch["region_is_abnormal"]

            batch_size = images.size(0)
            num_images += batch_size

            images = images.to(device, non_blocking=True)
            image_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in image_targets]
            region_has_sentence = region_has_sentence.to(device, non_blocking=True)
            region_is_abnormal = region_is_abnormal.to(device, non_blocking=True)

            if not PRETRAIN_WITHOUT_LM_MODEL:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
            else:
                input_ids = None
                attention_mask = None

            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(images, image_targets, input_ids, attention_mask, region_has_sentence, region_is_abnormal)
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True

                    with open(log_file, "a") as f:
                        f.write("Evaluation:\n")
                        f.write(f"OOM at epoch {epoch}, batch number {num_batch}.\n")
                        f.write(f"Error message: {str(e)}\n\n")
                else:
                    raise e

            if oom:
                # free up memory
                torch.cuda.empty_cache()
                oom = False

                num_images -= batch_size

                continue

            # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
            if output == -1:
                with open(log_file, "a") as f:
                    f.write("Evaluation:\n")
                    f.write(f"Empty region features before language model at epoch {epoch}, batch number {num_batch}.\n\n")

                num_images -= batch_size

                continue

            if PRETRAIN_WITHOUT_LM_MODEL:
                (
                    obj_detector_loss_dict,
                    classifier_loss_region_selection,
                    classifier_loss_region_abnormal,
                    detections,
                    class_detected,
                    selected_regions,
                    predicted_abnormal_regions,
                ) = output
            else:
                (
                    obj_detector_loss_dict,
                    classifier_loss_region_selection,
                    classifier_loss_region_abnormal,
                    language_model_loss,
                    detections,
                    class_detected,  # bool tensor of shape [batch_size x 29]
                    selected_regions,  # bool tensor of shape [batch_size x 29]
                    predicted_abnormal_regions,  # bool tensor of shape [batch_size x 29]
                ) = output

            # detections is a dict with keys "top_region_boxes" and "top_scores"
            # "top_region_boxes" maps to a tensor of shape [batch_size x 29 x 4]
            # "top_scores" maps to a tensor of shape [batch_size x 29]

            # sum up all 4 losses from the object detector
            obj_detector_losses = sum(loss for loss in obj_detector_loss_dict.values())

            # sum up the rest of the losses
            total_loss = WEIGHT_OBJECT_DETECTOR_LOSS * obj_detector_losses + WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS * classifier_loss_region_selection + WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS * classifier_loss_region_abnormal

            if not PRETRAIN_WITHOUT_LM_MODEL:
                total_loss += WEIGHT_LANGUAGE_MODEL_LOSS * language_model_loss

            list_of_losses = [
                total_loss,
                obj_detector_losses,
                classifier_loss_region_selection,
                classifier_loss_region_abnormal,
            ]

            if not PRETRAIN_WITHOUT_LM_MODEL:
                list_of_losses.append(language_model_loss)

            # dicts are insertion ordered since Python 3.7
            for loss_type, loss in zip(val_losses_dict, list_of_losses):
                val_losses_dict[loss_type] += loss.item() * batch_size

            steps_taken += 1

            # update scores for object detector metrics
            update_object_detector_metrics(obj_detector_scores, detections, image_targets, class_detected)

            # update scores for region selection metrics
            update_region_selection_metrics(region_selection_scores, selected_regions, region_has_sentence, region_is_abnormal)

            # update scores for region abnormal detection metrics
            update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, region_is_abnormal, class_detected)

    # normalize the val losses by steps_taken
    for loss_type in val_losses_dict:
        val_losses_dict[loss_type] /= steps_taken

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

    return val_losses_dict, obj_detector_scores, region_selection_scores, region_abnormal_scores


def evaluate_model(model, train_losses_dict, val_dl, lr_scheduler, optimizer, scaler, writer, tokenizer, run_params, generated_sentences_and_reports_folder_path):
    model.eval()

    epoch = run_params["epoch"]
    steps_taken = run_params["steps_taken"]
    overall_steps_taken = run_params["overall_steps_taken"]
    log_file = run_params["log_file"]

    # normalize all train losses by steps_taken
    for loss_type in train_losses_dict:
        train_losses_dict[loss_type] /= steps_taken

    (
        val_losses_dict,
        obj_detector_scores,
        region_selection_scores,
        region_abnormal_scores,
    ) = get_val_losses_and_evaluate_obj_detector_and_binary_classifiers(model, val_dl, log_file, epoch)

    # the language model will generate gibberish in the beginning, so no need to evaluate it for first 100000 steps
    # (you may need to change this number based on the batch size you use, we used a small batch size of 2 for resource constraints)
    if not PRETRAIN_WITHOUT_LM_MODEL and overall_steps_taken > 100000:
        language_model_scores = evaluate_language_model(model, val_dl, tokenizer, writer, run_params, generated_sentences_and_reports_folder_path)
    else:
        language_model_scores = None

    current_lr = float(optimizer.param_groups[0]["lr"])

    write_all_losses_and_scores_to_tensorboard(
        writer,
        overall_steps_taken,
        train_losses_dict,
        val_losses_dict,
        obj_detector_scores,
        region_selection_scores,
        region_abnormal_scores,
        language_model_scores,
        current_lr
    )

    total_val_loss = val_losses_dict["total_loss"]

    # decrease lr if total_val_loss has not decreased after certain number of evaluations
    lr_scheduler.step(total_val_loss)

    # save model every time the val loss has decreased
    if total_val_loss < run_params["lowest_val_loss"]:
        run_params["lowest_val_loss"] = total_val_loss
        run_params["best_epoch"] = epoch

        save_path = os.path.join(run_params["checkpoints_folder_path"], f"checkpoint_val_loss_{total_val_loss:.3f}_overall_steps_{overall_steps_taken}.pt")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "current_epoch": epoch,
            "overall_steps_taken": overall_steps_taken,
            "lowest_val_loss": total_val_loss,
        }

        torch.save(checkpoint, save_path)
