"""
This module contains all functions used to evaluate the language model.

The (main) function evaluate_language_model of this module is called by the function evaluate_model in evaluate_model.py.

evaluate_language_model returns language_model_scores which include the BLEU 1-4 and BertScore for all generated sentences,
generated sentences with gt = normal (i.e. the region was considered normal by the radiologist) and generated sentences with gt = abnormal
(i.e. the region was considered abnormal by the radiologist).

It also calls subfunctions which:
    - save NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE (see run_configurations.py) batches of generated sentences as a txt file
    (for manual verification what the model generates)
    - save NUM_IMAGES_TO_PLOT (see run_configurations.py) images to tensorboard where gt and predicted bboxes for every region
        are depicted, as well as the generated sentences (if they exist) and reference sentences for every region
"""
import os

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.run_configurations import (
    BATCH_SIZE,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_SENTENCES_TO_GENERATE_FOR_EVALUATION,
    NUM_IMAGES_TO_PLOT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_plot_title(region_set, region_indices, region_colors, class_detected_img):
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
    region_set_text += f"  reference: {reference_sentence_region if reference_sentence_region != '#' else ''}\n"

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

    # put channel dimension (1st dim) last (0-th dim is batch-dim)
    images = images.numpy().transpose(0, 2, 3, 1)

    for num_img, image in enumerate(images):

        gt_boxes_img = gt_boxes_batch[num_img]
        pred_boxes_img = pred_boxes_batch[num_img]
        class_detected_img = class_detected[num_img].tolist()
        selected_regions = selected_regions.detach().cpu()
        reference_sentences_img = reference_sentences[num_img]

        for num_region_set, region_set in enumerate(regions_sets):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()

            plt.imshow(image, cmap="gray")
            plt.axis("on")

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

            title = get_plot_title(region_set, region_indices, region_colors, class_detected_img)
            ax.set_title(title)

            plt.xlabel(region_set_text, loc="left")

            writer.add_figure(f"img_{num_img}_region_set_{num_region_set}", fig, overall_steps_taken)


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
            # the hash symbol symbolizes an empty reference sentence, and thus can be replaced by '' when writing to file
            f.write(f"Reference sentence: {ref_sent if ref_sent != '#' else ''}\n\n")


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
    num_batches_to_process_for_sentence_generation = NUM_SENTENCES_TO_GENERATE_FOR_EVALUATION // BATCH_SIZE

    # we also want to plot a couple of images
    num_batches_to_process_for_image_plotting = NUM_IMAGES_TO_PLOT // BATCH_SIZE

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=num_batches_to_process_for_sentence_generation):
            if num_batch >= num_batches_to_process_for_sentence_generation:
                break

            images = batch["images"]  # shape [batch_size x 1 x 512 x 512]
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

            if num_batch < num_batches_to_process_for_image_plotting:
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
                )

    write_sentences_to_file(gen_and_ref_sentences_to_save_to_file, generated_sentences_folder_path, overall_steps_taken)

    # compute final scores for language model metrics
    compute_final_language_model_scores(language_model_scores)

    return language_model_scores
