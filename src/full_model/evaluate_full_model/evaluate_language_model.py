"""
This module contains all functions used to evaluate the language model.

The (main) function evaluate_language_model of this module is called by the function evaluate_model in evaluate_model.py.

evaluate_language_model returns language_model_scores which include:
    - the BLEU 1-4 and BertScore for all generated sentences
    - the BLEU 1-4 and BertScore for all generated sentences with gt = normal (i.e. the region was considered normal by the radiologist)
    - the BLEU 1-4 and BertScore for all generated sentences with gt = abnormal (i.e. the region was considered abnormal by the radiologist).

It also calls subfunctions which:
    - save NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE (see run_configurations.py) batches of generated sentences as a txt file
    (for manual verification what the model generates)
    - save NUM_IMAGES_TO_PLOT (see run_configurations.py) images to tensorboard where gt and predicted bboxes for every region are depicted,
    as well as the generated sentences (if they exist) and reference sentences for every region
"""
import io
import os

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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


def compute_final_language_model_scores(language_model_scores):
    for subset in language_model_scores:
        temp = {}
        for metric, score in language_model_scores[subset].items():
            if metric.startswith("bleu"):
                bleu_score_type = int(metric[-1])
                result = score.compute(max_order=bleu_score_type)
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
    def write_sentences(generated_sentences, reference_sentences, is_abnormal):
        txt_file_name = f"generated{'' if not is_abnormal else '_abnormal'}_sentences_step_{overall_steps_taken}"
        txt_file_name = os.path.join(generated_sentences_folder_path, txt_file_name)

        with open(txt_file_name, "w") as f:
            for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
                f.write(f"Generated sentence: {gen_sent}\n")
                # the hash symbol symbolizes an empty reference sentence, and thus can be replaced by '' when writing to file
                f.write(f"Reference sentence: {ref_sent if ref_sent != '#' else ''}\n\n")

    # generated_sentences is a list of str
    generated_sentences = gen_and_ref_sentences_to_save_to_file["generated_sentences"]
    generated_abnormal_sentences = gen_and_ref_sentences_to_save_to_file["generated_abnormal_sentences"]

    # reference_sentences is a list of str
    reference_sentences = gen_and_ref_sentences_to_save_to_file["reference_sentences"]
    reference_abnormal_sentences = gen_and_ref_sentences_to_save_to_file["reference_abnormal_sentences"]

    write_sentences(generated_sentences, reference_sentences, is_abnormal=False)
    write_sentences(generated_abnormal_sentences, reference_abnormal_sentences, is_abnormal=True)


def get_plot_title(region_set, region_indices, region_colors, class_detected_img) -> str:
    """
    Get a plot title like in the below example.
    1 region_set always contains 6 regions.
    The characters in the brackets represent the colors of the corresponding bboxes (e.g. b = blue),
    "nd" stands for "not detected" in case the region was not detected by the object detector.

    right lung (b), right costophrenic angle (g, nd), left lung (r)
    left costophrenic angle (c), cardiac silhouette (m), spine (y, nd)
    """
    # get a list of 6 boolean values that specify if that region was detected
    class_detected = [class_detected_img[region_index] for region_index in region_indices]

    # add color_code to region name (e.g. "(r)" for red)
    # also add nd to the brackets if region was not detected (e.g. "(r, nd)" if red region was not detected)
    region_set = [region + f" ({color})" if cls_detect else region + f" ({color}, nd)" for region, color, cls_detect in zip(region_set, region_colors, class_detected)]

    # add a line break to the title, as to not make it too long
    return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])


def get_generated_sentence_for_region(generated_sentences_for_selected_regions, selected_regions, num_img, region_index) -> str:
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
    cum_sum_true_values = torch.cumsum(selected_regions_flat, dim=0)

    cum_sum_true_values = cum_sum_true_values.reshape(selected_regions.shape)
    cum_sum_true_values -= 1

    index = cum_sum_true_values[num_img][region_index]

    return generated_sentences_for_selected_regions[index]


def transform_sentence_to_fit_under_image(sentence):
    """
    Adds line breaks and whitespaces such that long reference or generated sentence
    fits under the plotted image.
    Values like max_line_length and prefix_for_alignment were found by trial-and-error.
    """
    max_line_length = 60
    if len(sentence) < max_line_length:
        return sentence

    words = sentence.split()
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
    """
    Create a single string region_set_text like in the example below.
    Each update creates 1 paragraph for 1 region/bbox.
    The (b), (r) and (y) represent the colors of the bounding boxes (in this case blue, red and yellow).

    Example:

    (b):
      reference: Normal cardiomediastinal silhouette, hila, and pleura.
      generated: The mediastinal and hilar contours are unremarkable.

    (r):
      reference:
      generated: [REGION NOT SELECTED]

    (y):
      reference:
      generated: There is no pleural effusion or pneumothorax.

    (... continues for 3 more regions/bboxes, for a total of 6 per region_set)
    """
    region_set_text += f"({color}):\n"
    reference_sentence_region = reference_sentences_img[region_index]

    # in case sentence is too long
    reference_sentence_region = transform_sentence_to_fit_under_image(reference_sentence_region)

    # replace empty reference sentences (symbolized by #) by an empty string
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
    num_batch,
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

    selected_regions = selected_regions.detach().cpu()

    for num_img, image in enumerate(images):

        gt_boxes_img = gt_boxes_batch[num_img]
        pred_boxes_img = pred_boxes_batch[num_img]
        class_detected_img = class_detected[num_img].tolist()
        reference_sentences_img = reference_sentences[num_img]

        for num_region_set, region_set in enumerate(regions_sets):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()

            plt.imshow(image, cmap="gray")
            plt.axis("on")

            region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]

            region_set_text = ""

            for region_index, color in zip(region_indices, region_colors):
                # box_gt and box_pred are both [List[float]] of len 4
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

            # using writer.add_figure does not correctly display the region_set_text in tensorboard
            # so instead, fig is first saved as a png file to memory via BytesIO
            # (this also saves the region_set_text correctly in the png when bbox_inches="tight" is set)
            # then the png is loaded from memory and the 4th channel (alpha channel) is discarded
            # finally, writer.add_image is used to display the image in tensorboard
            buf = io.BytesIO()
            fig.savefig(buf, bbox_inches="tight")
            buf.seek(0)
            im = Image.open(buf)
            im = np.asarray(im)[..., :3]

            writer_image_num = num_batch * BATCH_SIZE + num_img
            writer.add_image(f"img_{writer_image_num}_region_set_{num_region_set}", im, global_step=overall_steps_taken, dataformats="HWC")


def update_language_model_scores(language_model_scores, generated_sentences_for_selected_regions, reference_sentences_for_selected_regions, selected_regions, region_is_abnormal):
    def get_sents_for_normal_abnormal_selected_regions():
        selected_region_is_abnormal = region_is_abnormal[selected_regions]
        # selected_region_is_abnormal is a bool array of shape [num_regions_selected_in_batch] that specifies if a selected region is abnormal (True) or normal (False)

        selected_region_is_abnormal = selected_region_is_abnormal.detach().cpu().numpy()

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

    for score in language_model_scores["all"].values():
        score.add_batch(predictions=generated_sentences_for_selected_regions, references=reference_sentences_for_selected_regions)

    # for computing the scores for the normal and abnormal reference sentences, we have to filter the generated and reference sentences accordingly
    (
        gen_sents_for_normal_selected_regions,
        gen_sents_for_abnormal_selected_regions,
        ref_sents_for_normal_selected_regions,
        ref_sents_for_abnormal_selected_regions,
    ) = get_sents_for_normal_abnormal_selected_regions()

    if len(ref_sents_for_normal_selected_regions) != 0:
        for score in language_model_scores["normal"].values():
            score.add_batch(predictions=gen_sents_for_normal_selected_regions, references=ref_sents_for_normal_selected_regions)

    if len(ref_sents_for_abnormal_selected_regions) != 0:
        for score in language_model_scores["abnormal"].values():
            score.add_batch(predictions=gen_sents_for_abnormal_selected_regions, references=ref_sents_for_abnormal_selected_regions)

    return gen_sents_for_abnormal_selected_regions, ref_sents_for_abnormal_selected_regions


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


def evaluate_language_model(model, val_dl, tokenizer, writer, run_params, generated_sentences_folder_path):
    epoch = run_params["epoch"]
    overall_steps_taken = run_params["overall_steps_taken"]
    log_file = run_params["log_file"]

    # compute scores for all, normal and abnormal reference sentences
    subsets = ["all", "normal", "abnormal"]
    language_model_scores = {}

    for subset in subsets:
        language_model_scores[subset] = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}
        language_model_scores[subset]["bert_score"] = evaluate.load("bertscore")

    gen_and_ref_sentences_to_save_to_file = {
        "generated_sentences": [],
        "reference_sentences": [],
        "generated_abnormal_sentences": [],
        "reference_abnormal_sentences": [],
    }

    # since generating sentences takes some time, we only generate NUM_SENTENCES_TO_GENERATE sentences
    num_batches_to_process_for_sentence_generation = NUM_SENTENCES_TO_GENERATE_FOR_EVALUATION // BATCH_SIZE

    # we also want to plot a couple of images
    num_batches_to_process_for_image_plotting = NUM_IMAGES_TO_PLOT // BATCH_SIZE

    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=num_batches_to_process_for_sentence_generation):
            if num_batch >= num_batches_to_process_for_sentence_generation:
                break

            images = batch["images"]  # shape [batch_size x 1 x 512 x 512]
            image_targets = batch["image_targets"]
            region_is_abnormal = batch["region_is_abnormal"]  # boolean tensor of shape [batch_size x 36]

            # List[List[str]] that holds the reference phrases. The inner list holds all reference phrases of a single image
            reference_sentences = batch["reference_sentences"]

            try:
                output = model.generate(images.to(device, non_blocking=True), max_length=MAX_NUM_TOKENS_GENERATE, num_beams=NUM_BEAMS, early_stopping=True)
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True

                    with open(log_file, "a") as f:
                        f.write("Generation:\n")
                        f.write(f"OOM at epoch {epoch}, batch number {num_batch}.\n")
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
                with open(log_file, "a") as f:
                    f.write("Generation:\n")
                    f.write(f"Empty region features before language model at epoch {epoch}, batch number {num_batch}.\n\n")

                continue
            else:
                beam_search_output, selected_regions, detections, class_detected = output

            # generated_sentences_for_selected_regions is a List[str] of length "num_regions_selected_in_batch"
            generated_sentences_for_selected_regions = tokenizer.batch_decode(beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # filter reference_sentences to those that correspond to the generated_sentences for the selected regions.
            # reference_sentences_for_selected_regions will therefore be a List[str] of length "num_regions_selected_in_batch"
            # (i.e. same length as generated_sentences_for_selected_regions)
            reference_sentences_for_selected_regions = get_ref_sentences_for_selected_regions(reference_sentences, selected_regions)

            gen_sents_for_abnormal_selected_regions, ref_sents_for_abnormal_selected_regions = update_language_model_scores(
                language_model_scores,
                generated_sentences_for_selected_regions,
                reference_sentences_for_selected_regions,
                selected_regions,
                region_is_abnormal,
            )

            if num_batch < NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE:
                gen_and_ref_sentences_to_save_to_file["generated_sentences"].extend(generated_sentences_for_selected_regions)
                gen_and_ref_sentences_to_save_to_file["generated_abnormal_sentences"].extend(gen_sents_for_abnormal_selected_regions)
                gen_and_ref_sentences_to_save_to_file["reference_sentences"].extend(reference_sentences_for_selected_regions)
                gen_and_ref_sentences_to_save_to_file["reference_abnormal_sentences"].extend(ref_sents_for_abnormal_selected_regions)

            if num_batch < num_batches_to_process_for_image_plotting:
                plot_detections_and_sentences_to_tensorboard(
                    writer,
                    num_batch,
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

    compute_final_language_model_scores(language_model_scores)

    return language_model_scores
