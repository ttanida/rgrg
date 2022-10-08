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
import csv
import io
import os
import tempfile

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import spacy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from tqdm import tqdm

from src.CheXbert.src.constants import CONDITIONS
from src.CheXbert.src.label import label
from src.CheXbert.src.models.bert_labeler import bert_labeler
from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.run_configurations import (
    BATCH_SIZE,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
    NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
    NUM_IMAGES_TO_PLOT,
    BERTSCORE_SIMILARITY_THRESHOLD,
)
from src.path_datasets_and_weights import path_chexbert_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_language_model_scores(gen_and_ref_sentences, gen_and_ref_reports):
    def compute_clinical_efficacy_scores(subset: str, gen_reports: list[str], ref_reports: list[str]):
        """
        Note that this function is also used to compute the CE scores for generated and reference sentences (as opposed to reports).

        To get the CE scores, we first need the disease labels extracted by CheXbert

        The function label from module CheXbert/src/label.py that extracts these labels requires 2 input arguments:
            1. chexbert (nn.Module): instantiated chexbert model
            2. csv_path (str): path to the csv file with the reports. The csv file has to have 1 column titled "Report Impression"
            under which the reports can be found

        We use a temporary directory to create the csv files for the generated and reference reports.

        The function label returns preds_gen_reports and preds_ref_reports respectively, which are List[List[int]],
        with the outer list always having len=14 (for 14 conditions, specified in CheXbert/src/constants.py),
        and the inner list of len=num_reports.

        E.g. the 1st inner list could be [2, 1, 0, 3], which means the 1st report has label 2 for the 1st condition (which is 'Enlarged Cardiomediastinum'),
        the 2nd report has label 1 for the 1st condition, the 3rd report has label 0 for the 1st condition, the 4th and final report label 3 for the 1st condition.

        There are 4 possible labels:
            0: blank/NaN (i.e. no prediction could be made about a condition, because it was no mentioned in a report)
            1: positive (condition was mentioned as present in a report)
            2: negative (condition was mentioned as not present in a report)
            3: uncertain (condition was mentioned as possibly present in a report)

        Following the implementation of the paper "Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation"
        by Miura et. al., we merge negative and blank/NaN into one whole negative class, and positive and uncertain into one whole positive class.
        For reference, see lines 141 and 143 of Miura's implementation: https://github.com/ysmiura/ifcc/blob/master/eval_prf.py#L141,
        where label 3 is converted to label 1, and label 2 is converted to label 0.
        """
        def convert_labels(preds_reports: list[list[int]]):
            """
            See doc string of update_clinical_efficacy_scores function for more details.
            Converts label 2 -> label 0 and label 3 -> label 1.
            """
            def convert_label(label: int):
                if label == 2:
                    return 0
                elif label == 3:
                    return 1
                else:
                    return label

            preds_reports = [[convert_label(label) for label in condition_list] for condition_list in preds_reports]

            return preds_reports

        def get_chexbert():
            model = bert_labeler()
            model = nn.DataParallel(model)  # needed since weights were saved with nn.DataParallel
            checkpoint = torch.load(path_chexbert_weights, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model = model.to(device)
            model.eval()

            return model

        # note that this function works just as well for generated and reference sentences
        # I just didn't want to make the variable names more complicated

        chexbert = get_chexbert()

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_gen_reports_file_path = os.path.join(temp_dir, "gen_reports.csv")
            csv_ref_reports_file_path = os.path.join(temp_dir, "ref_reports.csv")

            header = ["Report Impression"]

            with open(csv_gen_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[gen_report] for gen_report in gen_reports])

            with open(csv_ref_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[ref_report] for ref_report in ref_reports])

            # preds_*_reports are List[List[int]] with the labels extracted by CheXbert (see doc string for details)
            preds_gen_reports = label(chexbert, csv_gen_reports_file_path)
            preds_ref_reports = label(chexbert, csv_ref_reports_file_path)

        preds_gen_reports = convert_labels(preds_gen_reports)
        preds_ref_reports = convert_labels(preds_ref_reports)

        # for the CE scores, we follow Miura (https://arxiv.org/pdf/2010.10042.pdf) in averaging them over these 5 conditions:
        five_conditions_to_evaluate = {"Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"}

        total_preds_gen_reports_5_conditions = []
        total_preds_ref_reports_5_conditions = []

        # iterate over the 14 conditions
        for preds_gen_reports_condition, preds_ref_reports_condition, condition in zip(preds_gen_reports, preds_ref_reports, CONDITIONS):
            if condition in five_conditions_to_evaluate:
                total_preds_gen_reports_5_conditions.extend(preds_gen_reports_condition)
                total_preds_ref_reports_5_conditions.extend(preds_ref_reports_condition)

            # only evaluate each individual condition for the reference reports
            if subset == "report":
                precision, recall, f1, _ = precision_recall_fscore_support(preds_ref_reports_condition, preds_gen_reports_condition, average="binary")
                acc = accuracy_score(preds_ref_reports_condition, preds_gen_reports_condition)

                language_model_scores[subset]["CE"][condition]["precision"] = precision
                language_model_scores[subset]["CE"][condition]["recall"] = recall
                language_model_scores[subset]["CE"][condition]["f1"] = f1
                language_model_scores[subset]["CE"][condition]["acc"] = acc

        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions)

        language_model_scores[subset]["CE"]["precision"] = precision
        language_model_scores[subset]["CE"]["recall"] = recall
        language_model_scores[subset]["CE"]["f1"] = f1
        language_model_scores[subset]["CE"]["acc"] = acc

    def compute_sentence_level_scores():
        def remove_gen_sents_corresponding_to_empty_ref_sents(gen_sents, ref_sents):
            """
            We can't compute BLEU-scores on generated sentences, whose corresponding reference sentence is empty.
            So we need to discard them both.
            """
            filtered_gen_sents = []
            filtered_ref_sents = []

            for gen_sent, ref_sent in zip(gen_sents, ref_sents):
                if ref_sent != "":
                    filtered_gen_sents.append(gen_sent)
                    filtered_ref_sents.append(ref_sent)

            return filtered_gen_sents, filtered_ref_sents

        def compute_sent_level_scores_for_subset(subset, gen_sents, ref_sents):
            for metric, score in language_model_scores[subset].items():
                if metric.startswith("bleu"):
                    bleu_score_type = int(metric[-1])
                    bleu_result = score.compute(predictions=gen_sents, references=ref_sents, max_order=bleu_score_type)["bleu"]
                    language_model_scores[subset][metric] = bleu_result
                elif metric == "CE":
                    compute_clinical_efficacy_scores(subset, gen_sents, ref_sents)

        def compute_sent_level_scores_for_region(region_name, gen_sents, ref_sents):
            for metric, score in language_model_scores["region"][region_name].items():
                bleu_score_type = int(metric[-1])
                bleu_result = score.compute(predictions=gen_sents, references=ref_sents, max_order=bleu_score_type)["bleu"]
                language_model_scores["region"][region_name][metric] = bleu_result

        generated_sents = gen_and_ref_sentences["generated_sentences"]
        generated_sents_normal = gen_and_ref_sentences["generated_sentences_normal_selected_regions"]
        generated_sents_abnormal = gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"]

        reference_sents = gen_and_ref_sentences["reference_sentences"]
        reference_sents_normal = gen_and_ref_sentences["reference_sentences_normal_selected_regions"]
        reference_sents_abnormal = gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"]

        generated_sents, reference_sents = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents, reference_sents)
        generated_sents_normal, reference_sents_normal = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents_normal, reference_sents_normal)
        generated_sents_abnormal, reference_sents_abnormal = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents_abnormal, reference_sents_abnormal)

        compute_sent_level_scores_for_subset("all", generated_sents, reference_sents)
        compute_sent_level_scores_for_subset("normal", generated_sents_normal, reference_sents_normal)
        compute_sent_level_scores_for_subset("abnormal", generated_sents_abnormal, reference_sents_abnormal)

        for region_index, region_name in enumerate(ANATOMICAL_REGIONS):
            region_generated_sentences = gen_and_ref_sentences[region_index]["generated_sentences"]
            region_reference_sentences = gen_and_ref_sentences[region_index]["reference_sentences"]

            region_generated_sentences, region_reference_sentences = remove_gen_sents_corresponding_to_empty_ref_sents(region_generated_sentences, region_reference_sentences)

            if len(region_generated_sentences) != 0:
                compute_sent_level_scores_for_region(region_name, region_generated_sentences, region_reference_sentences)
            else:
                for metric, _ in language_model_scores["region"][region_name].items():
                    language_model_scores["region"][region_name][metric] = -1

    def compute_report_level_scores():
        gen_reports = gen_and_ref_reports["generated_reports"]
        ref_reports = gen_and_ref_reports["reference_reports"]

        for metric, score in language_model_scores["report"].items():
            if metric.startswith("bleu"):
                bleu_score_type = int(metric[-1])
                bleu_result = score.compute(predictions=gen_reports, references=ref_reports, max_order=bleu_score_type)["bleu"]
                language_model_scores["report"][metric] = float(bleu_result)
            elif metric == "meteor":
                meteor_result = score.compute(predictions=gen_reports, references=ref_reports)["meteor"]
                language_model_scores["report"][metric] = float(meteor_result)
            elif metric == "rouge":
                rouge_result = score.compute(predictions=gen_reports, references=ref_reports)["rougeL"]
                language_model_scores["report"][metric] = float(rouge_result)
            elif metric == "CE":
                compute_clinical_efficacy_scores("report", gen_reports, ref_reports)

    def create_language_model_scores_dict():
        language_model_scores = {}

        # compute bleu scores and clinical efficacy (CE) scores for all, normal, abnormal reference sentences and reference reports
        for subset in ["all", "normal", "abnormal", "report"]:
            language_model_scores[subset] = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}
            language_model_scores[subset]["CE"] = {
                # following Miura (https://arxiv.org/pdf/2010.10042.pdf), we evaluate the micro average CE scores over these 5 diseases:
                # Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"
                # the averages will be calculated with sklearn precision_recall_fscore_support and accuracy_score later
                "precision": None,
                "recall": None,
                "f1": None,
                "acc": None
            }

        # for the reference report, we also compute the CE scores for each of the 14 conditions individually
        for condition in CONDITIONS:
            language_model_scores["report"]["CE"][condition] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "acc": None
            }

        # compute meteor, rouge-L for reference reports
        language_model_scores["report"]["meteor"] = evaluate.load("meteor")
        language_model_scores["report"]["rouge"] = evaluate.load("rouge")

        # also compute bleu scores for reference sentences of each region individually
        language_model_scores["region"] = {}
        for region_name in ANATOMICAL_REGIONS:
            language_model_scores["region"][region_name] = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}

        return language_model_scores

    language_model_scores = create_language_model_scores_dict()

    compute_report_level_scores()
    compute_sentence_level_scores()

    return language_model_scores


def write_sentences_and_reports_to_file(
    gen_and_ref_sentences,
    gen_and_ref_reports,
    generated_sentences_and_reports_folder_path,
    overall_steps_taken,
):
    def write_sentences():
        txt_file_name = os.path.join(generated_sentences_and_reports_folder_path, "generated_sentences", f"generated_sentences_step_{overall_steps_taken}")
        txt_file_name_abnormal = os.path.join(generated_sentences_and_reports_folder_path, "generated_sentences", f"generated_abnormal_sentences_step_{overall_steps_taken}")

        with open(txt_file_name, "w") as f:
            for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")

        with open(txt_file_name_abnormal, "w") as f:
            for gen_sent, ref_sent in zip(generated_sentences_abnormal_regions, reference_sentences_abnormal_regions):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")

    def write_reports():
        txt_file_name = os.path.join(
            generated_sentences_and_reports_folder_path,
            "generated_reports",
            f"generated_reports_step_{overall_steps_taken}",
        )

        with open(txt_file_name, "w") as f:
            for gen_report, ref_report, removed_similar_gen_sents in zip(generated_reports, reference_reports, removed_similar_generated_sentences):
                f.write(f"Generated report: {gen_report}\n\n")
                f.write(f"Reference report: {ref_report}\n\n")
                f.write("Generated sentences that were removed:\n")
                for gen_sent, list_similar_gen_sents in removed_similar_gen_sents.items():
                    f.write(f"\t{gen_sent} == {list_similar_gen_sents}\n")
                f.write("\n")
                f.write("=" * 30)
                f.write("\n\n")

    # all below are list of str
    generated_sentences = gen_and_ref_sentences["generated_sentences"][:NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE]
    generated_sentences_abnormal_regions = gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"][:NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE]
    reference_sentences = gen_and_ref_sentences["reference_sentences"][:NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE]
    reference_sentences_abnormal_regions = gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"][:NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE]

    write_sentences()

    # all below are list of str except removed_similar_generated_sentences which is a list of dict
    generated_reports = gen_and_ref_reports["generated_reports"][:NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE]
    reference_reports = gen_and_ref_reports["reference_reports"][:NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE]
    removed_similar_generated_sentences = gen_and_ref_reports["removed_similar_generated_sentences"][:NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE]

    write_reports()


def get_plot_title(region_set, region_indices, region_colors, class_detected_img) -> str:
    """
    Get a plot title like in the below example.
    1 region_set always contains 6 regions (except for region_set_5, which has 5 regions).
    The characters in the brackets represent the colors of the corresponding bboxes (e.g. b = blue),
    "nd" stands for "not detected" in case the region was not detected by the object detector.

    right lung (b), right costophrenic angle (g, nd), left lung (r)
    left costophrenic angle (c), cardiac silhouette (m), spine (y, nd)
    """
    # get a list of 6 boolean values that specify if that region was detected
    class_detected = [class_detected_img[region_index] for region_index in region_indices]

    # add color_code to region name (e.g. "(r)" for red)
    # also add nd to the brackets if region was not detected (e.g. "(r, nd)" if red region was not detected)
    region_set = [
        region + f" ({color})" if cls_detect else region + f" ({color}, nd)"
        for region, color, cls_detect in zip(region_set, region_colors, class_detected)
    ]

    # add a line break to the title, as to not make it too long
    return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])


def get_generated_sentence_for_region(
    generated_sentences_for_selected_regions, selected_regions, num_img, region_index
) -> str:
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): holds the generated sentences for all regions that were selected in the batch, i.e. of length "num_regions_selected_in_batch"
        selected_regions (Tensor[bool]): of shape [batch_size x 29], specifies for each region if it was selected to get a sentences generated (True) or not by the binary classifier for region selection.
        Ergo has exactly "num_regions_selected_in_batch" True values.
        num_img (int): specifies the image we are currently processing in the batch, its value is in the range [0, batch_size-1]
        region_index (int): specifies the region we are currently processing of a single image, its value is in the range [0, 28]

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

    In this toy example, the batch_size = 2 and there are only 3 regions in total for simplicity (instead of the 29).
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


def update_region_set_text(
    region_set_text,
    color,
    reference_sentences_img,
    generated_sentences_for_selected_regions,
    region_index,
    selected_regions,
    num_img,
):
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

    region_set_text += f"  reference: {reference_sentence_region}\n"

    box_region_selected = selected_regions[num_img][region_index]
    if not box_region_selected:
        region_set_text += "  generated: [REGION NOT SELECTED]\n\n"
    else:
        generated_sentence_region = get_generated_sentence_for_region(
            generated_sentences_for_selected_regions, selected_regions, num_img, region_index
        )
        generated_sentence_region = transform_sentence_to_fit_under_image(generated_sentence_region)
        region_set_text += f"  generated: {generated_sentence_region}\n\n"

    return region_set_text


def plot_box(box, ax, clr, linestyle, region_detected=True):
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    ax.add_artist(
        plt.Rectangle(xy=(x0, y0), height=h, width=w, fill=False, color=clr, linewidth=1, linestyle=linestyle)
    )

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
    # pred_boxes_batch is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
    # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
    # the 2nd to the 2nd class and so on
    pred_boxes_batch = detections["top_region_boxes"]

    # image_targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
    # gt_boxes is of shape [batch_size x 29 x 4]
    gt_boxes_batch = torch.stack([t["boxes"] for t in image_targets], dim=0)

    # plot 6 regions at a time, as to not overload the image with boxes (except for region_set_5, which has 5 regions)
    # the region_sets were chosen as to minimize overlap between the contained regions (i.e. better visibility)
    region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
    region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
    region_set_3 = ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"]
    region_set_4 = ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"]
    region_set_5 = ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]

    regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5]

    # put channel dimension (1st dim) last (0-th dim is batch-dim)
    images = images.numpy().transpose(0, 2, 3, 1)

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
            region_colors = ["b", "g", "r", "c", "m", "y"]

            if num_region_set == 4:
                region_colors.pop()

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
                    region_set_text,
                    color,
                    reference_sentences_img,
                    generated_sentences_for_selected_regions,
                    region_index,
                    selected_regions,
                    num_img,
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
            writer.add_image(
                f"img_{writer_image_num}_region_set_{num_region_set}",
                im,
                global_step=overall_steps_taken,
                dataformats="HWC",
            )

            plt.close(fig)


def update_gen_and_ref_sentences_for_regions(
    gen_and_ref_sentences,
    generated_sents_for_selected_regions,
    reference_sents_for_selected_regions,
    selected_regions
):
    """Updates the gen_and_ref_sentences dict for each of the 29 regions, i.e. appends the generated and reference sentences for the regions (if they exist)

    Args:
        gen_and_ref_sentences (dict):
        generated_sents_for_selected_regions (List[str]): has exactly num_regions_selected_in_batch generated sentences
        reference_sents_for_selected_regions (List[str]): has exactly num_regions_selected_in_batch reference sentences
        selected_regions (np.array([bool])): of shape batch_size x 29, has exactly num_regions_selected_in_batch True values
        that specify the regions for whom sentences were generated

    Implementation is not too easy to understand, so here is a toy example with some toy values to explain.

    generated_sents_for_selected_regions = ["Heart is ok.", "Spine is ok."]
    reference_sents_for_selected_regions = ["Cardiac silhouette is ok.", "Spine is not ok."]
    selected_regions = [
        [False, False, True],
        [True, False, False]
    ]

    In this toy example, the batch_size = 2 and there are only 3 regions in total for simplicity (instead of the 29).
    The generated_sents_for_selected_regions and reference_sents_for_selected_regions are of len 2, meaning num_regions_selected_in_batch = 2.
    Therefore, the selected_regions boolean array also has exactly 2 True values.

    (1) Flatten selected_regions:
        selected_regions_flat = [False, False, True, True, False, False]

    (2) Iterate until 1st True value is found in selected_regions_flat:
        index_gen_ref_sentence = 0 at the moment
        curr_index = 2 at the moment
        We do a modulo operation to get the region_index, i.e. region_index = curr_index % 3 = 2

        We get the gen_sent and ref_sent at index_gen_ref_sentence, i.e.
        gen_sent = "Heart is ok."
        ref_sent = "Cardiac silhouette is ok."

        We append them to the respective lists in gen_and_ref_sentences[region_index]

        We increase index_gen_ref_sentence by 1, such that at the next True value the next gen_sent and ref_sent are taken.

    (2) Iterate until the 2nd True value is found in selected_regions_flat:
        index_gen_ref_sentence = 1 at the moment
        curr_index = 3 at the moment
        We do a modulo operation to get the region_index, i.e. region_index = curr_index % 3 = 0

        We get the gen_sent and ref_sent at index_gen_ref_sentence, i.e.
        gen_sent = "Spine is ok."
        ref_sent = "Spine is not ok."

        We append them to the respective lists in gen_and_ref_sentences[region_index]

        We increase index_gen_ref_sentence by 1, such that at the next True value the next gen_sent and ref_sent are taken.
    """
    index_gen_ref_sentence = 0

    # of shape (batch_size * 29)
    selected_regions_flat = selected_regions.reshape(-1)
    for curr_index, region_selected_bool in enumerate(selected_regions_flat):
        if region_selected_bool:
            region_index = curr_index % 29
            gen_sent = generated_sents_for_selected_regions[index_gen_ref_sentence]
            ref_sent = reference_sents_for_selected_regions[index_gen_ref_sentence]

            gen_and_ref_sentences[region_index]["generated_sentences"].append(gen_sent)
            gen_and_ref_sentences[region_index]["reference_sentences"].append(ref_sent)

            index_gen_ref_sentence += 1


def get_generated_reports(generated_sentences_for_selected_regions, selected_regions, sentence_tokenizer, bertscore_threshold):
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): of length "num_regions_selected_in_batch"
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
        sentence_tokenizer: used in remove_duplicate_generated_sentences to separate the generated sentences

    Return:
        generated_reports (List[str]): list of length batch_size containing generated reports for every image in batch
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

        # TODO:
        # the nested for loops below check each generated sentence with every other generated sentence
        # this is not particularly efficient, since e.g. generated sentences for the region "right lung" most likely
        # will never be similar to generated sentences for the region "abdomen"
        # thus, one could speed up these checks by only checking anatomical regions that are similar to each other

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

                if bert_score_result["f1"][0] > bertscore_threshold:
                    # remove the generated similar sentence that is shorter
                    if len(gen_sent_1) > len(gen_sent_2):
                        similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                    else:
                        similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

        gen_report_single_image = " ".join(
            sent for sent in gen_sents_single_image if not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
        )

        return gen_report_single_image, similar_generated_sents_to_be_removed

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


def get_ref_sentences_for_selected_regions(reference_sentences, selected_regions):
    """
    Args:
        reference_sentences (List[List[str]]): outer list has len batch_size, inner list has len 29 (the inner list holds all reference phrases of a single image)
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
    """
    # array of shape [batch_size x 29]
    reference_sentences = np.asarray(reference_sentences)

    ref_sentences_for_selected_regions = reference_sentences[selected_regions]

    return ref_sentences_for_selected_regions.tolist()


def get_sents_for_normal_abnormal_selected_regions(region_is_abnormal, selected_regions, generated_sentences_for_selected_regions, reference_sentences_for_selected_regions):
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


def evaluate_language_model(model, val_dl, tokenizer, writer, run_params, generated_sentences_and_reports_folder_path):
    epoch = run_params["epoch"]
    overall_steps_taken = run_params["overall_steps_taken"]
    log_file = run_params["log_file"]

    gen_and_ref_sentences = {
        "generated_sentences": [],
        "generated_sentences_normal_selected_regions": [],
        "generated_sentences_abnormal_selected_regions": [],
        "reference_sentences": [],
        "reference_sentences_normal_selected_regions": [],
        "reference_sentences_abnormal_selected_regions": [],
    }

    # also examine the generated and reference sentences on per region basis
    for region_index, _ in enumerate(ANATOMICAL_REGIONS):
        gen_and_ref_sentences[region_index] = {
            "generated_sentences": [],
            "reference_sentences": []
        }

    gen_and_ref_reports = {
        "generated_reports": [],
        "removed_similar_generated_sentences": [],
        "reference_reports": [],
    }

    # we also want to plot a couple of images
    num_batches_to_process_for_image_plotting = NUM_IMAGES_TO_PLOT // BATCH_SIZE

    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False

    # used in function get_generated_and_reference_reports
    sentence_tokenizer = spacy.load("en_core_web_trf")

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION):
            # since generating sentences takes some time, we limit the number of batches used to compute bleu/rouge-l/meteor
            if num_batch >= NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION:
                break

            images = batch["images"]  # shape [batch_size x 1 x 512 x 512]
            image_targets = batch["image_targets"]
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
                # selected_regions is of shape [batch_size x 29] and is True for regions that should get a sentence
                beam_search_output, selected_regions, detections, class_detected = output
                selected_regions = selected_regions.detach().cpu().numpy()

            # generated_sents_for_selected_regions is a List[str] of length "num_regions_selected_in_batch"
            generated_sents_for_selected_regions = tokenizer.batch_decode(
                beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # filter reference_sentences to those that correspond to the generated_sentences for the selected regions.
            # reference_sents_for_selected_regions will therefore be a List[str] of length "num_regions_selected_in_batch"
            # (i.e. same length as generated_sents_for_selected_regions)
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
                    generated_sents_for_selected_regions,
                )

    write_sentences_and_reports_to_file(
        gen_and_ref_sentences,
        gen_and_ref_reports,
        generated_sentences_and_reports_folder_path,
        overall_steps_taken,
    )

    language_model_scores = compute_language_model_scores(gen_and_ref_sentences, gen_and_ref_reports)

    return language_model_scores
