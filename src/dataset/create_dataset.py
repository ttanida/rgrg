"""
Script creates train.csv, valid.csv, test.csv and test-2.csv to train object detector (as a standalone module),
object detector + binary classifiers, and full model.

Each row in the csv files specifies information about a single image.

The specific information (i.e. columns) of each row are:
    - subject_id: id of the patient whose image is used
    - study_id: id of the study of that patient (since a patient can have several studies done to document the progression of a disease etc.)
    - image_id: id of the single image
    - mimic_image_file_path: file path to the jpg of the single image on the workstation
    - bbox_coordinates (List[List[int]]): a nested list where the outer list (usually) has a length of 29 and the inner list always a length of 4 (for 4 bbox coordinates).
    Contains the bbox coordinates for all (usually 29) regions of a single image. There are some images that don't have bbox coordinates for all 29 regions
    (see log_file_dataset_creation.txt), thus it's possible that the outer list does not have length 29.
    - bbox_labels (List[int]): a list of (usually) length 29 that has region/class labels corresponding to the bbox_coordinates. Usually, the bbox_labels list will be
    of the form [1, 2, 3, ..., 28, 29], i.e. continuously counting from 1 to 29. It starts at 1 since 0 is considered the background class for object detectors.
    However, since some images don't have bbox coordinates for all 29 regions, it's possible that there are missing numbers in the list.
    - bbox_phrases (List[str]): a list of (always) length 29 that has the reference phrases for every bbox of a single image. Note that a lot of these reference phrases
    will be "" (i.e. empty), since a radiology report describing an image will usually not contain phrases for all 29 regions.
    - bbox_phrase_exists (List[bool]): a list of (always) length 29 that indicates if a bbox has a reference phrase (True) or not
    - bbox_is_abnormal (List[bool]): a list of (always) length 29 that indicates if a bbox was described as abnormal (True) by its reference phrase or not. bboxes that do
    not have a reference phrase are considered normal by default.

The valid.csv, test.csv and test-2.csv have the additional information of:
    - reference_report (str): the "findings" section of the MIMIC-CXR report corresponding to the image (see function get_reference_report)
    - preds_chexbert_ref_report (list[int]): list of len 14 (corresponding to 14 conditions as specified in src/CheXbert/src/constants.py)
    that contains 0's and 1's, where 1 means a condition is mentioned in the reference report

For the validation set, we only include images that have bbox_coordinates, bbox_labels for all 29 regions.
This is done because:
    1. We will usually not evaluate on the whole validation set (which contains 23,953 images), but only on a fraction of it (e.g. 5% - 20%).
    2. Writing code that evaluates on all 29 regions is easier and more performant (-> e.g. vectorization possible). If there are some images with < 29 regions,
    then the code has to accomodate them, making vectorization more difficult.

For the test set, we split it into 1 test set that only contains images with bbox_coordinates, bbox_labels for all 29 regions (which are around 95% of all test set images),
and 1 test set (called test-2.csv) that contains the remaining images that do not have bbox_coordinates, bbox_labels for all 29 regions (the remaining 5% of test set images).

This is done such that we can apply vectorized, efficient code to evaluate the 1st test set (which contains 95% of all test set images),
and more inefficient code to evaluate the 2nd test set (which only contains 5% of test set images), and of course the results of 1st and 2nd test set are reported together.

The train set contains all train images, even those without bbox_coordinates, bbox_labels for all 29 regions (because we don't need to evaluate on the train set).
"""
import csv
import json
import logging
import os
import re
import tempfile

import imagesize
import numpy as np
import spacy
import torch
import torch.nn as nn
from tqdm import tqdm

from src.CheXbert.src.label import label
from src.CheXbert.src.models.bert_labeler import bert_labeler
from src.dataset.constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE, SUBSTRINGS_TO_REMOVE
import src.dataset.section_parser as sp
from src.path_datasets_and_weights import path_chest_imagenome, path_mimic_cxr, path_mimic_cxr_jpg, path_full_dataset, path_chexbert_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# to log certain statistics during dataset creation
txt_file_for_logging = "log_file_dataset_creation.txt"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# constant specifies how many rows to create in the customized csv files
# can be useful to create small sample datasets (e.g. of len 50) for testing things
# if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is None, then all possible rows are created
NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES = None


def write_stats_to_log_file(
    dataset: str,
    num_images_ignored_or_avoided: int,
    missing_images: list[str],
    missing_reports: list[str],
    num_faulty_bboxes: int,
    num_images_without_29_regions: int
):
    with open(txt_file_for_logging, "a") as f:
        f.write(f"{dataset}:\n")
        f.write(f"\tnum_images_ignored_or_avoided: {num_images_ignored_or_avoided}\n")

        f.write(f"\tnum_missing_images: {len(missing_images)}\n")
        for missing_img in missing_images:
            f.write(f"\t\tmissing_img: {missing_img}\n")

        f.write(f"\tnum_missing_reports: {len(missing_reports)}\n")
        for missing_rep in missing_reports:
            f.write(f"\t\tmissing_rep: {missing_rep}\n")

        f.write(f"\tnum_faulty_bboxes: {num_faulty_bboxes}\n")
        f.write(f"\tnum_images_without_29_regions: {num_images_without_29_regions}\n\n")


def write_rows_in_new_csv_file(dataset: str, csv_rows: list[list]) -> None:
    log.info(f"Writing rows into new {dataset}.csv file...")

    if dataset == "test":
        csv_rows, csv_rows_less_than_29_regions = csv_rows

    new_csv_file_path = os.path.join(path_full_dataset, dataset)
    new_csv_file_path += ".csv" if not NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES else f"-{NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES}.csv"

    header = ["subject_id", "study_id", "image_id", "mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_phrases", "bbox_phrase_exists", "bbox_is_abnormal"]
    if dataset in ["valid", "test"]:
        header.extend(["reference_report", "preds_chexbert_ref_report"])

    with open(new_csv_file_path, "w") as fp:
        csv_writer = csv.writer(fp)

        csv_writer.writerow(header)
        csv_writer.writerows(csv_rows)

    if dataset == "test":
        new_csv_file_path = new_csv_file_path.replace(".csv", "-2.csv")

        with open(new_csv_file_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            csv_writer.writerows(csv_rows_less_than_29_regions)


def get_chexbert_predictions(reference_reports: list[str]) -> list[list[int]]:
    """
    Returns preds, a list[list[int]] with len(outer_list)=num_reports and len(inner_list)=14 (for 14 conditions, specified in CheXbert/src/constants.py).

    The function label from module CheXbert/src/label.py that extracts the disease labels/predictions for 14 diseases requires 2 input arguments:
        1. model (nn.Module): instantiated CheXbert model
        2. csv_path (str): path to a csv file with the reports. The csv file has to have 1 column titled "Report Impression"
        under which the reports can be found

    We use a temporary directory to create the csv file.

    The function label returns preds, which is a List[List[int]], with len(outer_list)=14, and len(inner_list)=num_reports.

    E.g. the 1st inner list could be [2, 1, 0, 3], which means the 1st report has label 2 for the 1st condition (which is 'Enlarged Cardiomediastinum'),
    the 2nd report has label 1 for the 1st condition, the 3rd report has label 0 for the 1st condition, the 4th and final report label 3 for the 1st condition.

    There are 4 possible labels:
        0: blank/NaN (i.e. no prediction could be made about a condition, because it was not mentioned in a report)
        1: positive (condition was mentioned as present in a report)
        2: negative (condition was mentioned as not present in a report)
        3: uncertain (condition was mentioned as possibly present in a report)

    Following the implementation of the paper "Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation"
    by Miura et. al., we merge negative and blank/NaN into one whole negative class, and positive and uncertain into one whole positive class.
    For reference, see lines 141 and 143 of Miura's implementation: https://github.com/ysmiura/ifcc/blob/master/eval_prf.py#L141,
    where label 3 is converted to label 1, and label 2 is converted to label 0.

    Finally, we transpose preds, such that len(outer_list)=num_reports and len(inner_list)=14.
    """
    def get_chexbert():
        model = bert_labeler()
        model = nn.DataParallel(model)  # needed since weights were saved with nn.DataParallel
        checkpoint = torch.load(path_chexbert_weights, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        return model

    def convert_labels_and_transpose(preds: list[list[int]]) -> list[list[int]]:
        """
        See doc string of get_chexbert_predictions for details.
        Converts label 2 -> label 0 and label 3 -> label 1.
        """
        preds = np.array(preds)

        preds[preds == 2] = 0
        preds[preds == 3] = 1

        preds = preds.transpose()
        preds = preds.tolist()

        return preds

    chexbert = get_chexbert()

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_report_path = os.path.join(temp_dir, "report.csv")

        header = ["Report Impression"]

        with open(csv_report_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            csv_writer.writerows([[report] for report in reference_reports])

        # preds is list[list[int]] with len(outer_list)=14 (for 14 conditions) and len(inner_list)=num_reports
        preds = label(chexbert, csv_report_path)

    # convert the labels as specified in the doc string of get_chexbert_predictions,
    # and also transpose preds such that we have preds in the format list[list[int]] with len(outer_list)=num_reports and len(inner_list)=14.
    # This is because we want to store each report with the corresponding predicted 14 conditions in a row in a csv file.
    preds = convert_labels_and_transpose(preds)

    return preds


def append_ref_reports_and_chexbert_preds_to_csv_rows(csv_rows, reference_reports):
    # chexbert_preds is list[list[int]] with len(outer_list)=num_reports and len(inner_list)=14 (for 14 conditions, specified in CheXbert/src/constants.py)
    chexbert_preds = get_chexbert_predictions(reference_reports)

    for i, (ref_report, cb_preds_report) in enumerate(zip(reference_reports, chexbert_preds)):
        csv_rows[i].extend([ref_report, cb_preds_report])


def check_coordinate(coordinate: int, dim: int) -> int:
    """
    If an coordinate is smaller than 0 or bigger than its corresponding dimension (for x-coordinates the corresponding dim is width, for y-coordinates it's height),
    then we set the coordinate to 0 or dim respectively as to not get an exception when an image is cropped by the coordinates in the CustomImageDataset __getitem__ method.
    """
    if coordinate < 0:
        coordinate = 0
    elif coordinate > dim:
        coordinate = dim
    return coordinate


def coordinates_faulty(height, width, x1, y1, x2, y2) -> bool:
    """
    Bbox coordinates are faulty if:
        - bbox coordinates specify a bbox outside of the image
        - bbox coordinates specify a bbox of area = 0 (if x1 == x2 or y1 == y2).

    We have to make this check, since for some unknown reason in the chest-imagenome dataset there are:
        - negative bbox coordinates
        - bbox coordinates bigger than the given image height and weight
        - bbox coordinates where x1 == x2 or y1 == y2

    Returns True if coordinates are faulty, False otherwise.

    Firstly checks if area is zero, i.e. x1 == x2 or y1 == y2

    Secondly checks if the bottom right corner (specified by (x2, y2)) is within the image (see smaller_than_zero).
    Since we always have x1 < x2 and y1 < y2, we know that if x2 < 0, then x1 < x2 <= 0, thus the bbox is not within the image (same for y1, y2).

    Thirdly checks if the top left corner (specified by (x1, y1)) is within the image (see exceeds_limits).
    We know that if x1 > width, then x2 > x1 >= width, thus the bbox is not within the image (same for y1, y2).
    """
    area_of_bbox_is_zero = x1 == x2 or y1 == y2
    smaller_than_zero = x2 <= 0 or y2 <= 0
    exceeds_limits = x1 >= width or y1 >= height

    return area_of_bbox_is_zero or smaller_than_zero or exceeds_limits


def determine_if_abnormal(attributes_list: list[list]) -> bool:
    """
    attributes_list is a list of lists that contains attributes corresponding to the phrases describing a specific bbox.

    E.g. the phrases: ['Right lung is clear without pneumothorax.', 'No pneumothorax identified.'] have the attributes_list
    [['anatomicalfinding|no|lung opacity', 'anatomicalfinding|no|pneumothorax', 'nlp|yes|normal'], ['anatomicalfinding|no|pneumothorax']],
    where the 1st inner list contains the attributes pertaining to the 1st phrase, and the 2nd inner list contains attributes for the 2nd phrase respectively.

    Phrases describing abnormalities have the attribute 'nlp|yes|abnormal'.
    """
    for attributes in attributes_list:
        for attribute in attributes:
            if attribute == "nlp|yes|abnormal":
                return True

    # no abnormality could be detected
    return False


def convert_phrases_to_single_string(phrases: list[str], sentence_tokenizer) -> str:
    """
    Takes a list of phrases describing the region of a single bbox and returns a single string.

    Also performs operations to clean the single string, such as:
        - removes irrelevant substrings (like "PORTABLE UPRIGHT AP VIEW OF THE CHEST:")
        - removes whitespace characters (e.g. \n or \t) and redundant whitespaces
        - capitalizes the first word in each sentence
        - removes duplicate sentences

    Args:
        phrases (list[str]): in the attribute dictionary, phrases is originally a list of strings
        sentence_tokenizer (spacy sentence tokenizer): used in capitalize_first_word_in_sentence function

    Returns:
        phrases (str): a single string, with the list of strings concatenated
    """
    def remove_substrings(phrases):
        def remove_wet_read(phrases):
            """Removes substring like 'WET READ: ___ ___ 8:19 AM' that is irrelevant."""
            # since there can be multiple WET READS's, collect the indices where they start and end in index_slices_to_remove
            index_slices_to_remove = []
            for index in range(len(phrases)):
                if phrases[index:index + 8] == "WET READ":

                    # curr_index searches for "AM" or "PM" that signals the end of the WET READ substring
                    for curr_index in range(index + 8, len(phrases)):
                        # since it's possible that a WET READ substring does not have an"AM" or "PM" that signals its end, we also have to break out of the iteration
                        # if the next WET READ substring is encountered
                        if phrases[curr_index:curr_index + 2] in ["AM", "PM"] or phrases[curr_index:curr_index + 8] == "WET READ":
                            break

                    # only add (index, curr_index + 2) (i.e. the indices of the found WET READ substring) to index_slices_to_remove if an "AM" or "PM" were found
                    if phrases[curr_index:curr_index + 2] in ["AM", "PM"]:
                        index_slices_to_remove.append((index, curr_index + 2))

            # remove the slices in reversed order, such that the correct index order is preserved
            for indices_tuple in reversed(index_slices_to_remove):
                start_index, end_index = indices_tuple
                phrases = phrases[:start_index] + phrases[end_index:]

            return phrases

        phrases = remove_wet_read(phrases)
        phrases = re.sub(SUBSTRINGS_TO_REMOVE, "", phrases, flags=re.DOTALL)

        return phrases

    def remove_whitespace(phrases):
        phrases = " ".join(phrases.split())
        return phrases

    def capitalize_first_word_in_sentence(phrases, sentence_tokenizer):
        sentences = sentence_tokenizer(phrases).sents

        # capitalize the first letter of each sentence
        phrases = " ".join(sent.text[0].upper() + sent.text[1:] for sent in sentences)

        return phrases

    def remove_duplicate_sentences(phrases):
        # remove the last period
        if phrases[-1] == ".":
            phrases = phrases[:-1]

        # dicts are insertion ordered as of Python 3.6
        phrases_dict = {phrase: None for phrase in phrases.split(". ")}

        phrases = ". ".join(phrase for phrase in phrases_dict)

        # add last period
        return phrases + "."

    # convert list of phrases into a single phrase
    phrases = " ".join(phrases)

    # remove "PORTABLE UPRIGHT AP VIEW OF THE CHEST:" and similar substrings from phrases, since they don't add any relevant information
    phrases = remove_substrings(phrases)

    # remove all whitespace characters (multiple whitespaces, newlines, tabs etc.)
    phrases = remove_whitespace(phrases)

    # for consistency, capitalize the 1st word in each sentence
    phrases = capitalize_first_word_in_sentence(phrases, sentence_tokenizer)

    phrases = remove_duplicate_sentences(phrases)

    return phrases


def get_attributes_dict(image_scene_graph: dict, sentence_tokenizer) -> dict[tuple]:
    attributes_dict = {}
    for attribute in image_scene_graph["attributes"]:
        region_name = attribute["bbox_name"]

        # ignore region_names such as "left chest wall" or "right breast" that are not part of the 29 anatomical regions
        if region_name not in ANATOMICAL_REGIONS:
            continue

        phrases = convert_phrases_to_single_string(attribute["phrases"], sentence_tokenizer)
        is_abnormal = determine_if_abnormal(attribute["attributes"])

        attributes_dict[region_name] = (phrases, is_abnormal)

    return attributes_dict


def get_reference_report(subject_id: str, study_id: str, missing_reports: list[str]):
    def process_report(report: str):
        SUBSTRING_TO_REMOVE_FROM_REPORT = "1. |2. |3. |4. |5. |6. |7. |8. |9."

        # remove substrings
        report = re.sub(SUBSTRING_TO_REMOVE_FROM_REPORT, "", report, flags=re.DOTALL)

        # remove unnecessary whitespaces
        report = " ".join(report.split())

        if report[-1] != ".":
            report + "."

        return report

    # custom_section_names and custom_indices specify reports that don't have "findings" sections
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    if f"s{study_id}" in custom_section_names or f"s{study_id}" in custom_indices:
        return -1  # skip all reports without "findings" sections

    path_to_report = os.path.join(path_mimic_cxr, "files", f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")

    if not os.path.exists(path_to_report):
        shortened_path_to_report = os.path.join(f"p{subject_id[:2]}", f"p{subject_id}", f"s{study_id}.txt")
        missing_reports.append(shortened_path_to_report)
        return -1

    with open(path_to_report) as f:
        report = "".join(f.readlines())

    # split report into sections
    # section_names is a list that specifies the found sections, e.g. ["indication", "comparison", "findings", "impression"]
    # sections is a list of same length that contains the corresponding text from the sections specified in section_names
    sections, section_names, _ = sp.section_text(report)

    if "findings" in section_names:
        findings_index = section_names.index("findings")
        report = sections[findings_index]
    else:
        return -1  # skip all reports without "findings" sections

    report = process_report(report)

    return report


def get_total_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        return sum(1 for row in csv_reader)


def get_rows(dataset: str, path_csv_file: str, image_ids_to_avoid: set) -> list[list]:
    """
    Args:
        dataset (str): either "train", "valid" or "test
        path_csv_file (str): path to one of the csv files in the folder silver_dataset/splits of the chest-imagenome-dataset
        image_ids_to_avoid (set): as specified in "silver_dataset/splits/images_to_avoid.csv"

    Returns:
        csv_rows (list[list]): inner list contains information about a single image:
            - subject_id (str)
            - study_id (str)
            - image_id (str)
            - mimic_image_file_path (str): file path to image in mimic-cxr-jpg dataset
            - bbox_coordinates (list[list[int]]), where outer list usually has len 29 and inner list contains 4 bbox coordinates
            - bbox_labels (list[int]): list with class labels for each ground-truth box
            - bbox_phrases (list[str]): list with phrases for each bbox (note: phrases can be empty, i.e. "")
            - bbox_phrase_exist_vars (list[bool]): list that specifies if a phrase is non-empty (True) or empty (False) for a given bbox
            - bbox_is_abnormal_vars (list[bool]): list that specifies if a region depicted in a bbox is abnormal (True) or normal (False)

        valid.csv, test.csv and test-2.csv have these 2 additional fields:

            - reference_report (str): the findings section of the report extracted via https://github.com/MIT-LCP/mimic-cxr/tree/master/txt
            - preds_chexbert_ref_report (list[int]): of the form e.g. [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0] for 14 conditions
            (as specified in src/CheXbert/src/constants.py), where 1 means condition is present in the given image
    """
    csv_rows = []
    num_rows_created = 0

    # we split the test set into 1 that contains all images that have bbox coordinates for all 29 regions
    # (which will be around 44969 images in total, or around 95% of all test set images),
    # and 1 that contains the rest of the images (around 2420 images) that do not have bbox coordinates for all 29 regions
    # this is done such that we can efficiently evaluate the first test set (since vectorized code can be written for it),
    # and evaluate the second test set a bit more inefficiently (using for loops) afterwards
    if dataset == "test":
        csv_rows_less_than_29_regions = []

    total_num_rows = get_total_num_rows(path_csv_file)

    # used in function convert_phrases_to_single_string
    sentence_tokenizer = spacy.load("en_core_web_trf")

    # stats will be logged in path_to_log_file
    num_images_ignored_or_avoided = 0
    num_faulty_bboxes = 0
    num_images_without_29_regions = 0
    missing_images = []
    if dataset in ["valid", "test"]:
        missing_reports = []

    # for the validation and test sets, we need the reference reports for evaluation
    # for the test set, we differentiate between the reports correponding to the images in csv_rows,
    # and those corresponding to the images in csv_rows_less_than_29_regions
    if dataset == "valid":
        reference_reports = []
    if dataset == "test":
        reference_reports = []
        reference_reports_less_than_29_regions = []

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # iterate over all rows of the given csv file (i.e. over all images), if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is not set to a specific value
        for row in tqdm(csv_reader, total=total_num_rows):
            subject_id = row[1]
            study_id = row[2]
            image_id = row[3]

            # all images in set IMAGE_IDS_TO_IGNORE seem to be failed x-rays and thus have to be discarded
            # (they also don't have corresponding scene graph json files anyway)
            # all images in set image_ids_to_avoid are image IDs for images in the gold standard dataset,
            # which should all be excluded from model training and validation
            if image_id in IMAGE_IDS_TO_IGNORE or image_id in image_ids_to_avoid:
                num_images_ignored_or_avoided += 1
                continue

            # image_file_path is of the form "files/p10/p10000980/s50985099/6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d.dcm"
            # i.e. f"files/p{subject_id[:2]}/p{subject_id}/s{study_id}/{image_id}.dcm"
            # since we have the MIMIC-CXR-JPG dataset, we need to replace .dcm by .jpg
            image_file_path = row[4].replace(".dcm", ".jpg")
            mimic_image_file_path = os.path.join(path_mimic_cxr_jpg, image_file_path)

            if not os.path.exists(mimic_image_file_path):
                missing_images.append(mimic_image_file_path)
                continue

            # for the validation and test sets, we only want to include images that have corresponding reference reports with "findings" sections
            if dataset in ["valid", "test"]:
                reference_report = get_reference_report(subject_id, study_id, missing_reports)

                # skip images that don't have a reference report with "findings" section
                if reference_report == -1:
                    continue

                # the reference_report will be appended to the list reference_reports (or possibly reference_reports_less_than_29_regions in the case of test set)
                # later on, once the new_image_row (declared further below, which contains all information about a single image) is ultimately appended to csv_rows
                # (or possibly csv_rows_less_than_29_regions in the case of test set)

            chest_imagenome_scene_graph_file_path = os.path.join(path_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"

            with open(chest_imagenome_scene_graph_file_path) as fp:
                image_scene_graph = json.load(fp)

            # get the attributes specified for the specific image in its image_scene_graph
            # the attributes contain (among other things) phrases used in the reference report used to describe different bbox regions and
            # information whether a described bbox region is normal or abnormal
            #
            # anatomical_region_attributes is a dict with bbox_names as keys and lists that contain 2 elements as values. The 2 list elements are:
            # 1. (normalized) phrases, which is a single string that contains the phrases used to describe the region inside the bbox
            # 2. is_abnormal, a boolean that is True if the region inside the bbox is considered abnormal, else False for normal
            anatomical_region_attributes = get_attributes_dict(image_scene_graph, sentence_tokenizer)

            # new_image_row will store all information about 1 image as a row in the csv file
            new_image_row = [subject_id, study_id, image_id, mimic_image_file_path]
            bbox_coordinates = []
            bbox_labels = []
            bbox_phrases = []
            bbox_phrase_exist_vars = []
            bbox_is_abnormal_vars = []

            width, height = imagesize.get(mimic_image_file_path)

            # counter to see if given image contains bbox coordinates for all 29 regions
            # if image does not bbox coordinates for 29 regions, it's still added to the train and test dataset,
            # but not the val dataset (see reasoning in the module docstring on top of this file)
            num_regions = 0

            region_to_bbox_coordinates_dict = {}
            # objects is a list of obj_dicts where each dict contains the bbox coordinates for a single region
            for obj_dict in image_scene_graph["objects"]:
                region_name = obj_dict["bbox_name"]
                x1 = obj_dict["original_x1"]
                y1 = obj_dict["original_y1"]
                x2 = obj_dict["original_x2"]
                y2 = obj_dict["original_y2"]

                region_to_bbox_coordinates_dict[region_name] = [x1, y1, x2, y2]

            for anatomical_region in ANATOMICAL_REGIONS:
                bbox_coords = region_to_bbox_coordinates_dict.get(anatomical_region, None)

                # if there are no bbox coordinates or they are faulty, then don't add them to image information
                if bbox_coords is None or coordinates_faulty(height, width, *bbox_coords):
                    num_faulty_bboxes += 1
                else:
                    x1, y1, x2, y2 = bbox_coords

                    # it is possible that the bbox is only partially inside the image height and width (if e.g. x1 < 0, whereas x2 > 0)
                    # to prevent these cases from raising an exception, we set the coordinates to 0 if coordinate < 0, set to width if x-coordinate > width
                    # and set to height if y-coordinate > height
                    x1 = check_coordinate(x1, width)
                    y1 = check_coordinate(y1, height)
                    x2 = check_coordinate(x2, width)
                    y2 = check_coordinate(y2, height)

                    bbox_coords = [x1, y1, x2, y2]

                    # since background has class label 0 for object detection, shift the remaining class labels by 1
                    class_label = ANATOMICAL_REGIONS[anatomical_region] + 1

                    bbox_coordinates.append(bbox_coords)
                    bbox_labels.append(class_label)

                    num_regions += 1

                # get bbox_phrase (describing the region inside bbox) and bbox_is_abnormal boolean variable (indicating if region inside bbox is abnormal)
                # if there is no phrase, then the region inside bbox is normal and thus has "" for bbox_phrase (empty phrase) and False for bbox_is_abnormal
                bbox_phrase, bbox_is_abnormal = anatomical_region_attributes.get(anatomical_region, ("", False))
                bbox_phrase_exist = True if bbox_phrase != "" else False

                bbox_phrases.append(bbox_phrase)
                bbox_phrase_exist_vars.append(bbox_phrase_exist)
                bbox_is_abnormal_vars.append(bbox_is_abnormal)

            new_image_row.extend([bbox_coordinates, bbox_labels, bbox_phrases, bbox_phrase_exist_vars, bbox_is_abnormal_vars])

            # for train set, add all images (even those that don't have bbox information for all 29 regions)
            # for val set, only add images that have bbox information for all 29 regions
            # for test set, distinguish between test set 1 that contains test set images that have bbox information for all 29 regions
            # (around 95% of all test set images)
            if dataset == "train" or (dataset in ["valid", "test"] and num_regions == 29):
                csv_rows.append(new_image_row)

                if dataset in ["valid", "test"]:
                    reference_reports.append(reference_report)

                num_rows_created += 1
            # test set 2 will contain the remaining 5% of test set images, which do not have bbox information for all 29 regions
            elif dataset == "test" and num_regions != 29:
                csv_rows_less_than_29_regions.append(new_image_row)
                reference_reports_less_than_29_regions.append(reference_report)

            if num_regions != 29:
                num_images_without_29_regions += 1

            # break out of loop if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is specified
            if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES and num_rows_created >= NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES:
                break

    write_stats_to_log_file(dataset, num_images_ignored_or_avoided, missing_images, missing_reports, num_faulty_bboxes, num_images_without_29_regions)

    if dataset in ["valid", "test"]:
        append_ref_reports_and_chexbert_preds_to_csv_rows(csv_rows, reference_reports)

    if dataset == "test":
        append_ref_reports_and_chexbert_preds_to_csv_rows(csv_rows_less_than_29_regions, reference_reports_less_than_29_regions)
        return csv_rows, csv_rows_less_than_29_regions
    else:
        return csv_rows


def create_new_csv_file(dataset: str, path_csv_file: str, image_ids_to_avoid: set) -> None:
    log.info(f"Creating new {dataset}.csv file...")

    # get rows to create new csv_file
    # csv_rows is a list of lists, where an inner list specifies all information about a single image
    csv_rows = get_rows(dataset, path_csv_file, image_ids_to_avoid)

    # write those rows into a new csv file
    write_rows_in_new_csv_file(dataset, csv_rows)

    log.info(f"Creating new {dataset}.csv file... DONE!")


def create_new_csv_files(csv_files_dict, image_ids_to_avoid):
    if os.path.exists(path_full_dataset):
        log.error(f"Full dataset folder already exists at {path_full_dataset}.")
        log.error("Delete dataset folder or rename variable path_full_dataset in src/path_datasets_and_weights.py before running script to create new folder!")
        return None

    os.mkdir(path_full_dataset)
    for dataset, path_csv_file in csv_files_dict.items():
        create_new_csv_file(dataset, path_csv_file, image_ids_to_avoid)


def get_images_to_avoid():
    path_to_images_to_avoid = os.path.join(path_chest_imagenome, "silver_dataset", "splits", "images_to_avoid.csv")

    image_ids_to_avoid = set()

    with open(path_to_images_to_avoid) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        for row in csv_reader:
            image_id = row[2]
            image_ids_to_avoid.add(image_id)

    return image_ids_to_avoid


def get_train_val_test_csv_files():
    """Return a dict with datasets as keys and paths to the corresponding csv files in the chest-imagenome dataset as values"""
    path_to_splits_folder = os.path.join(path_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def main():
    csv_files_dict = get_train_val_test_csv_files()

    # the "splits" directory of chest-imagenome contains a csv file called "images_to_avoid.csv",
    # which contains image IDs for images in the gold standard dataset, which should all be excluded
    # from model training and validation
    image_ids_to_avoid = get_images_to_avoid()

    create_new_csv_files(csv_files_dict, image_ids_to_avoid)


if __name__ == "__main__":
    main()
