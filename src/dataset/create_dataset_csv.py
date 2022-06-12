"""
Script for creating custom train, valid, test csv files.

Each row in the csv files specifies information about a single bbox (i.e. single anatomical region) of a single image.

The specific information are:
    - subject_id: id of the patient whose image is used
    - study_id: id of the study of that patient (since a patient can have several studies done to document the progression of a disease etc.)
    - image_id: id of the single image
    - mimic_image_file_path: file path to the jpg of the single image on the workstation
    - bbox_name: name of one of the anatomical region that is specified by the row
    - x1 / y1 / x2 / y2 : bbox coordinates of said anatomical region in the single image
    - is_abnormal: boolean variable that specifies if anatomical region is abnormal.
    The value of the variable (True/False) is derived from the report corresponding to the single image (see determine_if_abnormal function).

The custom train, valid, test csv files contain the bbox information of the images specified in the train, valid, test csv files of the
chest-imagenome-dataset-1.0.0/silver_dataset/splits/ folder.
"""

import csv
import json
import logging
import os

import cv2
from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Lowercase
from tqdm import tqdm

from constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE

path_to_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"
path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_to_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr-jpg"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# constant specifies how many rows to create in the customized csv files
# if constant is None, then all possible rows are created (resulting in csv files of huge file sizes)
NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES = None


def write_rows_in_new_csv_file(dataset: str, new_rows: list[list]) -> None:
    log.info(f"Writing rows into new {dataset}.csv file...")

    new_csv_file_path = os.path.join(path_to_chest_imagenome_customized, dataset)
    new_csv_file_path += ".csv" if not NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES else f"-{NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES}.csv"

    with open(new_csv_file_path, "w") as fp:
        csv_writer = csv.writer(fp)

        header = ["index", "subject_id", "study_id", "image_id", "mimic_image_file_path", "bbox_name", "x1", "y1", "x2", "y2", "phrases", "is_abnormal"]

        csv_writer.writerow(header)
        csv_writer.writerows(new_rows)


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


def check_coordinates_out_of_bounds(height, width, x1, y1, x2, y2) -> bool:
    """
    Checks if bbox coordinates are outside the image.

    We have to make this check, since for some unknown reason there are negative bbox coordinates in the chest-imagenome dataset,
    as well as bbox coordinates bigger than the given image height and weight.

    Returns True if coordinates out of bounds, False otherwise.

    Firstly checks if the bottom right corner (specified by (x2, y2)) is within the image (see smaller_than_zero).
    Since we always have x1 < x2 and y1 < y2, we know that if x2 < 0, then x1 < x2 < 0, thus the bbox is not within the image (same for y1, y2).

    Secondly checks if the top left corner (specified by (x1, y1)) is within the image (see exceeds_limits).
    We know that if x1 > width, then x2 > x1 > width, thus the bbox is not within the image (same for y1, y2).
    """
    smaller_than_zero = x2 < 0 or y2 < 0
    exceeds_limits = x1 > width or y1 > height

    return smaller_than_zero or exceeds_limits


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


def normalize_text(phrases: list[str]) -> str:
    """
    Takes a list of phrases describing the region of a single bbox and returns a single normalized string.
    (Probably pre-trained tokenizer already applies normalization, so only joining list of strings to a single string is strictly necessary.)

    Normalization operations:

    - concatenation of list of strings to a single string
    - unicode normalization (NFKC)
    - lowercasing
    - removing whitespace characters (e.g. \n or \t) and redundant whitespaces

    Args:
        phrases (list[str]): in the attribute dictionary, phrases is originally a list of strings

    Returns:
        phrases (str): a single normalized string, with the list of strings concatenated
    """
    # convert list of phrases into a single phrase
    phrases = " ".join(phrases)

    # apply a sequence of normalization operations
    normalizer = normalizers.Sequence([NFKC(), Lowercase()])
    phrases = normalizer.normalize_str(phrases)

    # remove all whitespace characters (multiple whitespaces, newlines, tabs etc.)
    phrases = " ".join(phrases.split())

    return phrases


def get_attributes_dict(image_scene_graph: dict) -> dict[list]:
    attributes_dict = {}
    for attribute in image_scene_graph["attributes"]:
        bbox_name = attribute["bbox_name"]

        # ignore bbox_names such as "left chest wall" or "right breast" that don't appear in the 36 anatomical regions that have bbox coordiantes
        if bbox_name not in ANATOMICAL_REGIONS:
            continue

        phrases = normalize_text(attribute["phrases"])
        is_abnormal = determine_if_abnormal(attribute["attributes"])

        attributes_dict[bbox_name] = [phrases, is_abnormal]

    return attributes_dict


def get_total_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        return sum(1 for row in csv_reader)


def get_rows(path_csv_file: str) -> list[list]:
    """
    Args:
        path_csv_file (str): path to one of the csv files in the folder silver_dataset/splits of the chest-imagenome-dataset

    Returns:
        new_rows (list[list]): inner list contains information about a single bbox:
            - subject_id
            - study_id
            - image_id
            - file path to image in mimic-cxr-jpg dataset on workstation
            - bbox_name
            - bbox coordinates
            - phrases describing region inside bbox (if those phrases exist, else None)
            - is_abnormal, boolean variable specifying if region inside bbox is normal or abnormal
    """
    new_rows = []
    index = 0

    total_num_rows = get_total_num_rows(path_csv_file)

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
            if image_id in IMAGE_IDS_TO_IGNORE:
                continue

            # image_file_path is of the form "files/p10/p10000980/s50985099/6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d.dcm"
            # i.e. f"files/p../p{subject_id}/s{study_id}/{image_id}.dcm"
            # since we have the MIMIC-CXR-JPG dataset, we need to replace .dcm by .jpg
            image_file_path = row[4].replace(".dcm", ".jpg")
            mimic_image_file_path = os.path.join(path_to_mimic_cxr, image_file_path)

            if not os.path.exists(mimic_image_file_path):
                # print("Does not exist: ", image_file_path)
                continue

            chest_imagenome_scene_graph_file_path = os.path.join(path_to_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"

            with open(chest_imagenome_scene_graph_file_path) as fp:
                image_scene_graph = json.load(fp)

            # get the attributes specified for the specific image in its image_scene_graph
            # the attributes contain (among other things) phrases used in the reference report used to describe different bbox regions and
            # information whether a described bbox region is normal or abnormal
            #
            # anatomical_region_attributes is a dict with bbox_names as keys and lists that contain 2 elements as values. The 2 list elements are:
            # 1. (normalized) phrases, which is a single string that contains the phrases used to describe the region inside the bbox
            # 2. is_abnormal, a boolean that is True if the region inside the bbox is considered abnormal, else False for normal
            anatomical_region_attributes = get_attributes_dict(image_scene_graph)

            image = cv2.imread(mimic_image_file_path, cv2.IMREAD_UNCHANGED)
            height, width = image.shape

            # iterate over all 36 anatomical regions of the given image (note: there are not always 36 regions present for all images)
            for anatomical_region in image_scene_graph["objects"]:
                bbox_name = anatomical_region["bbox_name"]
                x1 = anatomical_region["original_x1"]
                y1 = anatomical_region["original_y1"]
                x2 = anatomical_region["original_x2"]
                y2 = anatomical_region["original_y2"]

                # check if whole bbox is outside of image height and width
                # if so, skip the anatomical region/bbox
                if check_coordinates_out_of_bounds(height, width, x1, y1, x2, y2):
                    continue

                # it is possible that the bbox is only partially inside the image height and width (if e.g. x1 < 0, whereas x2 > 0)
                # to prevent these cases from raising an exception, we set the coordinates to 0 if coordinate < 0, set to width if x-coordinate > width
                # and set to height if y-coordinate > height
                x1 = check_coordinate(x1, width)
                y1 = check_coordinate(y1, height)
                x2 = check_coordinate(x2, width)
                y2 = check_coordinate(y2, height)

                new_row = [index, subject_id, study_id, image_id, mimic_image_file_path, bbox_name, x1, y1, x2, y2]

                # add phrases (describing the region inside bbox) and is_abnormal boolean variable (indicating if region inside bbox is abnormal) to new_row
                # if there is no phrase, then the region inside bbox is normal and the new_row is extended with None for phrases and False for is_abnormal
                new_row.extend(anatomical_region_attributes.get(bbox_name, [None, False]))
                new_rows.append(new_row)

                index += 1

                if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES and index >= NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES:
                    return new_rows

    return new_rows


def create_new_csv_file(dataset: str, path_csv_file: str) -> None:
    log.info(f"Creating new {dataset}.csv file...")

    # get rows to create new csv_file
    # new_rows is a list of lists, where an inner list specifies all information about a single bbox of a single image
    new_rows = get_rows(path_csv_file)

    # write those rows into a new csv file
    write_rows_in_new_csv_file(dataset, new_rows)

    log.info(f"Creating new {dataset}.csv file... DONE!")


def create_new_csv_files(csv_files_dict):
    if os.path.exists(path_to_chest_imagenome_customized):
        log.error(f"Customized chest imagenome dataset folder already exists at {path_to_chest_imagenome_customized}.")
        log.error("Delete dataset folder before running script to create new folder!")
        return None

    os.mkdir(path_to_chest_imagenome_customized)
    for dataset, path_csv_file in csv_files_dict.items():
        create_new_csv_file(dataset, path_csv_file)


def get_train_val_test_csv_files():
    """Return a dict with datasets as keys and paths to the corresponding csv files in the chest-imagenome dataset as values"""
    path_to_splits_folder = os.path.join(path_to_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def main():
    csv_files_dict = get_train_val_test_csv_files()
    create_new_csv_files(csv_files_dict)


if __name__ == "__main__":
    main()
