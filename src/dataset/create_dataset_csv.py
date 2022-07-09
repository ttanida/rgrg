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
import re

import imagesize
from tqdm import tqdm

from constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE, SUBSTRINGS_TO_REMOVE

path_to_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"
path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_to_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr-jpg"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# constant specifies how many rows to create in the customized csv files
# if constant is None, then all possible rows are created
NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES = None

# boolean variable to determine if customized csv files should have an additional column that specifies if a phrase describing
# a region/bbox states that there is a finding (e.g. “There is pneumothorax.”) or no finding (e.g. “There is no pleural effusion.”)
CREATE_FINDINGS_COLUMN = True


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


def determine_if_finding_exist(attributes_list: list[list]) -> bool:
    """
    attributes_list is a list of lists that contains attributes corresponding to the phrases describing a specific bbox.

    E.g. the phrases: ['Right lung is clear without pneumothorax.', 'No pneumothorax identified.'] have the attributes_list
    [['anatomicalfinding|no|lung opacity', 'anatomicalfinding|no|pneumothorax', 'nlp|yes|normal'], ['anatomicalfinding|no|pneumothorax']],
    where the 1st inner list contains the attributes pertaining to the 1st phrase, and the 2nd inner list contains attributes for the 2nd phrase respectively.

    Phrases where a finding is specified as existing have the attribute 'anatomicalfinding|yes'.
    """
    for attributes in attributes_list:
        for attribute in attributes:
            if attribute.startswith("anatomicalfinding|yes"):
                return True

    # there was no finding
    return False


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


def convert_phrases_to_single_string(phrases: list[str]) -> str:
    """
    Takes a list of phrases describing the region of a single bbox and returns a single string.

    Also performs operations to clean the single string, such as:
        - removes irrelevant substrings (like "PORTABLE UPRIGHT AP VIEW OF THE CHEST:")
        - removes whitespace characters (e.g. \n or \t) and redundant whitespaces
        - removes duplicate sentences

    Args:
        phrases (list[str]): in the attribute dictionary, phrases is originally a list of strings

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
        phrases = re.sub(SUBSTRINGS_TO_REMOVE, '', phrases, flags=re.DOTALL)

        return phrases

    def remove_whitespace(phrases):
        """Remove white space and capitalize words that come after a period."""
        # new_phrases collects all words
        new_phrases = ""

        # set the previous word as a period, such that the first word in new_phrases is capitalized
        prev_word = "."

        for word in phrases.split():
            if prev_word[-1] == ".":
                new_phrases += (word[0].upper() + word[1:])  # capitalize the word
            else:
                new_phrases += word

            # add a space for the next word
            new_phrases += " "

            # set current word as previous word for the next word
            prev_word = word

        # remove the trailing whitespace
        return new_phrases.rstrip()

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

    phrases = remove_duplicate_sentences(phrases)

    return phrases


def get_attributes_dict(image_scene_graph: dict) -> dict[list]:
    attributes_dict = {}
    for attribute in image_scene_graph["attributes"]:
        bbox_name = attribute["bbox_name"]

        # ignore bbox_names such as "left chest wall" or "right breast" that don't appear in the 36 anatomical regions that have bbox coordiantes
        if bbox_name not in ANATOMICAL_REGIONS:
            continue

        phrases = convert_phrases_to_single_string(attribute["phrases"])
        is_abnormal = determine_if_abnormal(attribute["attributes"])

        attributes_dict[bbox_name] = [phrases, is_abnormal]

        if CREATE_FINDINGS_COLUMN:
            finding_exist = determine_if_finding_exist(attribute["attributes"])
            attributes_dict[bbox_name].append(finding_exist)

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

            width, height = imagesize.get(mimic_image_file_path)

            # iterate over all 36 anatomical regions of the given image (note: there are not always 36 regions present for all images)
            for anatomical_region in image_scene_graph["objects"]:
                bbox_name = anatomical_region["bbox_name"]
                x1 = anatomical_region["original_x1"]
                y1 = anatomical_region["original_y1"]
                x2 = anatomical_region["original_x2"]
                y2 = anatomical_region["original_y2"]

                # check if bbox coordinates are faulty
                # if so, skip the anatomical region/bbox
                if coordinates_faulty(height, width, x1, y1, x2, y2):
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
                # if there is no phrase, then the region inside bbox is normal and the new_row is extended with "" for phrases (empty phrase) and False for is_abnormal

                # if CREATE_FINDINGS_COLUMN == True, then also add finding_exist boolean variable (indicating if phrase describing region specifies that a finding was found or not)
                # if there is no phrase, then set finding_exist variable to False
                if not CREATE_FINDINGS_COLUMN:
                    new_row.extend(anatomical_region_attributes.get(bbox_name, ["", False]))
                else:
                    new_row.extend(anatomical_region_attributes.get(bbox_name, ["", False, False]))

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
