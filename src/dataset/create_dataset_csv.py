import csv
import json
import logging
import os

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Lowercase
from tqdm import tqdm

from constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE

path_to_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full-dataset"
path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_to_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr-jpg"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# constant specifies how many rows to create in the customized csv files
# if constant is None, then all possible rows are created (resulting in csv files of huge file sizes)
NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES = 1500


def write_rows_in_new_csv_file(dataset: str, new_rows: list[list]) -> None:
    log.info(f"Writing rows into new {dataset}.csv file...")

    header = ["index", "subject_id", "study_id", "image_id", "mimic_image_file_path", "bbox_name", "x1", "y1", "x2", "y2", "phrases", "is_abnormal"]

    new_csv_file_path = os.path.join(path_to_chest_imagenome_customized, dataset) + ".csv"
    with open(new_csv_file_path, "w") as fp:
        csv_writer = csv.writer(fp)

        csv_writer.writerow(header)
        csv_writer.writerows(new_rows)


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
        str: a single normalized string, with the list of strings concatenated
    """
    # convert list of phrases into a single phrase
    phrases = " ".join(phrases)

    # apply a sequence of normalization operations
    normalizer = normalizers.Sequence([NFKC(), Lowercase()])
    phrases = normalizer.normalize_str(phrases)

    # remove all whitespace characters (multiple whitespaces, newlines, tabs etc.)
    phrases = " ".join(phrases.split())

    return phrases


def get_attributes_dict(image_scene_graph):
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


def get_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        return sum(1 for row in csv_reader)


def get_rows(path_csv_file: str) -> list[list]:
    """_summary_

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

    num_rows = get_num_rows(path_csv_file)

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # iterate over all rows of the given csv file (i.e. over all images), if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is not set to a specific value
        for row in tqdm(csv_reader, total=num_rows):

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

            # iterate over all 36 anatomical regions of the given image
            for anatomical_region in image_scene_graph["objects"]:
                bbox_name = anatomical_region["bbox_name"]
                x1 = anatomical_region["original_x1"]
                y1 = anatomical_region["original_y1"]
                x2 = anatomical_region["original_x2"]
                y2 = anatomical_region["original_y2"]

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
    # new_rows is a list of lists, where an inner list specifies all attributes of a single bbox of a single image
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
    path_to_splits_folder = os.path.join(path_to_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def main():
    csv_files_dict = get_train_val_test_csv_files()
    create_new_csv_files(csv_files_dict)


if __name__ == "__main__":
    main()
