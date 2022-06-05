import csv
import json
import logging
import os

from tokenizers import normalizers
from tokenizers.normalizers import NFKC, Lowercase
from tqdm import tqdm

from constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE

path_to_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized"
path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_to_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr-jpg"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)


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


def get_rows(path_csv_file: str) -> list[list]:
    new_rows = []

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # variable used to log which folder (between p10-p19) of MIMIC-CXR is currently being iterated through
        current_folder_name = None

        # iterate over all rows of the given csv file (i.e. over all images)
        for index, row in enumerate(tqdm(csv_reader)):
            if (index + 1) % 50000 == 0:
                print(f"Index: {index + 1}")
                print_memory()

            subject_id = row[1]
            study_id = row[2]
            image_id = row[3]

            # log folder name if it changes
            if subject_id[:2] != current_folder_name:
                current_folder_name = subject_id[:2]
                log.info(f"Current folder: p{current_folder_name}")

            # all images in set IMAGE_IDS_TO_IGNORE seem to be failed x-rays and thus have to be discarded
            # (they also don't have correspoding scene graph json files anyway)
            if image_id in IMAGE_IDS_TO_IGNORE:
                continue

            # image_file_path is of the form "files/p10/p10000980/s50985099/6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d.dcm"
            # i.e. f"files/p../p{subject_id}/s{study_id}/{image_id}.dcm"
            # since we have the MIMIC-CXR-JPG dataset, we need to replace .dcm by .jpg
            image_file_path = row[4].replace(".dcm", ".jpg")
            mimic_image_file_path = os.path.join(path_to_mimic_cxr, image_file_path)

            chest_imagenome_scene_graph_file_path = os.path.join(path_to_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"

            try:
                with open(chest_imagenome_scene_graph_file_path) as fp:
                    image_scene_graph = json.load(fp)
            except:
                print()
                print("Image to ignore:")
                print("\tsubject_id", subject_id)
                print("\tstudy_id", study_id)
                print("\timage_id", image_id)
                print()
                continue

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
                # if they exist, otherwise extend list with [None, None]
                new_row.extend(anatomical_region_attributes.get(bbox_name, [None, None]))
                new_rows.append(new_row)

    return new_rows


def create_new_csv_file(dataset: str, path_csv_file: str):
    log.info(f"Creating new {dataset}.csv file")

    # get rows to create new csv_file
    # new_rows is a list of lists, where an inner list specifies all attributes of a single bbox of a single image
    new_rows = get_rows(path_csv_file)


def create_new_csv_files(csv_files_dict):
    # if os.path.exists(path_to_chest_imagenome_customized):
    #     log.error(f"Customized chest imagenome dataset already exists at {path_to_chest_imagenome_customized}.")
    #     log.error("Delete dataset folder before running script to create new folder!")
    #     return None

    # os.mkdir(path_to_chest_imagenome_customized)
    for dataset, path_csv_file in csv_files_dict.items():
        create_new_csv_file(dataset, path_csv_file)


def get_train_val_test_csv_files():
    path_to_splits_folder = os.path.join(path_to_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def print_memory():
    import psutil

    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / (1024 * 1024 * 1024):.2f} GB")
    print(f"Available: {mem.available / (1024 * 1024 * 1024):.2f} GB")
    print(f"Percentage: {mem.percent}%")
    print(f"Used: {mem.used / (1024 * 1024 * 1024):.2f} GB")
    print(f"Free: {mem.free / (1024 * 1024 * 1024):.2f} GB")


def main():
    print_memory()
    csv_files_dict = get_train_val_test_csv_files()
    create_new_csv_files(csv_files_dict)


if __name__ == "__main__":
    main()
