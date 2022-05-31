import csv
import json
import logging
import os

from tqdm import tqdm

path_to_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized"
path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_to_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr-jpg"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)


def determine_if_abnormal(attributes_list):
    for attributes in attributes_list:
        for attribute in attributes:
            # if "nlp|yes|abnormal" in attribute or \
            #    "anatomicalfinding|yes" in attribute or \
            #    "disease|yes" in attribute:
            #     return True

            if "technicalassessment" in attribute:
                print(attributes_list)
                return True
    
    # no abnormality could be detected
    return False


def normalize_text(phrases: list[str]) -> str:
    """
    Apply:

    - unicode normalization
    - stripping accents
    - lowercasing
    - removing control characters (e.g. '\n')
    - normalizing whitespace characters (i.e. replacing all whitespaces like tabs by the default whitespace)
    and removing redundant whitespaces
    - removing or normalizing special characters

    Args:
        phrases (list[str]): in the attribute dictionary, phrases is originally a list of strings

    Returns:
        str: a single normalized string, with the list of strings concatenated
    """
    return " ".join(phrases).lower().replace("\n", "")


def get_attributes_dict(image_scene_graph):
    attributes_dict = {}
    for attribute in image_scene_graph["attributes"]:
        bbox_name = attribute["bbox_name"]

        phrases = normalize_text(attribute["phrases"])

        is_abnormal = determine_if_abnormal(attribute["attributes"])
        if is_abnormal:
            print(phrases)
            print()

        attributes_dict[bbox_name] = [phrases, is_abnormal]

    return attributes_dict


def get_rows(path_csv_file):
    new_rows = []

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # iterate over all rows of the given csv file
        for index, row in enumerate(tqdm(csv_reader)):
            subject_id = row[1]
            study_id = row[2]
            image_id = row[3]
            # image_file_path is of the form 'files/p10/p10000980/s50985099/6ad03ed1-97ee17ee-9cf8b320-f7011003-cd93b42d.dcm'
            # i.e. 'files/p../subject_id/study_id/image_id.dcm'
            # since we have the MIMIC-CXR-JPG dataset, we need to replace .dcm by .jpg
            image_file_path = row[4].replace(".dcm", ".jpg")
            mimic_image_file_path = os.path.join(path_to_mimic_cxr, image_file_path)

            chest_imagenome_scene_graph_file_path = os.path.join(path_to_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"
            with open(chest_imagenome_scene_graph_file_path) as fp:
                image_scene_graph = json.load(fp)

            # get a dict with bbox_names as keys and lists that contain 2 elements as values. The 2 list elements are:
            # 1. phrases, which is a single string that contains the phrases used to describe the region inside the bbox
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


def create_new_csv_file(dataset, path_csv_file):
    log.info(f"Creating {dataset}.csv file.")

    # get rows to create new csv_file
    # new_rows is a list of lists, where an inner list specifies all attributes of a single bbox of a single image
    new_rows = get_rows(path_csv_file)


def create_new_csv_files(csv_files_dict):
    if os.path.exists(path_to_chest_imagenome_customized):
        log.error(f"Customized chest imagenome dataset already exists at {path_to_chest_imagenome_customized}.")
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
