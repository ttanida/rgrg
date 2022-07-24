"""
Script creates train.csv, valid.csv and test.csv for object detection (i.e. without phrases).
"""
import csv
import json
import logging
import os

from tqdm import tqdm

from src.dataset_bounding_boxes.constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE

path_to_object_detector_dataset = "/u/home/tanida/datasets/object-detector-dataset"
path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_to_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr-jpg"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# constant specifies how many rows to create in the customized csv files
# if constant is None, then all possible rows are created
NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES = 50


def write_rows_in_new_csv_file(dataset: str, new_rows: list[list]) -> None:
    log.info(f"Writing rows into new {dataset}.csv file...")

    new_csv_file_path = os.path.join(path_to_object_detector_dataset, dataset)
    new_csv_file_path += ".csv" if not NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES else f"-{NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES}.csv"

    with open(new_csv_file_path, "w") as fp:
        csv_writer = csv.writer(fp)

        header = ["index", "subject_id", "study_id", "image_id", "mimic_image_file_path", "bbox_coordinates", "labels"]

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


def get_total_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        return sum(1 for row in csv_reader)


def get_rows(path_csv_file: str, image_ids_to_avoid: set) -> list[list]:
    """
    Args:
        path_csv_file (str): path to one of the csv files in the folder silver_dataset/splits of the chest-imagenome-dataset

    Returns:
        new_rows (list[list]): inner list contains information about a single image:
            - subject_id
            - study_id
            - image_id
            - file path to image in mimic-cxr-jpg dataset on workstation
            - bbox coordinates as a list of lists, where each inner list contains 4 bbox coordinates
            - labels as a list, with class labels for each ground-truth box
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
            # all images in set image_ids_to_avoid are image IDs for images in the gold standard dataset,
            # which should all be excluded from model training and validation
            if image_id in IMAGE_IDS_TO_IGNORE or image_id in image_ids_to_avoid:
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

            # we assume that the images were already padded and resized to 224x224 (see custom_object_detector_dataset.py)
            width, height = 224, 224

            new_image_row = [index, subject_id, study_id, image_id, mimic_image_file_path]
            bbox_coordinates = []
            labels = []

            # iterate over all 36 anatomical regions of the given image (note: there are not always 36 regions present for all images)
            for anatomical_region in image_scene_graph["objects"]:
                bbox_name = anatomical_region["bbox_name"]

                # use the coordinates for a padded and resized 224x224 CXR image (e.g. "x1" instead of "original_x1")
                x1 = anatomical_region["x1"]
                y1 = anatomical_region["y1"]
                x2 = anatomical_region["x2"]
                y2 = anatomical_region["y2"]

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

                bbox_coords = [x1, y1, x2, y2]

                # since background has class label 0 for object detection, shift the remaining class labels by 1
                class_label = ANATOMICAL_REGIONS[bbox_name] + 1

                bbox_coordinates.append(bbox_coords)
                labels.append(class_label)

            # store the lists as strings (excel cannot handle lists)
            new_image_row.extend([str(bbox_coordinates), str(labels)])

            new_rows.append(new_image_row)
            index += 1

            if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES and index >= NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES:
                return new_rows

    return new_rows


def create_new_csv_file(dataset: str, path_csv_file: str, image_ids_to_avoid: set) -> None:
    log.info(f"Creating new {dataset}.csv file...")

    # get rows to create new csv_file
    # new_rows is a list of lists, where an inner list specifies all information about a single image (i.e. bounding boxes + labels)
    new_rows = get_rows(path_csv_file, image_ids_to_avoid)

    # write those rows into a new csv file
    write_rows_in_new_csv_file(dataset, new_rows)

    log.info(f"Creating new {dataset}.csv file... DONE!")


def create_new_dataframes(csv_files_dict, image_ids_to_avoid):
    if os.path.exists(path_to_object_detector_dataset):
        log.error(f"Customized chest imagenome dataset folder already exists at {path_to_object_detector_dataset}.")
        log.error("Delete dataset folder before running script to create new folder!")
        return None

    os.mkdir(path_to_object_detector_dataset)
    for dataset, path_csv_file in csv_files_dict.items():
        create_new_csv_file(dataset, path_csv_file, image_ids_to_avoid)


def get_images_to_avoid():
    path_to_images_to_avoid = os.path.join(path_to_chest_imagenome, "silver_dataset", "splits", "images_to_avoid.csv")

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
    path_to_splits_folder = os.path.join(path_to_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def main():
    csv_files_dict = get_train_val_test_csv_files()

    # the "splits" directory of chest-imagenome contains a csv file called "images_to_avoid.csv",
    # which contains image IDs for images in the gold standard dataset, which should all be excluded
    # from model training and validation
    image_ids_to_avoid = get_images_to_avoid()

    create_new_dataframes(csv_files_dict, image_ids_to_avoid)


if __name__ == "__main__":
    main()
