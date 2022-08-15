import csv
import json
import logging
import os

import imagesize
from tqdm import tqdm

from constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE

path_to_normality_pool_folder = "/u/home/tanida/datasets/normality-pool"
path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_to_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr-jpg"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# normality pool size per region
NORMALITY_POOL_SIZE = 5


def write_rows_in_new_csv_file(new_rows: dict[list[list]]) -> None:
    log.info("Writing csv file...")
    new_csv_file_path = os.path.join(path_to_normality_pool_folder, f"normality-pool-{NORMALITY_POOL_SIZE}.csv")

    with open(new_csv_file_path, "w") as fp:
        csv_writer = csv.writer(fp)

        header = ["subject_id", "study_id", "image_id", "mimic_image_file_path", "bbox_name", "x1", "y1", "x2", "y2"]

        csv_writer.writerow(header)

        for rows in new_rows.values():
            csv_writer.writerows(rows)

    log.info("DONE!")


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


def get_attributes_dict(image_scene_graph: dict) -> dict[bool]:
    attributes_dict = {}
    for attribute in image_scene_graph["attributes"]:
        bbox_name = attribute["bbox_name"]

        # ignore bbox_names such as "left chest wall" or "right breast" that don't appear in the 36 anatomical regions that have bbox coordiantes
        if bbox_name not in ANATOMICAL_REGIONS:
            continue

        is_abnormal = determine_if_abnormal(attribute["attributes"])

        attributes_dict[bbox_name] = is_abnormal

    return attributes_dict


def get_rows(path_csv_file: str, image_ids_to_avoid: set) -> list[list]:
    new_rows = {region: [] for region in ANATOMICAL_REGIONS}

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        for row in tqdm(csv_reader):
            # check if all regions have NORMALITY_POOL_SIZE rows already
            for rows in new_rows.values():
                if len(rows) < NORMALITY_POOL_SIZE:
                    break
                else:
                    return new_rows

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

            anatomical_region_attributes = get_attributes_dict(image_scene_graph)

            width, height = imagesize.get(mimic_image_file_path)

            # iterate over all 36 anatomical regions of the given image (note: there are not always 36 regions present for all images)
            for anatomical_region in image_scene_graph["objects"]:
                bbox_name = anatomical_region["bbox_name"]

                # ignore bbox_names such as "left chest wall" or "right breast" that don't appear in the 36 anatomical regions that have bbox coordiantes
                if bbox_name not in ANATOMICAL_REGIONS:
                    continue

                # if the region already has NORMALITY_POOL_SIZE bboxes in the new_rows dict, then don't add more
                if len(new_rows[bbox_name]) >= NORMALITY_POOL_SIZE:
                    continue

                # ignore bboxes that are abnormal (if a bbox/region did not have a key, then it means it was normal, i.e. False)
                if anatomical_region_attributes.get(bbox_name, False):
                    continue

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

                new_row = [subject_id, study_id, image_id, mimic_image_file_path, bbox_name, x1, y1, x2, y2]

                new_rows[bbox_name].append(new_row)

    return new_rows


def create_normality_pool_csv_file(path_train_csv: str, image_ids_to_avoid: set) -> None:
    # get rows to create normality pool csv file
    new_rows = get_rows(path_train_csv, image_ids_to_avoid)

    # write those rows into the csv file
    write_rows_in_new_csv_file(new_rows)


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


def main():
    path_to_splits_folder = os.path.join(path_to_chest_imagenome, "silver_dataset", "splits")
    path_train_csv = os.path.join(path_to_splits_folder, "train.csv")

    # the "splits" directory of chest-imagenome contains a csv file called "images_to_avoid.csv",
    # which contains image IDs for images in the gold standard dataset, which should all be excluded
    # from model training and validation
    image_ids_to_avoid = get_images_to_avoid()

    create_normality_pool_csv_file(path_train_csv, image_ids_to_avoid)


if __name__ == "__main__":
    main()
