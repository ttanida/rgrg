import csv
import logging
import os

import cv2
from tqdm import tqdm

from src.dataset.constants import IMAGE_IDS_TO_IGNORE

path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_to_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr-jpg"
path_to_missing_images_file = "/u/home/tanida/region-guided-chest-x-ray-report-generation/src/missing_images"
path_to_corrupted_images_file = "/u/home/tanida/region-guided-chest-x-ray-report-generation/src/corrupted_images"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)


def get_total_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        return sum(1 for row in csv_reader)


def find_images(path_csv_file, image_ids_to_avoid):
    total_num_rows = get_total_num_rows(path_csv_file)

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # iterate over all rows of the given csv file (i.e. over all images), if NUM_ROWS_TO_CREATE_IN_NEW_CSV_FILES is not set to a specific value
        for row in tqdm(csv_reader, total=total_num_rows):
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
                log.info(f"Image does not exist: {mimic_image_file_path}")
                with open(path_to_missing_images_file, "a") as f:
                    f.write(f"{mimic_image_file_path}\n")
                    continue

            try:
                _ = cv2.imread(mimic_image_file_path, cv2.IMREAD_UNCHANGED)
            except Exception:
                log.info(f"Image is corrupted: {mimic_image_file_path}")
                with open(path_to_corrupted_images_file, "a") as f:
                    f.write(f"{mimic_image_file_path}\n")
                    continue


def find_missing_images(csv_files_dict, image_ids_to_avoid):
    for _, path_csv_file in csv_files_dict.items():
        find_images(path_csv_file, image_ids_to_avoid)


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

    image_ids_to_avoid = get_images_to_avoid()

    find_missing_images(csv_files_dict, image_ids_to_avoid)


if __name__ == "__main__":
    main()
