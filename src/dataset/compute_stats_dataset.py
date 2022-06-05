from collections import defaultdict
import csv
import json
import logging
import os

from tqdm import tqdm

from constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE

path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)


def print_stats(dataset, stats):
    num_images, num_ignored_images, num_bboxes, num_bboxes_with_phrases, num_outlier_bboxes, bbox_with_phrases_counter_dict, outlier_bbox_counter_dict = stats

    print(f"\n{dataset}:")
    print(f"\t{num_images:,} images in total")
    print(f"\t{num_ignored_images} images were ignored (due to faulty x-rays etc.)")

    print(f"\n\t{num_bboxes:,} bboxes in total")
    print(f"\t{num_bboxes_with_phrases:,} bboxes have corresponding phrases")
    print(f"\t-> {(num_bboxes_with_phrases/num_bboxes) * 100:.2f}% of bboxes have corresponding phrases")

    print(f"\n\t{num_outlier_bboxes:,} bboxes with phrases have 'outlier' anatomical region names")
    print(f"\t-> {(num_outlier_bboxes/num_bboxes_with_phrases) * 100:.2f}% of bboxes with phrases have 'outlier' names")

    print("\n\tCounts and percentages of 'outlier' bboxes:")
    total_count = sum(value for value in outlier_bbox_counter_dict.values())
    for bbox_name, count in sorted(outlier_bbox_counter_dict.items(), key=lambda k_v: k_v[1], reverse=True):
        print(f"\t\t{bbox_name}: {count:,} ({(count/total_count) * 100:.2f}%)")

    print("\n\tCounts and percentages of normal bboxes with phrases:")
    total_count = sum(value for value in bbox_with_phrases_counter_dict.values())
    for bbox_name, count in sorted(bbox_with_phrases_counter_dict.items(), key=lambda k_v: k_v[1], reverse=True):
        print(f"\t\t{bbox_name}: {count:,} ({(count/total_count) * 100:.2f}%)")
    print()


def get_stats_image(image_scene_graph: dict, num_outlier_bboxes: int, num_bboxes_with_phrases: int, bbox_counter_dict: dict, outlier_bbox_counter_dict: dict) -> tuple[int, int]:
    for attribute in image_scene_graph["attributes"]:
        bbox_name = attribute["bbox_name"]
        num_bboxes_with_phrases += 1

        # bbox_names such as "left chest wall" or "right breast" don't appear in the 36 anatomical regions that have bbox coordiantes
        # they are considered outliers
        if bbox_name not in ANATOMICAL_REGIONS:
            num_outlier_bboxes += 1
            outlier_bbox_counter_dict[bbox_name] += 1
        else:
            bbox_counter_dict[bbox_name] += 1

    return num_outlier_bboxes, num_bboxes_with_phrases


def get_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        return sum(1 for row in csv_reader)


def compute_stats_for_csv_file(dataset, path_csv_file):
    num_images = get_num_rows(path_csv_file)
    num_ignored_images = 0  # images that are ignored because they are faulty (due to faulty x-rays etc.)

    num_bboxes = 0
    num_bboxes_with_phrases = 0
    num_outlier_bboxes = 0  # bboxes that have bbox names (like 'left breast' etc.) that are not in the 36 anatomical regions are considered outliers

    bbox_with_phrases_counter_dict = defaultdict(int)  # dict to count how often each of the 36 anatomical regions have phrases
    outlier_bbox_counter_dict = defaultdict(int) # dict to count how often each of the outlier regions have phrases

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # iterate over all rows of the given csv file (i.e. over all images)
        for row in tqdm(csv_reader, total=num_images):
            image_id = row[3]

            # all images in set IMAGE_IDS_TO_IGNORE seem to be failed x-rays and thus have to be discarded
            # (they also don't have corresponding scene graph json files anyway)
            if image_id in IMAGE_IDS_TO_IGNORE:
                num_ignored_images += 1
                continue

            chest_imagenome_scene_graph_file_path = os.path.join(path_to_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"

            with open(chest_imagenome_scene_graph_file_path) as fp:
                image_scene_graph = json.load(fp)

            num_outlier_bboxes, num_bboxes_with_phrases = get_stats_image(image_scene_graph, num_outlier_bboxes, num_bboxes_with_phrases, bbox_with_phrases_counter_dict, outlier_bbox_counter_dict)

            # for each image, there are 36 bboxes for 36 anatomical regions
            num_bboxes += 36

    stats = num_images, num_ignored_images, num_bboxes, num_bboxes_with_phrases, num_outlier_bboxes, bbox_with_phrases_counter_dict, outlier_bbox_counter_dict
    print_stats(dataset, stats)

    return stats


def compute_and_print_stats_for_csv_files(csv_files_dict):
    total_num_images = 0
    total_num_ignored_images = 0

    total_num_bboxes = 0
    total_num_bboxes_with_phrases = 0
    total_num_outlier_bboxes = 0

    total_bbox_with_phrases_counter_dict = defaultdict(int)
    total_outlier_bbox_counter_dict = defaultdict(int)

    for dataset, path_csv_file in csv_files_dict.items():
        stats = compute_stats_for_csv_file(dataset, path_csv_file)
        num_images, num_ignored_images, num_bboxes, num_bboxes_with_phrases, num_outlier_bboxes, bbox_with_phrases_counter_dict, outlier_bbox_counter_dict = stats

        total_num_images += num_images
        total_num_ignored_images += num_ignored_images
        total_num_bboxes += num_bboxes
        total_num_bboxes_with_phrases += num_bboxes_with_phrases
        total_num_outlier_bboxes += num_outlier_bboxes

        for key, value in bbox_with_phrases_counter_dict.items():
            total_bbox_with_phrases_counter_dict[key] += value

        for key, value in outlier_bbox_counter_dict.items():
            total_outlier_bbox_counter_dict[key] += value

    stats = total_num_images, total_num_ignored_images, total_num_bboxes, total_num_bboxes_with_phrases, total_num_outlier_bboxes, total_bbox_with_phrases_counter_dict, total_outlier_bbox_counter_dict
    print_stats("Total", stats)


def get_train_val_test_csv_files():
    path_to_splits_folder = os.path.join(path_to_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def main():
    csv_files_dict = get_train_val_test_csv_files()
    compute_and_print_stats_for_csv_files(csv_files_dict)


if __name__ == "__main__":
    main()
