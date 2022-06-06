from collections import defaultdict
import csv
import json
import os

from tqdm import tqdm

from constants import ANATOMICAL_REGIONS, IMAGE_IDS_TO_IGNORE

path_to_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"


def print_stats_counter_dicts(counter_dict):
    """Print the counts in descending order"""
    total_count = sum(value for value in counter_dict.values())
    for bbox_name, count in sorted(counter_dict.items(), key=lambda k_v: k_v[1], reverse=True):
        print(f"\t\t{bbox_name}: {count:,} ({(count/total_count) * 100:.2f}%)")


def print_stats(dataset: str, stats: dict) -> None:
    if dataset != "Total":
        num_images = stats["num_images"]
        num_ignored_images = stats["num_ignored_images"]
        num_bboxes = stats["num_bboxes"]
        num_bboxes_with_phrases = stats["num_bboxes_with_phrases"]
        num_outlier_bboxes = stats["num_outlier_bboxes"]
        bbox_with_phrases_counter_dict = stats["bbox_with_phrases_counter_dict"]
        outlier_bbox_counter_dict = stats["outlier_bbox_counter_dict"]
    else:
        num_images = stats["total_num_images"]
        num_ignored_images = stats["total_num_ignored_images"]
        num_bboxes = stats["total_num_bboxes"]
        num_bboxes_with_phrases = stats["total_num_bboxes_with_phrases"]
        num_outlier_bboxes = stats["total_num_outlier_bboxes"]
        bbox_with_phrases_counter_dict = stats["total_bbox_with_phrases_counter_dict"]
        outlier_bbox_counter_dict = stats["total_outlier_bbox_counter_dict"]

    print(f"\n{dataset}:")
    print(f"\t{num_images:,} images in total")
    print(f"\t{num_ignored_images} images were ignored (due to faulty x-rays etc.)")

    print(f"\n\t{num_bboxes:,} bboxes in total")
    print(f"\t{num_bboxes_with_phrases:,} bboxes have corresponding phrases")
    print(f"\t-> {(num_bboxes_with_phrases/num_bboxes) * 100:.2f}% of bboxes have corresponding phrases")

    print(f"\n\t{num_outlier_bboxes:,} bboxes with phrases have 'outlier' anatomical region names")
    print(f"\t-> {(num_outlier_bboxes/num_bboxes_with_phrases) * 100:.2f}% of bboxes with phrases have 'outlier' names")

    print("\n\tCounts and percentages of 'outlier' bboxes:")
    print_stats_counter_dicts(outlier_bbox_counter_dict)

    print("\n\tCounts and percentages of normal bboxes with phrases:")
    print_stats_counter_dicts(bbox_with_phrases_counter_dict)

    print()


def update_stats_for_image(image_scene_graph: dict, stats: dict) -> None:
    for attribute in image_scene_graph["attributes"]:
        bbox_name = attribute["bbox_name"]
        stats["num_bboxes_with_phrases"] += 1

        # bbox_names such as "left chest wall" or "right breast" don't appear in the 36 anatomical regions that have bbox coordiantes
        # they are considered outliers
        if bbox_name not in ANATOMICAL_REGIONS:
            stats["num_outlier_bboxes"] += 1
            stats["outlier_bbox_counter_dict"][bbox_name] += 1
        else:
            stats["bbox_with_phrases_counter_dict"][bbox_name] += 1


def get_num_rows(path_csv_file: str) -> int:
    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        return sum(1 for row in csv_reader)


def compute_stats_for_csv_file(dataset: str, path_csv_file: str) -> dict:
    stats = {stat: 0 for stat in ["num_images", "num_ignored_images", "num_bboxes", "num_bboxes_with_phrases", "num_outlier_bboxes"]}
    stats["bbox_with_phrases_counter_dict"] = defaultdict(int)
    stats["outlier_bbox_counter_dict"] = defaultdict(int)

    stats["num_images"] += get_num_rows(path_csv_file)

    with open(path_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        # iterate over all rows of the given csv file (i.e. over all images)
        for row in tqdm(csv_reader, total=stats["num_images"]):
            image_id = row[3]

            # all images in set IMAGE_IDS_TO_IGNORE seem to be failed x-rays and thus have to be discarded
            # (they also don't have corresponding scene graph json files anyway)
            if image_id in IMAGE_IDS_TO_IGNORE:
                stats["num_ignored_images"] += 1
                continue

            chest_imagenome_scene_graph_file_path = os.path.join(path_to_chest_imagenome, "silver_dataset", "scene_graph", image_id) + "_SceneGraph.json"

            with open(chest_imagenome_scene_graph_file_path) as fp:
                image_scene_graph = json.load(fp)

            # update num_bboxes_with_phrases and num_outlier_bboxes based on current image
            # also update the bbox_with_phrases and outlier_bbox counter dicts
            update_stats_for_image(image_scene_graph, stats)

            # for each image, there are normally 36 bboxes for 36 anatomical regions, but there may be less occasionally
            for anatomical_region in image_scene_graph["objects"]:
                stats["num_bboxes"] += 1

    print_stats(dataset=dataset, stats=stats)

    return stats


def compute_and_print_stats_for_csv_files(csv_files_dict):
    """
    total_num_ignored_images: images that are ignored because of failed x-rays
    total_num_outlier_bboxes: bboxes that have bbox names (like 'left breast' etc.) that are not in the 36 anatomical regions are considered outliers
    total_bbox_with_phrases_counter_dict: dict to count how often each of the 36 anatomical regions have phrases
    total_outlier_bbox_counter_dict: dict to count how often each of the outlier regions have phrases
    """
    total_stats = {stat: 0 for stat in ["total_num_images", "total_num_ignored_images", "total_num_bboxes", "total_num_bboxes_with_phrases", "total_num_outlier_bboxes"]}
    total_stats["total_bbox_with_phrases_counter_dict"] = defaultdict(int)
    total_stats["total_outlier_bbox_counter_dict"] = defaultdict(int)

    for dataset, path_csv_file in csv_files_dict.items():
        stats = compute_stats_for_csv_file(dataset, path_csv_file)

        for key, value in stats.items():
            if key not in ["bbox_with_phrases_counter_dict", "outlier_bbox_counter_dict"]:
                total_stats["total_" + key] += value
            else:
                for bbox_name, count in value.items():  # value is a counter dict in this case
                    total_stats["total_" + key][bbox_name] += count

    print_stats(dataset="Total", stats=total_stats)


def get_train_val_test_csv_files():
    path_to_splits_folder = os.path.join(path_to_chest_imagenome, "silver_dataset", "splits")
    return {dataset: os.path.join(path_to_splits_folder, dataset) + ".csv" for dataset in ["train", "valid", "test"]}


def main():
    csv_files_dict = get_train_val_test_csv_files()
    compute_and_print_stats_for_csv_files(csv_files_dict)


if __name__ == "__main__":
    main()
