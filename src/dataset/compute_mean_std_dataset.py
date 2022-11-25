import os

import cv2
import numpy as np
from tqdm import tqdm

from src.path_datasets_and_weights import path_mimic_cxr_jpg

TOL = 1e-4
COUNTER_PATIENCE = 50


def last_and_curr_mean_std_close(curr_mean, curr_std, last_mean_values, last_std_values):
    mean_tol_check = np.abs(curr_mean - last_mean_values) <= TOL
    std_tol_check = np.abs(curr_std - last_std_values) <= TOL
    return np.all([mean_tol_check, std_tol_check])


def get_mean_std(image_paths) -> tuple([float, float]):
    """
    Returns an approximation of the dataset's mean and std.

    Compares the currently computed mean and std values (named curr_mean and curr_std)
    to the last k computed mean and std values, stored in the lists last_mean_values and last_std_values.

    If the absolute difference between each value of last_mean_values and curr_mean as well as last_std_values and curr_std
    is smaller than a specified tolerance (named TOL) for k consecutive iterations, then the loop through all image_paths breaks
    and the currently computed mean and std values are returned

    The k variable is specified by the COUNTER_PATIENCE constant.
    """
    last_mean_values = []
    last_std_values = []
    mean = 0.0
    std = 0.0
    counter = 0

    for num_image, image_path in enumerate(image_paths, start=1):
        # image is a np array of shape (h, w) with pixel (integer) values between [0, 255]
        # note that there is no channel dimension, because images are grayscales and cv2.IMREAD_UNCHANGED is specified
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # pixel values have to be normalized to between [0.0, 1.0], since we need mean and std values in the range [0.0, 1.0]
        # this is because the transforms.Normalize class applies normalization by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        # with max_pixel_value=255.0
        image = image / 255.

        mean += image.mean()
        std += image.std()

        curr_mean = mean / num_image
        curr_std = std / num_image

        print(f"current mean: {curr_mean:.3f} \tcurrent std: {curr_std:.3f} \tcounter: {counter}")

        if last_and_curr_mean_std_close(curr_mean, curr_std, last_mean_values, last_std_values):
            counter += 1
        else:
            last_mean_values = []
            last_std_values = []
            counter = 0

        if counter >= COUNTER_PATIENCE:
            break

        last_mean_values.append(curr_mean)
        last_std_values.append(curr_std)

    return mean / num_image, std / num_image


def get_image_paths_mimic() -> list:
    """
    Returns a list of all file paths to mimic-cxr images.
    """
    print("Reading in the file paths to all MIMIC-CXR images")
    image_paths = []
    path_mimic_cxr_files = os.path.join(path_mimic_cxr_jpg, "files")
    for root, _, files in tqdm(os.walk(path_mimic_cxr_files)):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            if image_path.endswith(".jpg"):
                image_paths.append(image_path)

    return image_paths


def main():
    image_paths = get_image_paths_mimic()
    mean, std = get_mean_std(image_paths)
    print()
    print(f"mean: {mean:.3f}")
    print(f"std: {std:.3f}")


if __name__ == "__main__":
    main()
