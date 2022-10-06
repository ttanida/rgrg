"""
Chest ImaGenome dataset path should have a (sub-)directory called "silver_dataset" in its directory.
MIMIC-CXR and MIMIC-CXR-JPG dataset paths should both have a (sub-)directory called "files" in their directories.

Note that we only need the report txt files from MIMIC-CXR, which are in the file mimic-cxr-report.zip at
https://physionet.org/content/mimic-cxr/2.0.0/.

path_full_dataset specifies the path where the folder will be created (by module src/dataset/create_dataset.py) that will hold the
train, valid, test and test-2 csv files, which will be used for training, evaluation and testing. See doc string of create_dataset.py for more information.

path_chexbert_weights specifies the path to the weights of the CheXbert labeler needed to extract the disease labels from the generated and reference reports.
The weights can be downloaded here: https://github.com/stanfordmlgroup/CheXbert#checkpoint-download
"""

path_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr"
path_mimic_cxr_jpg = "/u/home/tanida/datasets/mimic-cxr-jpg"
path_full_dataset = "/u/home/tanida/datasets/dataset-with-chexbert"
path_chexbert_weights = "/u/home/tanida/region-guided-chest-x-ray-report-generation/src/CheXbert/src/models/chexbert.pth"
