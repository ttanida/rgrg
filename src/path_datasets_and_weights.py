"""
Chest ImaGenome dataset path should have a (sub-)directory called "silver_dataset" in its directory.
MIMIC-CXR and MIMIC-CXR-JPG dataset paths should both have a (sub-)directory called "files" in their directories.

Note that we only need the report txt files from MIMIC-CXR, which can be downloaded via the command:
"wget -r -N -c -np -A txt --user your_user_name --ask-password https://physionet.org/files/mimic-cxr/2.0.0/",
which specifies that only txt files are to be downloaded.

path_full_dataset specifies the path where the folder will be created that will hold the train, valid, test and test-2 csv files,
which will be used for training, evaluation and testing. See doc string of module src/dataset/create_dataset.py for more information.

path_chexbert_weights specifies the path to the weights of the CheXbert label extractor needed to compute the clinical efficacy metric scores.
The weights can be downloaded here: https://github.com/stanfordmlgroup/CheXbert#checkpoint-download
"""

path_chest_imagenome = "/u/home/tanida/datasets/chest-imagenome-dataset"
path_mimic_cxr = "/u/home/tanida/datasets/mimic-cxr"
path_mimic_cxr_jpg = "/u/home/tanida/datasets/mimic-cxr-jpg"
path_full_dataset = "/u/home/tanida/datasets/dataset-full-model-complete-new-method"
path_chexbert_weights = "/u/home/tanida/region-guided-chest-x-ray-report-generation/src/CheXbert/src/models/chexbert.pth"
