# Interactive and Explainable Region-guided Radiology Report Generation

## Abstract

The automatic generation of radiology reports has the potential to assist radiologists in the time-consuming task of report writing. Existing methods generate the full report from image-level features, failing to explicitly focus on anatomical regions in the image. We propose a simple yet effective region-guided report generation model that detects anatomical regions and then describes individual, salient regions to form the final report. While previous methods generate reports without the possibility of human intervention and with limited explainability, our method opens up novel clinical use cases through additional human-in-the-loop capabilities and introduces a high degree of transparency and explainability. Comprehensive experiments demonstrate the effectiveness of our method in both the report generation task and the human-in-the-loop capabilities, outperforming previous state-of-the-art models.

## Results

![image info](./figures_repo/nlg_metrics_table.png)
![image info](./figures_repo/clinical_efficacy_metrics_table.png)

## Setup

1. Create conda environment with "**conda env create -f environment.yml**"
2. In src/path_datasets_and_weights.py, specify the paths to the various datasets (Chest ImaGenome, MIMIC-CXR, MIMIC-CXR-JPG), CheXbert weights, and folders in which the runs are saved. Follow the instructions of the doc string of path_datasets_and_weights.py.

## Create train, val and test csv files

After the setup, run "**python create_dataset.py**" in src/dataset/ to create training, val and test csv files, in which each row contains specific information about a single image. See doc string of create_dataset.py for more details.

## Training and Testing

Please read the README_TRAIN_TEST.md for specific information on trainig and testing the model.