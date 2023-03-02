# Interactive and Explainable Region-guided Radiology Report Generation (accepted to CVPR 2023)

## arXiv

arXiv link coming soon...

## Abstract

The automatic generation of radiology reports has the potential to assist radiologists in the time-consuming task of report writing. Existing methods generate the full report from image-level features, failing to explicitly focus on anatomical regions in the image. We propose a simple yet effective region-guided report generation model that detects anatomical regions and then describes individual, salient regions to form the final report. While previous methods generate reports without the possibility of human intervention and with limited explainability, our method opens up novel clinical use cases through additional human-in-the-loop capabilities and introduces a high degree of transparency and explainability. Comprehensive experiments demonstrate the effectiveness of our method in both the report generation task and the human-in-the-loop capabilities, outperforming previous state-of-the-art models.

## Method

![image info](./figures_repo/method.png) *Figure 1. **R**egion-**G**uided Radiology **R**eport **G**eneration (RGRG): the object detector extracts visual features for 29 unique anatomical regions in the chest. Two subsequent binary classifiers select salient region features for the final report and encode strong abnormal information in the features, respectively. The language model generates sentences for each of the selected regions (in this example 4), forming the final report. For conciseness, residual connections and layer normalizations in the language model are not depicted.*

## Quantitative Results

![image info](./figures_repo/nlg_metrics_table.png) *Table 1. Natural language generation (NLG) metrics for the full report generation task. Our model is competitive with or outperforms previous state-of-the-art models on a variety of metrics.*

![image info](./figures_repo/clinical_efficacy_metrics_table.png) *Table 2. Clinical efficacy (CE) metrics micro-averaged over 5 observations (denoted by mic-5) and example-based averaged over 14 observations (denoted by ex-14). RL represents reinforcement learning. Our model outperforms all non-RL models by large margins and is competitive with the two RL-based models directly optimized on CE metrics. Dashed lines highlight the scores of the best non-RL baseline.*

## Qualitative Results

<p align="center">
  <img src="figures_repo/full_report_generation.png" alt="Full report generation" width="30%">
</p>
<p align="left">Figure 2. <b>Full report generation</b> for a test set image. Detected anatomical regions (solid boxes), corresponding generated sentences, and semantically matching reference sentences are colored the same. The generated report mostly captures the information contained in the reference report, as reflected by the matching colors.</p>

<p align="center">
  <img src="figures_repo/anatomy_based_sentence_generation.png" alt="Anatomy-based generation" width="60%">
</p>
<p align="left">Figure 3. Interactive capability of <b>anatomy-based sentence generation</b> allows for the generation of descriptions for anatomical regions explicitly chosen by the radiologist. We show predicted (dashed boxes) and ground-truth (solid boxes) anatomical regions and color sentences accordingly. We observe that the model generates pertinent, anatomy-related sentences.</p>

## Setup

1. Create conda environment with "**conda env create -f environment.yml**"
2. In src/path_datasets_and_weights.py, specify the paths to the various datasets (Chest ImaGenome, MIMIC-CXR, MIMIC-CXR-JPG), CheXbert weights, and folders in which the runs are saved. Follow the instructions of the doc string of path_datasets_and_weights.py.

## Create train, val and test csv files

After the setup, run "**python create_dataset.py**" in src/dataset/ to create training, val and test csv files, in which each row contains specific information about a single image. See doc string of create_dataset.py for more details.

## Training and Testing

Please read the README_TRAIN_TEST.md for specific information on training and testing the model.

If you have any questions, please don't hesitate to contact me via linkedIn: https://www.linkedin.com/in/tim-tanida/
