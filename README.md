# region-guided-chest-x-ray-report-generation
Repo to store code for my master's thesis on chest x-ray report generation

Setup:

1. Create virtual environment (Python 3.10.4)
2. Run "pip install -e ." in the root directory (i.e. region-guided-chest-x-ray-report-generation) to install src as a standalone package. This allows imports from sibling directories.
3. Run "pip install -r requirements.txt"[^1]

[^1]: I would recommend first installing torch via https://pytorch.org/get-started/locally/ (since your local setup is likely different to mine), then commenting out torch and torchvision in the requirements file before running the pip install command

- Download und unzip file mimic-cxr-reports.zip in https://physionet.org/content/mimic-cxr/2.0.0/
to get all reports as txt files.
