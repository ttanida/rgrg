# region-guided-chest-x-ray-report-generation
Repo to store code for my master's thesis on chest x-ray report generation

Setup:

1. Create virtual environment (Python 3.10.4)
2. Run "pip install -r requirements.txt"
3. Run "pip install -e ." in the root directory (i.e. region-guided-chest-x-ray-report-generation)
to install src as a standalone package


- To download report txt files from mimic-cxr:
"wget -r -N -c -np -A txt --user your_user_name --ask-password https://physionet.org/files/mimic-cxr/2.0.0/"