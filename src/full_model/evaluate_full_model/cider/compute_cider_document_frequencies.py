"""
Code is loosely based on Miura's (https://arxiv.org/abs/2010.10042) implementation: https://github.com/ysmiura/ifcc/blob/master/cider-df.py

It calculates the document frequencies which will be used in the CIDEr score calculations.
"""
import csv
import gzip
import os
import pickle
import pathlib

from nltk.tokenize import wordpunct_tokenize
from pycocoevalcap.cider.cider_scorer import CiderScorer

# from src.path_datasets_and_weights import path_full_dataset


def get_reference_reports_val_set():
    ref_reports = []

    # Miura computes the document frequency on the "findings" section of the reference reports of the train set,
    # but since my train set does not have the reference reports, I calculate it on the val set
    path_val_set_csv_file = os.path.join("/u/home/tanida/datasets/dataset-with-reference-reports", "valid.csv")

    with open(path_val_set_csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        for row in csv_reader:
            reference_report = row[-1]
            ref_reports.append(reference_report)

    return ref_reports


def compute_cider_df():
    # 2 functions below are based on the default command line arguments that Miura uses
    tokenize_func = wordpunct_tokenize
    textfilter_func = str.lower

    ref_reports = get_reference_reports_val_set()

    # processed_ref_reports is the equivalent of what Miura calls "ftexts" in line 58 of his implementation
    processed_ref_reports = []
    for ref_report in ref_reports:
        tokens = tokenize_func(textfilter_func(ref_report))
        processed_ref_report = " ".join(tokens)
        processed_ref_reports.append(processed_ref_report)

    # these 3 lines are equivalent to line 65 of Miura's implementation
    scorer = CiderScorer(refs=ref_reports)
    scorer.compute_doc_freq()
    df = scorer.document_frequency

    parent_path_of_this_file = pathlib.Path(__file__).parent.resolve()
    output_path = os.path.join(parent_path_of_this_file, "mimic-cxr-document-frequency-unprocessed.bin.gz")
    with gzip.open(output_path, 'w') as f:
        pickle.dump(df, f)


if __name__ == "__main__":
    compute_cider_df()
