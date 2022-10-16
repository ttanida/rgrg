import gzip
import os
import pathlib
import pickle

import numpy as np
from pycocoevalcap.cider.cider_scorer import CiderScorer

from src.full_model.evaluate_full_model.cider.compute_cider_document_frequencies import compute_cider_df


class CustomCiderScorer(CiderScorer):
    """
    Custom Cider Scorer uses document frequency calculated on the reference reports of the validation set.
    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        super().__init__(test, refs, n, sigma)

        self.document_frequency = self._get_document_frequency()

    def _get_document_frequency(self):
        parent_path_of_this_file = pathlib.Path(__file__).parent.resolve()
        df_file = os.path.join(parent_path_of_this_file, "mimic-cxr-document-frequency.bin.gz")

        if not os.path.exists(df_file):
            compute_cider_df()

        with gzip.open(df_file) as f:
            cider_df = pickle.load(f)

        return cider_df

    def compute_score(self):
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)
