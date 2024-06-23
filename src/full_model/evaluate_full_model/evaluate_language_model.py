"""
This module contains all functions used to evaluate the language model.

The (main) function evaluate_language_model of this module is called by the function evaluate_model in evaluate_model.py.

evaluate_language_model returns language_model_scores which include:
    - METEOR for:
        - all generated sentences
        - generated sentences for each region
        - generated sentences with gt = normal region (i.e. the region was considered normal by the radiologist)
        - generated sentences with gt = abnormal region (i.e. the region was considered abnormal by the radiologist)

    - BLEU 1-4, METEOR, ROUGE-L, CIDEr-D for all generated reports
    - Clinical efficacy metrics for all generated reports:
        - micro-averaged over 5 observations
        - exampled-based averaged over all 14 observations
        - computed for each observation individually

It also calls subfunctions which:
    - save NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE (see run_configurations.py) batches of generated sentences as a txt file
    (for manual verification what the model generates)
    - save NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE (see run_configurations.py) batches of generated reports as a txt file
    (for manual verification what the model generates)
    - save NUM_IMAGES_TO_PLOT (see run_configurations.py) images to tensorboard where gt and predicted bboxes for every region are depicted,
    as well as the generated sentences (if they exist) and reference sentences for every region
"""
from collections import defaultdict
import csv
import io
import os
import re
import tempfile

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import spacy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from tqdm import tqdm

from src.CheXbert.src.constants import CONDITIONS
from src.CheXbert.src.label import label
from src.CheXbert.src.models.bert_labeler import bert_labeler
from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.evaluate_full_model.cider.cider import Cider
from src.full_model.run_configurations import (
    BATCH_SIZE,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
    NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
    NUM_IMAGES_TO_PLOT,
    BERTSCORE_SIMILARITY_THRESHOLD,
)
from src.path_datasets_and_weights import path_chexbert_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_NLG_scores(nlg_metrics: list[str], gen_sents_or_reports: list[str], ref_sents_or_reports: list[str]) -> dict[str, float]:
    def convert_for_pycoco_scorer(sents_or_reports: list[str]):
        """
        The compute_score methods of the scorer objects require the input not to be list[str],
        but of the form:
        generated_reports =
        {
            "image_id_0" = ["1st generated report"],
            "image_id_1" = ["2nd generated report"],
            ...
        }

        Hence we convert the generated/reference sentences/reports into the appropriate format and also tokenize them
        following Nicolson's (https://arxiv.org/pdf/2201.09405.pdf) implementation (https://github.com/aehrc/cvt2distilgpt2/blob/main/transmodal/metrics/chen.py):
        see lines 132 and 133
        """
        sents_or_reports_converted = {}
        for num, text in enumerate(sents_or_reports):
            sents_or_reports_converted[str(num)] = [re.sub(' +', ' ', text.replace(".", " ."))]

        return sents_or_reports_converted
    """
    Computes NLG metrics that are specified in metrics list (1st input argument):
        - Bleu 1-4
        - Meteor
        - Rouge-L
        - Cider-D

    Returns a dict that maps from the metrics specified to the corresponding scores.
    """
    scorers = {}
    if "bleu" in nlg_metrics:
        scorers["bleu"] = Bleu(4)
    if "meteor" in nlg_metrics:
        scorers["meteor"] = Meteor()
    if "rouge" in nlg_metrics:
        scorers["rouge"] = Rouge()  # this is actually the Rouge-L score, even if the class name only says Rouge
    if "cider" in nlg_metrics:
        scorers["cider"] = Cider()  # this is actually the Cider-D score, even if the class name only says Cider

    gen_sents_or_reports = convert_for_pycoco_scorer(gen_sents_or_reports)
    ref_sents_or_reports = convert_for_pycoco_scorer(ref_sents_or_reports)

    nlg_scores = {}

    for metric_name, scorer in scorers.items():
        score, _ = scorer.compute_score(ref_sents_or_reports, gen_sents_or_reports)
        if metric_name == "bleu":
            nlg_scores["bleu_1"] = score[0]
            nlg_scores["bleu_2"] = score[1]
            nlg_scores["bleu_3"] = score[2]
            nlg_scores["bleu_4"] = score[3]
        else:
            nlg_scores[metric_name] = score

    return nlg_scores


def compute_clinical_efficacy_scores(language_model_scores: dict, gen_reports: list[str], ref_reports: list[str]):
    """
    This function computes:
        - micro average CE scores over all 14 conditions
        - micro average CE scores over 5 conditions ("Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion")
        -> this is done following Miura (https://arxiv.org/pdf/2010.10042.pdf)
        - (micro) average CE scores of each condition
        - example-based CE scores over all 14 conditions
        -> this is done following Nicolson (https://arxiv.org/pdf/2201.09405.pdf)

    To compute these scores, we first need to get the disease labels extracted by CheXbert for both the generated and reference reports.
    This is done by the (nested) function "get_chexbert_labels_for_gen_and_ref_reports". Inside this function, there is another function
    called "label" from the module src/CheXbert/src/label.py that extracts these labels requiring 2 input arguments:
        1. chexbert (nn.Module): instantiated chexbert model
        2. csv_path (str): path to the csv file with the reports. The csv file has to have 1 column titled "Report Impression"
        under which the reports can be found

    We use a temporary directory to create the csv files for the generated and reference reports.

    The function label returns preds_gen_reports and preds_ref_reports respectively, which are List[List[int]],
    with the outer list always having len=14 (for 14 conditions, specified in CheXbert/src/constants.py),
    and the inner list has len=num_reports.

    E.g. the 1st inner list could be [2, 1, 0, 3], which means the 1st report has label 2 for the 1st condition (which is 'Enlarged Cardiomediastinum'),
    the 2nd report has label 1 for the 1st condition, the 3rd report has label 0 for the 1st condition, the 4th and final report label 3 for the 1st condition.

    There are 4 possible labels:
        0: blank/NaN (i.e. no prediction could be made about a condition, because it was no mentioned in a report)
        1: positive (condition was mentioned as present in a report)
        2: negative (condition was mentioned as not present in a report)
        3: uncertain (condition was mentioned as possibly present in a report)

    To compute the micro average scores (i.e. all the scores except of the example-based scores), we follow the implementation of the paper
    by Miura et. al., who considered the negative and blank/NaN to be one whole negative class, and positive and uncertain to be one whole positive class.
    For reference, see lines 141 and 143 of Miura's implementation: https://github.com/ysmiura/ifcc/blob/master/eval_prf.py#L141,
    where label 3 is converted to label 1, and label 2 is converted to label 0.

    To compute the example-based scores, we follow Nicolson's implementation, who considered blank/NaN, negative and uncertain to be the negative class,
    and only positive to be the positive class. Meaning labels 2 and 3 are converted to label 0.
    """

    def get_chexbert():
        model = bert_labeler()
        model = nn.DataParallel(model)  # needed since weights were saved with nn.DataParallel
        checkpoint = torch.load(path_chexbert_weights, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        return model

    def get_chexbert_labels_for_gen_and_ref_reports():
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_gen_reports_file_path = os.path.join(temp_dir, "gen_reports.csv")
            csv_ref_reports_file_path = os.path.join(temp_dir, "ref_reports.csv")

            header = ["Report Impression"]

            with open(csv_gen_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[gen_report] for gen_report in gen_reports])

            with open(csv_ref_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[ref_report] for ref_report in ref_reports])

            # preds_*_reports are List[List[int]] with the labels extracted by CheXbert (see doc string for details)
            preds_gen_reports = label(chexbert, csv_gen_reports_file_path)
            preds_ref_reports = label(chexbert, csv_ref_reports_file_path)

        return preds_gen_reports, preds_ref_reports

    def compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports):
        def convert_labels_like_miura(preds_reports: list[list[int]]):
            """
            See doc string of update_clinical_efficacy_scores function for more details.
            Miura (https://arxiv.org/pdf/2010.10042.pdf) considers blank/NaN (label 0) and negative (label 2) to be the negative class,
            and positive (label 1) and uncertain (label 3) to be the positive class.

            Thus we convert label 2 -> label 0 and label 3 -> label 1.
            """
            def convert_label(label: int):
                if label == 2:
                    return 0
                elif label == 3:
                    return 1
                else:
                    return label

            preds_reports_converted = [[convert_label(label) for label in condition_list] for condition_list in preds_reports]

            return preds_reports_converted

        preds_gen_reports_converted = convert_labels_like_miura(preds_gen_reports)
        preds_ref_reports_converted = convert_labels_like_miura(preds_ref_reports)

        # for the CE scores, we follow Miura (https://arxiv.org/pdf/2010.10042.pdf) in micro averaging them over these 5 conditions:
        five_conditions_to_evaluate = {"Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"}

        total_preds_gen_reports_5_conditions = []
        total_preds_ref_reports_5_conditions = []

        # we also compute the micro average over all 14 conditions:
        total_preds_gen_reports_14_conditions = []
        total_preds_ref_reports_14_conditions = []

        # iterate over the 14 conditions
        for preds_gen_reports_condition, preds_ref_reports_condition, condition in zip(preds_gen_reports_converted, preds_ref_reports_converted, CONDITIONS):
            if condition in five_conditions_to_evaluate:
                total_preds_gen_reports_5_conditions.extend(preds_gen_reports_condition)
                total_preds_ref_reports_5_conditions.extend(preds_ref_reports_condition)

            total_preds_gen_reports_14_conditions.extend(preds_gen_reports_condition)
            total_preds_ref_reports_14_conditions.extend(preds_ref_reports_condition)

            # compute and save scores for the given condition
            precision, recall, f1, _ = precision_recall_fscore_support(preds_ref_reports_condition, preds_gen_reports_condition, average="binary")
            acc = accuracy_score(preds_ref_reports_condition, preds_gen_reports_condition)

            language_model_scores["report"]["CE"][condition]["precision"] = precision
            language_model_scores["report"]["CE"][condition]["recall"] = recall
            language_model_scores["report"]["CE"][condition]["f1"] = f1
            language_model_scores["report"]["CE"][condition]["acc"] = acc

        # compute and save scores for all 14 conditions
        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions)

        language_model_scores["report"]["CE"]["precision_micro_all"] = precision
        language_model_scores["report"]["CE"]["recall_micro_all"] = recall
        language_model_scores["report"]["CE"]["f1_micro_all"] = f1
        language_model_scores["report"]["CE"]["acc_all"] = acc

        # compute and save scores for the 5 conditions
        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions)

        language_model_scores["report"]["CE"]["precision_micro_5"] = precision
        language_model_scores["report"]["CE"]["recall_micro_5"] = recall
        language_model_scores["report"]["CE"]["f1_micro_5"] = f1
        language_model_scores["report"]["CE"]["acc_5"] = acc

    def compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports):
        """
        example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
        """
        preds_gen_reports_np = np.array(preds_gen_reports)  # array of shape (14 x num_reports), 14 for 14 conditions
        preds_ref_reports_np = np.array(preds_ref_reports)  # array of shape (14 x num_reports)

        # convert label 1 to True and everything else (i.e. labels 0, 2, 3) to False
        # (effectively doing the label conversion as done by Nicolson, see doc string of compute_clinical_efficacy_scores for more details)
        preds_gen_reports_np = preds_gen_reports_np == 1
        preds_ref_reports_np = preds_ref_reports_np == 1

        tp = np.logical_and(preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fp = np.logical_and(preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fn = np.logical_and(~preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        tn = np.logical_and(~preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)

        # sum up the TP, FP, FN and TN for each report (i.e. for each column)
        tp_example = tp.sum(axis=0)  # int array of shape (num_reports)
        fp_example = fp.sum(axis=0)  # int array of shape (num_reports)
        fn_example = fn.sum(axis=0)  # int array of shape (num_reports)
        tn_example = tn.sum(axis=0)  # int array of shape (num_reports)

        # compute the scores for each report
        precision_example = tp_example / (tp_example + fp_example)  # float array of shape (num_reports)
        recall_example = tp_example / (tp_example + fn_example)  # float array of shape (num_reports)
        f1_example = (2 * tp_example) / (2 * tp_example + fp_example + fn_example)  # float array of shape (num_reports)
        acc_example = (tp_example + tn_example) / (tp_example + tn_example + fp_example + fn_example)  # float array of shape (num_reports)

        # since there can be cases of zero division, we have to replace the resulting nan values with 0.0
        precision_example[np.isnan(precision_example)] = 0.0
        recall_example[np.isnan(recall_example)] = 0.0
        f1_example[np.isnan(f1_example)] = 0.0
        acc_example[np.isnan(acc_example)] = 0.0

        # finally, take the mean over the scores for all reports
        precision_example = float(precision_example.mean())
        recall_example = float(recall_example.mean())
        f1_example = float(f1_example.mean())
        acc_example = float(acc_example.mean())

        language_model_scores["report"]["CE"]["precision_example_all"] = precision_example
        language_model_scores["report"]["CE"]["recall_example_all"] = recall_example
        language_model_scores["report"]["CE"]["f1_example_all"] = f1_example
        language_model_scores["report"]["CE"]["acc_example_all"] = acc_example

    chexbert = get_chexbert()
    preds_gen_reports, preds_ref_reports = get_chexbert_labels_for_gen_and_ref_reports()

    compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports)
    compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports)


def compute_language_model_scores(gen_and_ref_sentences, gen_and_ref_reports):

    def compute_sentence_level_scores():
        def remove_gen_sents_corresponding_to_empty_ref_sents(gen_sents, ref_sents):
            """
            We can't compute scores on generated sentences whose corresponding reference sentence is empty.
            So we need to discard them both.
            """
            filtered_gen_sents = []
            filtered_ref_sents = []

            for gen_sent, ref_sent in zip(gen_sents, ref_sents):
                if ref_sent != "":
                    filtered_gen_sents.append(gen_sent)
                    filtered_ref_sents.append(ref_sent)

            return filtered_gen_sents, filtered_ref_sents

        def compute_sent_level_scores_for_subset(subset, gen_sents, ref_sents):
            nlg_metrics = ["meteor"]
            nlg_scores = compute_NLG_scores(nlg_metrics, gen_sents, ref_sents)
            meteor_score = nlg_scores["meteor"]
            language_model_scores[subset]["meteor"] = meteor_score

        def compute_sent_level_scores_for_region(region_name, gen_sents, ref_sents):
            nlg_metrics = ["meteor"]
            nlg_scores = compute_NLG_scores(nlg_metrics, gen_sents, ref_sents)
            meteor_score = nlg_scores["meteor"]
            language_model_scores["region"][region_name]["meteor"] = meteor_score

        def compute_sent_level_meteor_ratio_score(gen_sents, ref_sents):
            """
            We want to compute the ratio of the meteor scores for when:
            - a generated sentence is paired with its corresponding reference sentence of a given image (value for the numerator)
            vs
            - a generated sentence is paired with all other non-corresponding reference sentences of a given image (value for the denominator)

            the numerator value is already computed by language_model_scores["all"]["meteor"], since this is exactly the meteor score for when the generated sentences
            are paried with their corresponding reference sentences. Hence only the denominator value has to be calculated separately
            """
            gen_sents_for_computing_meteor_ratio_score = []
            ref_sents_for_computing_meteor_ratio_score = []

            # List[int] that can be used to get all generated and reference sentences that correspond to the same image
            num_generated_sentences_per_image = gen_and_ref_sentences["num_generated_sentences_per_image"]

            curr_index = 0
            for num_gen_sents in num_generated_sentences_per_image:
                gen_sents_single_image = gen_sents[curr_index:curr_index + num_gen_sents]
                ref_sents_single_image = ref_sents[curr_index:curr_index + num_gen_sents]
                # the number of generated sentences per image is exactly the same as the number of "retrieved" reference sentences per image
                # (see function get_ref_sentences_for_selected_regions)

                curr_index += num_gen_sents

                gen_sents_single_image_filtered, ref_sents_single_image_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(gen_sents_single_image, ref_sents_single_image)

                # now that we have the generated and (non-empty) reference sentences for a single image,
                # we have to pair-wise match them except for the "correct" match between the corresponding gen and ref sents (see doc string of this function)
                for i, gen_sent in enumerate(gen_sents_single_image_filtered):
                    for j, ref_sent in enumerate(ref_sents_single_image_filtered):
                        if i == j:
                            continue  # skip "correct" match
                        else:
                            gen_sents_for_computing_meteor_ratio_score.append(gen_sent)
                            ref_sents_for_computing_meteor_ratio_score.append(ref_sent)

            # compute the "denominator" meteor score
            nlg_metrics = ["meteor"]
            nlg_scores = compute_NLG_scores(nlg_metrics, gen_sents_for_computing_meteor_ratio_score, ref_sents_for_computing_meteor_ratio_score)
            denominator_meteor_score = nlg_scores["meteor"]

            numerator_meteor_score = language_model_scores["all"]["meteor"]

            language_model_scores["all"]["meteor_ratio"] = numerator_meteor_score / denominator_meteor_score

        generated_sents = gen_and_ref_sentences["generated_sentences"]
        generated_sents_normal = gen_and_ref_sentences["generated_sentences_normal_selected_regions"]
        generated_sents_abnormal = gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"]

        reference_sents = gen_and_ref_sentences["reference_sentences"]
        reference_sents_normal = gen_and_ref_sentences["reference_sentences_normal_selected_regions"]
        reference_sents_abnormal = gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"]

        gen_sents_filtered, ref_sents_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents, reference_sents)
        gen_sents_normal_filtered, ref_sents_normal_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents_normal, reference_sents_normal)
        gen_sents_abnormal_filtered, ref_sents_abnormal_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents_abnormal, reference_sents_abnormal)

        compute_sent_level_scores_for_subset("all", gen_sents_filtered, ref_sents_filtered)
        compute_sent_level_scores_for_subset("normal", gen_sents_normal_filtered, ref_sents_normal_filtered)
        compute_sent_level_scores_for_subset("abnormal", gen_sents_abnormal_filtered, ref_sents_abnormal_filtered)

        compute_sent_level_meteor_ratio_score(generated_sents, reference_sents)

        for region_index, region_name in enumerate(ANATOMICAL_REGIONS):
            region_generated_sentences = gen_and_ref_sentences[region_index]["generated_sentences"]
            region_reference_sentences = gen_and_ref_sentences[region_index]["reference_sentences"]

            region_gen_sents_filtered, region_ref_sents_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(region_generated_sentences, region_reference_sentences)

            if len(region_gen_sents_filtered) != 0:
                compute_sent_level_scores_for_region(region_name, region_gen_sents_filtered, region_ref_sents_filtered)
            else:
                language_model_scores["region"][region_name]["meteor"] = -1

    def compute_report_level_scores():
        gen_reports = gen_and_ref_reports["generated_reports"]
        ref_reports = gen_and_ref_reports["reference_reports"]

        nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
        nlg_scores = compute_NLG_scores(nlg_metrics, gen_reports, ref_reports)

        for nlg_metric_name, score in nlg_scores.items():
            language_model_scores["report"][nlg_metric_name] = score

        compute_clinical_efficacy_scores(language_model_scores, gen_reports, ref_reports)

    def create_language_model_scores_dict():
        language_model_scores = {}

        # on report-level, we evalute on:
        # BLEU 1-4
        # METEOR
        # ROUGE-L
        # Cider-D
        # CE scores (P, R, F1, acc)
        language_model_scores["report"] = {f"bleu_{i}": None for i in range(1, 5)}
        language_model_scores["report"]["meteor"] = None
        language_model_scores["report"]["rouge"] = None
        language_model_scores["report"]["cider"] = None
        language_model_scores["report"]["CE"] = {
            # following Miura (https://arxiv.org/pdf/2010.10042.pdf), we evaluate the micro average CE scores over these 5 diseases/conditions:
            # "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"
            "precision_micro_5": None,
            "recall_micro_5": None,
            "f1_micro_5": None,
            "acc_5": None,

            # we additionally compute the micro average CE scores over all conditions
            "precision_micro_all": None,
            "recall_micro_all": None,
            "acc_all": None
        }

        # we also compute the CE scores for each of the 14 conditions individually
        for condition in CONDITIONS:
            language_model_scores["report"]["CE"][condition] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "acc": None
            }

        # following Nicolson (https://arxiv.org/pdf/2201.09405.pdf), we evaluate the example-based CE scores over all conditions
        # example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
        language_model_scores["report"]["CE"]["precision_example_all"] = None
        language_model_scores["report"]["CE"]["recall_example_all"] = None
        language_model_scores["report"]["CE"]["f1_example_all"] = None
        language_model_scores["report"]["CE"]["acc_example_all"] = None

        # on sentence-level, we only evaluate on METEOR, since this metric gives meaningful scores on sentence-level (as opposed to e.g. BLEU)
        # we distinguish between generated sentences for all, normal, and abnormal regions
        for subset in ["all", "normal", "abnormal"]:
            language_model_scores[subset] = {"meteor": None}

        # we also compute these scores for each region individually
        language_model_scores["region"] = {}
        for region_name in ANATOMICAL_REGIONS:
            language_model_scores["region"][region_name] = {"meteor": None}

        # and finally, on sentence-level we also compute the ratio of the meteor scores for when:
        #   - a generated sentence is paired with its corresponding reference sentence of a given image (value for the numerator)
        #   vs
        #   - a generated sentence is paired with all other non-corresponding reference sentences of a given image (value for the denominator)
        #
        # the numerator value is already computed by language_model_scores["all"]["meteor"], since this is exactly the meteor score for when the generated sentences
        # are paired with their corresponding reference sentences. Hence only the denominator value has to be calculated separately
        language_model_scores["all"]["meteor_ratio"] = None

        return language_model_scores

    language_model_scores = create_language_model_scores_dict()

    compute_report_level_scores()
    compute_sentence_level_scores()

    return language_model_scores


def write_sentences_and_reports_to_file(
    gen_and_ref_sentences,
    gen_and_ref_reports,
    gen_sentences_with_corresponding_regions,
    generated_sentences_and_reports_folder_path,
    overall_steps_taken,
):
    def write_sentences():
        txt_file_name = os.path.join(generated_sentences_and_reports_folder_path, "generated_sentences", f"generated_sentences_step_{overall_steps_taken}")
        txt_file_name_abnormal = os.path.join(generated_sentences_and_reports_folder_path, "generated_sentences", f"generated_abnormal_sentences_step_{overall_steps_taken}")

        with open(txt_file_name, "w") as f:
            for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")

        with open(txt_file_name_abnormal, "w") as f:
            for gen_sent, ref_sent in zip(generated_sentences_abnormal_regions, reference_sentences_abnormal_regions):
                f.write(f"Generated sentence: {gen_sent}\n")
                f.write(f"Reference sentence: {ref_sent}\n\n")

    def write_reports():
        txt_file_name = os.path.join(
            generated_sentences_and_reports_folder_path,
            "generated_reports",
            f"generated_reports_step_{overall_steps_taken}",
        )

        with open(txt_file_name, "w") as f:
            for gen_report, ref_report, removed_similar_gen_sents, gen_sents_with_regions_single_report in zip(
                generated_reports,
                reference_reports,
                removed_similar_generated_sentences,
                gen_sentences_with_corresponding_regions
            ):
                f.write(f"Generated report: {gen_report}\n\n")
                f.write(f"Reference report: {ref_report}\n\n")

                f.write("Generated sentences with their regions:\n")
                for region_name, gen_sent in gen_sents_with_regions_single_report:
                    f.write(f"\t{region_name}: {gen_sent}\n")
                f.write("\n")

                f.write("Generated sentences that were removed:\n")
                for gen_sent, list_similar_gen_sents in removed_similar_gen_sents.items():
                    f.write(f"\t{gen_sent} == {list_similar_gen_sents}\n")
                f.write("\n")

                f.write("=" * 30)
                f.write("\n\n")

    num_generated_sentences_to_save = NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE * BATCH_SIZE
    num_generated_reports_to_save = NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE

    # all below are list of str
    generated_sentences = gen_and_ref_sentences["generated_sentences"][:num_generated_sentences_to_save]
    generated_sentences_abnormal_regions = gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"][:num_generated_sentences_to_save]
    reference_sentences = gen_and_ref_sentences["reference_sentences"][:num_generated_sentences_to_save]
    reference_sentences_abnormal_regions = gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"][:num_generated_sentences_to_save]

    write_sentences()

    # all below are list of str except removed_similar_generated_sentences which is a list of dict
    generated_reports = gen_and_ref_reports["generated_reports"][:num_generated_reports_to_save]
    reference_reports = gen_and_ref_reports["reference_reports"][:num_generated_reports_to_save]
    removed_similar_generated_sentences = gen_and_ref_reports["removed_similar_generated_sentences"][:num_generated_reports_to_save]

    write_reports()


def get_plot_title(region_set, region_indices, region_colors, class_detected_img) -> str:
    """
    Get a plot title like in the below example.
    1 region_set always contains 6 regions (except for region_set_5, which has 5 regions).
    The characters in the brackets represent the colors of the corresponding bboxes (e.g. b = blue),
    "nd" stands for "not detected" in case the region was not detected by the object detector.

    right lung (b), right costophrenic angle (g, nd), left lung (r)
    left costophrenic angle (c), cardiac silhouette (m), spine (y, nd)
    """
    # get a list of 6 boolean values that specify if that region was detected
    class_detected = [class_detected_img[region_index] for region_index in region_indices]

    # add color_code to region name (e.g. "(r)" for red)
    # also add nd to the brackets if region was not detected (e.g. "(r, nd)" if red region was not detected)
    region_set = [
        region + f" ({color})" if cls_detect else region + f" ({color}, nd)"
        for region, color, cls_detect in zip(region_set, region_colors, class_detected)
    ]

    # add a line break to the title, as to not make it too long
    return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])


def get_generated_sentence_for_region(
    generated_sentences_for_selected_regions, selected_regions, num_img, region_index
) -> str:
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): holds the generated sentences for all regions that were selected in the batch, i.e. of length "num_regions_selected_in_batch"
        selected_regions (Tensor[bool]): of shape [batch_size x 29], specifies for each region if it was selected to get a sentences generated (True) or not by the binary classifier for region selection.
        Ergo has exactly "num_regions_selected_in_batch" True values.
        num_img (int): specifies the image we are currently processing in the batch, its value is in the range [0, batch_size-1]
        region_index (int): specifies the region we are currently processing of a single image, its value is in the range [0, 28]

    Returns:
        str: generated sentence for region specified by num_img and region_index

    Implementation is not too easy to understand, so here is a toy example with some toy values to explain.

    generated_sentences_for_selected_regions = ["Heart is ok.", "Spine is ok."]
    selected_regions = [
        [False, False, True],
        [True, False, False]
    ]
    num_img = 0
    region_index = 2

    In this toy example, the batch_size = 2 and there are only 3 regions in total for simplicity (instead of the 29).
    The generated_sentences_for_selected_regions is of len 2, meaning num_regions_selected_in_batch = 2.
    Therefore, the selected_regions boolean tensor also has exactly 2 True values.

    (1) Flatten selected_regions:
        selected_regions_flat = [False, False, True, True, False, False]

    (2) Compute cumsum (to get an incrementation each time there is a True value):
        cum_sum_true_values = [0, 0, 1, 2, 2, 2]

    (3) Reshape cum_sum_true_values to shape of selected_regions
        cum_sum_true_values = [
            [0, 0, 1],
            [2, 2, 2]
        ]

    (4) Subtract 1 from tensor, such that 1st True value in selected_regions has the index value 0 in cum_sum_true_values,
        the 2nd True value has index value 1 and so on.
        cum_sum_true_values = [
            [-1, -1, 0],
            [1, 1, 1]
        ]

    (5) Index cum_sum_true_values with num_img and region_index to get the final index for the generated sentence list
        index = cum_sum_true_values[num_img][region_index] = cum_sum_true_values[0][2] = 0

    (6) Get generated sentence:
        generated_sentences_for_selected_regions[index] = "Heart is ok."
    """
    selected_regions_flat = selected_regions.reshape(-1)
    cum_sum_true_values = np.cumsum(selected_regions_flat)

    cum_sum_true_values = cum_sum_true_values.reshape(selected_regions.shape)
    cum_sum_true_values -= 1

    index = cum_sum_true_values[num_img][region_index]

    return generated_sentences_for_selected_regions[index]


def transform_sentence_to_fit_under_image(sentence):
    """
    Adds line breaks and whitespaces such that long reference or generated sentence
    fits under the plotted image.
    Values like max_line_length and prefix_for_alignment were found by trial-and-error.
    """
    max_line_length = 60
    if len(sentence) < max_line_length:
        return sentence

    words = sentence.split()
    transformed_sent = ""
    current_line_length = 0
    prefix_for_alignment = "\n" + " " * 20
    for word in words:
        if len(word) + current_line_length > max_line_length:
            word = f"{prefix_for_alignment}{word}"
            current_line_length = -len(prefix_for_alignment)

        current_line_length += len(word)
        transformed_sent += word + " "

    return transformed_sent


def update_region_set_text(
    region_set_text,
    color,
    reference_sentences_img,
    generated_sentences_for_selected_regions,
    region_index,
    selected_regions,
    num_img,
):
    """
    Create a single string region_set_text like in the example below.
    Each update creates 1 paragraph for 1 region/bbox.
    The (b), (r) and (y) represent the colors of the bounding boxes (in this case blue, red and yellow).

    Example:

    (b):
      reference: Normal cardiomediastinal silhouette, hila, and pleura.
      generated: The mediastinal and hilar contours are unremarkable.

    (r):
      reference:
      generated: [REGION NOT SELECTED]

    (y):
      reference:
      generated: There is no pleural effusion or pneumothorax.

    (... continues for 3 more regions/bboxes, for a total of 6 per region_set)
    """
    region_set_text += f"({color}):\n"
    reference_sentence_region = reference_sentences_img[region_index]

    # in case sentence is too long
    reference_sentence_region = transform_sentence_to_fit_under_image(reference_sentence_region)

    region_set_text += f"  reference: {reference_sentence_region}\n"

    box_region_selected = selected_regions[num_img][region_index]
    if not box_region_selected:
        region_set_text += "  generated: [REGION NOT SELECTED]\n\n"
    else:
        generated_sentence_region = get_generated_sentence_for_region(
            generated_sentences_for_selected_regions, selected_regions, num_img, region_index
        )
        generated_sentence_region = transform_sentence_to_fit_under_image(generated_sentence_region)
        region_set_text += f"  generated: {generated_sentence_region}\n\n"

    return region_set_text


def plot_box(box, ax, clr, linestyle, region_detected=True):
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    ax.add_artist(
        plt.Rectangle(xy=(x0, y0), height=h, width=w, fill=False, color=clr, linewidth=1, linestyle=linestyle)
    )

    # add an annotation to the gt box, that the pred box does not exist (i.e. the corresponding region was not detected)
    if not region_detected:
        ax.annotate("not detected", (x0, y0), color=clr, weight="bold", fontsize=10)


def plot_detections_and_sentences_to_tensorboard(
    writer,
    num_batch,
    overall_steps_taken,
    images,
    image_targets,
    selected_regions,
    detections,
    class_detected,
    reference_sentences,
    generated_sentences_for_selected_regions,
):
    # pred_boxes_batch is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
    # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
    # the 2nd to the 2nd class and so on
    pred_boxes_batch = detections["top_region_boxes"]

    # image_targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
    # gt_boxes is of shape [batch_size x 29 x 4]
    gt_boxes_batch = torch.stack([t["boxes"] for t in image_targets], dim=0)

    # plot 6 regions at a time, as to not overload the image with boxes (except for region_set_5, which has 5 regions)
    # the region_sets were chosen as to minimize overlap between the contained regions (i.e. better visibility)
    region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
    region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
    region_set_3 = ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"]
    region_set_4 = ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"]
    region_set_5 = ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]

    regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5]

    # put channel dimension (1st dim) last (0-th dim is batch-dim)
    images = images.numpy().transpose(0, 2, 3, 1)

    for num_img, image in enumerate(images):

        gt_boxes_img = gt_boxes_batch[num_img]
        pred_boxes_img = pred_boxes_batch[num_img]
        class_detected_img = class_detected[num_img].tolist()
        reference_sentences_img = reference_sentences[num_img]

        for num_region_set, region_set in enumerate(regions_sets):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()

            plt.imshow(image, cmap="gray")
            plt.axis("on")

            region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]
            region_colors = ["b", "g", "r", "c", "m", "y"]

            if num_region_set == 4:
                region_colors.pop()

            region_set_text = ""

            for region_index, color in zip(region_indices, region_colors):
                # box_gt and box_pred are both [List[float]] of len 4
                box_gt = gt_boxes_img[region_index].tolist()
                box_pred = pred_boxes_img[region_index].tolist()
                box_region_detected = class_detected_img[region_index]

                plot_box(box_gt, ax, clr=color, linestyle="solid", region_detected=box_region_detected)

                # only plot predicted box if class was actually detected
                if box_region_detected:
                    plot_box(box_pred, ax, clr=color, linestyle="dashed")

                region_set_text = update_region_set_text(
                    region_set_text,
                    color,
                    reference_sentences_img,
                    generated_sentences_for_selected_regions,
                    region_index,
                    selected_regions,
                    num_img,
                )

            title = get_plot_title(region_set, region_indices, region_colors, class_detected_img)
            ax.set_title(title)

            plt.xlabel(region_set_text, loc="left")

            # using writer.add_figure does not correctly display the region_set_text in tensorboard
            # so instead, fig is first saved as a png file to memory via BytesIO
            # (this also saves the region_set_text correctly in the png when bbox_inches="tight" is set)
            # then the png is loaded from memory and the 4th channel (alpha channel) is discarded
            # finally, writer.add_image is used to display the image in tensorboard
            buf = io.BytesIO()
            fig.savefig(buf, bbox_inches="tight")
            buf.seek(0)
            im = Image.open(buf)
            im = np.asarray(im)[..., :3]

            writer_image_num = num_batch * BATCH_SIZE + num_img
            writer.add_image(
                f"img_{writer_image_num}_region_set_{num_region_set}",
                im,
                global_step=overall_steps_taken,
                dataformats="HWC",
            )

            plt.close(fig)


def update_gen_sentences_with_corresponding_regions(
    gen_sentences_with_corresponding_regions,
    generated_sents_for_selected_regions,
    selected_regions
):
    """
    Args:
        gen_sentences_with_corresponding_regions (list[list[tuple[str, str]]]):
            len(outer_list)= (NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE),
            and inner list has len of how many regions were selected for a given image.
            Inner list hold tuples of (region_name, gen_sent), i.e. region name and its corresponding generated sentence
        generated_sentences_for_selected_regions (List[str]): of length "num_regions_selected_in_batch"
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
    """
    def get_region_name(region_index: int):
        for i, region_name in enumerate(ANATOMICAL_REGIONS):
            if i == region_index:
                return region_name

    index_gen_sentence = 0

    # selected_regions_single_image is a row with 29 bool values corresponding to a single image
    for selected_regions_single_image in selected_regions:
        gen_sents_with_regions_single_image = []

        for region_index, region_selected_bool in enumerate(selected_regions_single_image):
            if region_selected_bool:
                region_name = get_region_name(region_index)
                gen_sent = generated_sents_for_selected_regions[index_gen_sentence]

                gen_sents_with_regions_single_image.append((region_name, gen_sent))

                index_gen_sentence += 1

        gen_sentences_with_corresponding_regions.append(gen_sents_with_regions_single_image)


def update_num_generated_sentences_per_image(
    gen_and_ref_sentences: dict,
    selected_regions: np.array
):
    """
    selected_regions is a boolean array of shape (batch_size x 29) that will have a True value for all regions that were selected and hence for which sentences were generated.
    Thus to get the number of generated sentences per image, we just have to add up the True value along axis 1 (i.e. along the region dimension)
    """
    num_gen_sents_per_image = selected_regions.sum(axis=1).tolist()  # indices is a list[int] of len(batch_size)
    gen_and_ref_sentences["num_generated_sentences_per_image"].extend(num_gen_sents_per_image)


def update_gen_and_ref_sentences_for_regions(
    gen_and_ref_sentences: dict,
    generated_sents_for_selected_regions: list[str],
    reference_sents_for_selected_regions: list[str],
    selected_regions: np.array
):
    """Updates the gen_and_ref_sentences dict for each of the 29 regions, i.e. appends the generated and reference sentences for the regions (if they exist)

    Args:
        gen_and_ref_sentences (dict):
        generated_sents_for_selected_regions (List[str]): has exactly num_regions_selected_in_batch generated sentences
        reference_sents_for_selected_regions (List[str]): has exactly num_regions_selected_in_batch reference sentences
        selected_regions (np.array([bool])): of shape batch_size x 29, has exactly num_regions_selected_in_batch True values
        that specify the regions for whom sentences were generated

    Implementation is not too easy to understand, so here is a toy example with some toy values to explain.

    generated_sents_for_selected_regions = ["Heart is ok.", "Spine is ok."]
    reference_sents_for_selected_regions = ["Cardiac silhouette is ok.", "Spine is not ok."]
    selected_regions = [
        [False, False, True],
        [True, False, False]
    ]

    In this toy example, the batch_size = 2 and there are only 3 regions in total for simplicity (instead of the 29).
    The generated_sents_for_selected_regions and reference_sents_for_selected_regions are of len 2, meaning num_regions_selected_in_batch = 2.
    Therefore, the selected_regions boolean array also has exactly 2 True values.

    (1) Flatten selected_regions:
        selected_regions_flat = [False, False, True, True, False, False]

    (2) Iterate until 1st True value is found in selected_regions_flat:
        index_gen_ref_sentence = 0 at the moment
        curr_index = 2 at the moment
        We do a modulo operation to get the region_index, i.e. region_index = curr_index % 3 = 2

        We get the gen_sent and ref_sent at index_gen_ref_sentence, i.e.
        gen_sent = "Heart is ok."
        ref_sent = "Cardiac silhouette is ok."

        We append them to the respective lists in gen_and_ref_sentences[region_index]

        We increase index_gen_ref_sentence by 1, such that at the next True value the next gen_sent and ref_sent are taken.

    (2) Iterate until the 2nd True value is found in selected_regions_flat:
        index_gen_ref_sentence = 1 at the moment
        curr_index = 3 at the moment
        We do a modulo operation to get the region_index, i.e. region_index = curr_index % 3 = 0

        We get the gen_sent and ref_sent at index_gen_ref_sentence, i.e.
        gen_sent = "Spine is ok."
        ref_sent = "Spine is not ok."

        We append them to the respective lists in gen_and_ref_sentences[region_index]

        We increase index_gen_ref_sentence by 1, such that at the next True value the next gen_sent and ref_sent are taken.
    """
    index_gen_ref_sentence = 0

    # of shape (batch_size * 29)
    selected_regions_flat = selected_regions.reshape(-1)
    for curr_index, region_selected_bool in enumerate(selected_regions_flat):
        if region_selected_bool:
            region_index = curr_index % 29
            gen_sent = generated_sents_for_selected_regions[index_gen_ref_sentence]
            ref_sent = reference_sents_for_selected_regions[index_gen_ref_sentence]

            gen_and_ref_sentences[region_index]["generated_sentences"].append(gen_sent)
            gen_and_ref_sentences[region_index]["reference_sentences"].append(ref_sent)

            index_gen_ref_sentence += 1


def get_generated_reports(
    generated_sentences_for_selected_regions,
    selected_regions,
    sentence_tokenizer,
    bertscore_threshold,
    bert_score,
):
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): of length "num_regions_selected_in_batch"
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
        sentence_tokenizer: used in remove_duplicate_generated_sentences to separate the generated sentences
        bert_score: instance of the evaluate bert score evaluation module

    Return:
        generated_reports (List[str]): list of length batch_size containing generated reports for every image in batch
        removed_similar_generated_sentences (List[Dict[str, List]): list of length batch_size containing dicts that map from one generated sentence to a list
        of other generated sentences that were removed because they were too similar. Useful for manually verifying if removing similar generated sentences was successful
    """
    def remove_duplicate_generated_sentences(gen_report_single_image, bert_score):
        def check_gen_sent_in_sents_to_be_removed(gen_sent, similar_generated_sents_to_be_removed):
            for lists_of_gen_sents_to_be_removed in similar_generated_sents_to_be_removed.values():
                if gen_sent in lists_of_gen_sents_to_be_removed:
                    return True

            return False

        # since different (closely related) regions can have the same generated sentence, we first remove exact duplicates

        # use sentence tokenizer to separate the generated sentences
        gen_sents_single_image = sentence_tokenizer(gen_report_single_image).sents

        # convert spacy.tokens.span.Span object into str by using .text attribute
        gen_sents_single_image = [sent.text for sent in gen_sents_single_image]

        # remove exact duplicates using a dict as an ordered set
        # note that dicts are insertion ordered as of Python 3.7
        gen_sents_single_image = list(dict.fromkeys(gen_sents_single_image))

        # there can still be generated sentences that are not exact duplicates, but nonetheless very similar
        # e.g. "The cardiomediastinal silhouette is normal." and "The cardiomediastinal silhouette is unremarkable."
        # to remove these "soft" duplicates, we use bertscore

        # similar_generated_sents_to_be_removed maps from one sentence to a list of similar sentences that are to be removed
        similar_generated_sents_to_be_removed = defaultdict(list)

        # TODO:
        # the nested for loops below check each generated sentence with every other generated sentence
        # this is not particularly efficient, since e.g. generated sentences for the region "right lung" most likely
        # will never be similar to generated sentences for the region "abdomen"
        # thus, one could speed up these checks by only checking anatomical regions that are similar to each other

        for i in range(len(gen_sents_single_image)):
            gen_sent_1 = gen_sents_single_image[i]

            for j in range(i + 1, len(gen_sents_single_image)):
                if check_gen_sent_in_sents_to_be_removed(gen_sent_1, similar_generated_sents_to_be_removed):
                    break

                gen_sent_2 = gen_sents_single_image[j]
                if check_gen_sent_in_sents_to_be_removed(gen_sent_2, similar_generated_sents_to_be_removed):
                    continue

                bert_score_result = bert_score.compute(
                    lang="en", predictions=[gen_sent_1], references=[gen_sent_2], model_type="distilbert-base-uncased"
                )

                if bert_score_result["f1"][0] > bertscore_threshold:
                    # remove the generated similar sentence that is shorter
                    if len(gen_sent_1) > len(gen_sent_2):
                        similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                    else:
                        similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

        gen_report_single_image = " ".join(
            sent for sent in gen_sents_single_image if not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
        )

        return gen_report_single_image, similar_generated_sents_to_be_removed

    generated_reports = []
    removed_similar_generated_sentences = []
    curr_index = 0

    for selected_regions_single_image in selected_regions:
        # sum up all True values for a single row in the array (corresponing to a single image)
        num_selected_regions_single_image = np.sum(selected_regions_single_image)

        # use curr_index and num_selected_regions_single_image to index all generated sentences corresponding to a single image
        gen_sents_single_image = generated_sentences_for_selected_regions[
            curr_index: curr_index + num_selected_regions_single_image
        ]

        # update curr_index for next image
        curr_index += num_selected_regions_single_image

        # concatenate generated sentences of a single image to a continuous string gen_report_single_image
        gen_report_single_image = " ".join(sent for sent in gen_sents_single_image)

        gen_report_single_image, similar_generated_sents_to_be_removed = remove_duplicate_generated_sentences(
            gen_report_single_image, bert_score
        )

        generated_reports.append(gen_report_single_image)
        removed_similar_generated_sentences.append(similar_generated_sents_to_be_removed)

    return generated_reports, removed_similar_generated_sentences


def get_ref_sentences_for_selected_regions(reference_sentences, selected_regions):
    """
    Args:
        reference_sentences (List[List[str]]): outer list has len batch_size, inner list has len 29 (the inner list holds all reference phrases of a single image)
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
    """
    # array of shape [batch_size x 29]
    reference_sentences = np.asarray(reference_sentences)

    ref_sentences_for_selected_regions = reference_sentences[selected_regions]

    return ref_sentences_for_selected_regions.tolist()


def get_sents_for_normal_abnormal_selected_regions(region_is_abnormal, selected_regions, generated_sentences_for_selected_regions, reference_sentences_for_selected_regions):
    selected_region_is_abnormal = region_is_abnormal[selected_regions]
    # selected_region_is_abnormal is a bool array of shape [num_regions_selected_in_batch] that specifies if a selected region is abnormal (True) or normal (False)

    gen_sents_for_selected_regions = np.asarray(generated_sentences_for_selected_regions)
    ref_sents_for_selected_regions = np.asarray(reference_sentences_for_selected_regions)

    gen_sents_for_normal_selected_regions = gen_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
    gen_sents_for_abnormal_selected_regions = gen_sents_for_selected_regions[selected_region_is_abnormal].tolist()

    ref_sents_for_normal_selected_regions = ref_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
    ref_sents_for_abnormal_selected_regions = ref_sents_for_selected_regions[selected_region_is_abnormal].tolist()

    return (
        gen_sents_for_normal_selected_regions,
        gen_sents_for_abnormal_selected_regions,
        ref_sents_for_normal_selected_regions,
        ref_sents_for_abnormal_selected_regions,
    )


def evaluate_language_model(model, val_dl, tokenizer, writer, run_params, generated_sentences_and_reports_folder_path):
    epoch = run_params["epoch"]
    overall_steps_taken = run_params["overall_steps_taken"]
    log_file = run_params["log_file"]

    # whilst iterating over the validation loader, save the (all, normal, abnormal) generated and reference sentences in the respective lists
    # the list under the key "num_generated_sentences_per_image" will hold integers that represent how many sentences were generated for each image
    # this is useful to be able to get all generated and reference sentences that correspond to the same image
    # (since we append all generated and reference sentences to the "generated_sentences" and "reference_sentences" lists indiscriminately, this information would be lost otherwise)
    gen_and_ref_sentences = {
        "generated_sentences": [],
        "generated_sentences_normal_selected_regions": [],
        "generated_sentences_abnormal_selected_regions": [],
        "reference_sentences": [],
        "reference_sentences_normal_selected_regions": [],
        "reference_sentences_abnormal_selected_regions": [],
        "num_generated_sentences_per_image": []
    }

    # also save the generated and reference sentences on a per region basis
    for region_index, _ in enumerate(ANATOMICAL_REGIONS):
        gen_and_ref_sentences[region_index] = {
            "generated_sentences": [],
            "reference_sentences": []
        }

    # and of course the generated and reference reports, and additionally keep track of the generated sentences
    # that were removed because they were too similar to other generated sentences (only as a sanity-check/for writing to file)
    gen_and_ref_reports = {
        "generated_reports": [],
        "removed_similar_generated_sentences": [],
        "reference_reports": [],
    }

    # gen_sentences_with_corresponding_regions will be a list[list[tuple[str, str]]],
    # where len(outer_list) will be NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE
    # the inner list will correspond to a single generated report / single image and hold tuples of (region_name, generated_sentence),
    # i.e. each region that was selected for that single image, with the corresponding generated sentence (without any removal of similar generated sentences)
    #
    # gen_sentences_with_corresponding_regions will be used such that each generated sentences in a generated report can be directly attributed to a region
    # because this information gets lost when we concatenated generated sentences
    # this is only used to get more insights into the generated reports that are written to file
    gen_sentences_with_corresponding_regions = []

    # we also want to plot a couple of images
    num_batches_to_process_for_image_plotting = NUM_IMAGES_TO_PLOT // BATCH_SIZE

    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False

    # used in function get_generated_reports
    sentence_tokenizer = spacy.load("en_core_web_trf")
    bert_score = evaluate.load("bertscore")

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION):
            # since generating sentences takes some time, we limit the number of batches used to compute bleu/rouge-l/meteor
            if num_batch >= NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION:
                break

            images = batch["images"]  # shape [batch_size x 1 x 512 x 512]
            image_targets = batch["image_targets"]
            region_is_abnormal = batch["region_is_abnormal"].numpy()  # boolean array of shape [batch_size x 29]

            # List[List[str]] that holds the reference phrases. The inner list holds all reference phrases of a single image
            reference_sentences = batch["reference_sentences"]

            # List[str] that holds the reference report for the images in the batch
            reference_reports = batch["reference_reports"]

            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model.generate(
                        images.to(device, non_blocking=True),
                        max_length=MAX_NUM_TOKENS_GENERATE,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True

                    with open(log_file, "a") as f:
                        f.write("Generation:\n")
                        f.write(f"OOM at epoch {epoch}, batch number {num_batch}.\n")
                        f.write(f"Error message: {str(e)}\n\n")
                else:
                    raise e

            if oom:
                # free up memory
                torch.cuda.empty_cache()
                oom = False
                continue

            # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
            if output == -1:
                with open(log_file, "a") as f:
                    f.write("Generation:\n")
                    f.write(f"Empty region features before language model at epoch {epoch}, batch number {num_batch}.\n\n")

                continue
            else:
                # selected_regions is of shape [batch_size x 29] and is True for regions that should get a sentence
                beam_search_output, selected_regions, detections, class_detected = output
                selected_regions = selected_regions.detach().cpu().numpy()

            # generated_sents_for_selected_regions is a List[str] of length "num_regions_selected_in_batch"
            generated_sents_for_selected_regions = tokenizer.batch_decode(
                beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # filter reference_sentences to those that correspond to the generated_sentences for the selected regions.
            # reference_sents_for_selected_regions will therefore be a List[str] of length "num_regions_selected_in_batch"
            # (i.e. same length as generated_sents_for_selected_regions)
            reference_sents_for_selected_regions = get_ref_sentences_for_selected_regions(
                reference_sentences, selected_regions
            )

            (
                gen_sents_for_normal_selected_regions,
                gen_sents_for_abnormal_selected_regions,
                ref_sents_for_normal_selected_regions,
                ref_sents_for_abnormal_selected_regions,
            ) = get_sents_for_normal_abnormal_selected_regions(region_is_abnormal, selected_regions, generated_sents_for_selected_regions, reference_sents_for_selected_regions)

            generated_reports, removed_similar_generated_sentences = get_generated_reports(
                generated_sents_for_selected_regions,
                selected_regions,
                sentence_tokenizer,
                BERTSCORE_SIMILARITY_THRESHOLD,
                bert_score,
            )

            gen_and_ref_sentences["generated_sentences"].extend(generated_sents_for_selected_regions)
            gen_and_ref_sentences["generated_sentences_normal_selected_regions"].extend(gen_sents_for_normal_selected_regions)
            gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"].extend(gen_sents_for_abnormal_selected_regions)
            gen_and_ref_sentences["reference_sentences"].extend(reference_sents_for_selected_regions)
            gen_and_ref_sentences["reference_sentences_normal_selected_regions"].extend(ref_sents_for_normal_selected_regions)
            gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"].extend(ref_sents_for_abnormal_selected_regions)
            gen_and_ref_reports["generated_reports"].extend(generated_reports)
            gen_and_ref_reports["reference_reports"].extend(reference_reports)
            gen_and_ref_reports["removed_similar_generated_sentences"].extend(removed_similar_generated_sentences)

            update_gen_and_ref_sentences_for_regions(gen_and_ref_sentences, generated_sents_for_selected_regions, reference_sents_for_selected_regions, selected_regions)
            update_num_generated_sentences_per_image(gen_and_ref_sentences, selected_regions)

            if num_batch < NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE:
                update_gen_sentences_with_corresponding_regions(gen_sentences_with_corresponding_regions, generated_sents_for_selected_regions, selected_regions)

            if num_batch < num_batches_to_process_for_image_plotting:
                plot_detections_and_sentences_to_tensorboard(
                    writer,
                    num_batch,
                    overall_steps_taken,
                    images,
                    image_targets,
                    selected_regions,
                    detections,
                    class_detected,
                    reference_sentences,
                    generated_sents_for_selected_regions,
                )

    write_sentences_and_reports_to_file(
        gen_and_ref_sentences,
        gen_and_ref_reports,
        gen_sentences_with_corresponding_regions,
        generated_sentences_and_reports_folder_path,
        overall_steps_taken,
    )

    language_model_scores = compute_language_model_scores(gen_and_ref_sentences, gen_and_ref_reports)

    return language_model_scores
