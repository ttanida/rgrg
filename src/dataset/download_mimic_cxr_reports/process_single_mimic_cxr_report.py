import os
import re

import src.dataset.section_parser as sp

path_to_folder_with_mimic_cxr_reports = "path/to/folder"

SUBSTRING_TO_REMOVE_FROM_REPORT = "1. |2. |3. |4. |5. |6. |7. |8. |9."


def remove_duplicate_sentences(report):
    # remove the last period
    if report[-1] == ".":
        report = report[:-1]

    # dicts are insertion ordered as of Python 3.6
    sentence_dict = {sentence: None for sentence in report.split(". ")}

    report = ". ".join(sentence for sentence in sentence_dict)

    # add last period
    return report + "."


def list_rindex(list_, s):
    """Helper function: *last* matching element in a list"""
    return len(list_) - list_[-1::-1].index(s) - 1


def convert_mimic_cxr_report_to_single_string(study_txt_file):
    """
    Args:
        study_txt_file (str): e.g. "s56522600.txt"
    Returns:
        report (str or int): single str that contains information of the findings section,
        or -1 if neither were found.
    """
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    study_stem = study_txt_file[:-4]
    if study_stem in custom_section_names or study_stem in custom_indices:
        return -1

    path_to_report_txt_file = os.path.join(path_to_folder_with_mimic_cxr_reports, study_txt_file)
    with open(path_to_report_txt_file) as f:
        text = "".join(f.readlines())

    # split text into sections
    sections, section_names, _ = sp.section_text(text)

    report = ""

    for section_name in ["findings", "impression"]:
        if section_name in section_names:
            idx = list_rindex(section_names, section_name)
            section_text = sections[idx]
            report += section_text

    if len(report) == 0:
        return -1

    # remove substrings
    report = re.sub(SUBSTRING_TO_REMOVE_FROM_REPORT, "", report, flags=re.DOTALL)

    # remove unnecessary whitespaces
    report = " ".join(report.split())

    report = remove_duplicate_sentences(report)

    return report


def main():
    def convert_labels(preds_reports: list[list[int]]):
        """
        See doc string of update_clinical_efficacy_scores function for more details.
        Converts label 2 -> label 0 and label 3 -> label 1.
        """
        def convert_label(label: int):
            if label == 2:
                return 0
            elif label == 3:
                return 1
            else:
                return label

        preds_reports = [[convert_label(label) for label in condition_list] for condition_list in preds_reports]

        return preds_reports
    import csv
    import tempfile

    from src.CheXbert.src.label import label
    from src.path_datasets_and_weights import path_chexbert_weights

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_report_path = os.path.join(temp_dir, "report.csv")

        header = ["Report Impression"]
        report = "Severe pneumonia left lung, and a smaller region of pneumonia, right lower lobe, worsened on ___, subsequently stable. Small left pleural effusion is presumed. Heart size normal. Left jugular line ends in the low SVC. No pneumothorax."

        with open(csv_report_path, "w") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(header)
            csv_writer.writerow([report])

        preds = label(path_chexbert_weights, csv_report_path)

    print(preds)
    preds_new = convert_labels(preds)
    print(preds_new)


if __name__ == "__main__":
    main()
