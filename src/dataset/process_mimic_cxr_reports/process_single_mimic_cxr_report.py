import os
import re

import section_parser as sp

path_to_folder_with_mimic_cxr_reports = "path/to/folder"

SUBSTRING_TO_REMOVE_FROM_REPORT = "FINDINGS:|IMPRESSION:|1. |2. |3. |4. |5. |6. |7. |8. |9."


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
        report (str or int): single str that contains information of the findings and impression sections,
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
