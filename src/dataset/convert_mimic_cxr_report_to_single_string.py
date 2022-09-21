import os
import re
import webbrowser

path_to_folder_with_mimic_cxr_reports = "path/to/folder"

SUBSTRING_TO_REMOVE_FROM_REPORT = "FINDINGS:|IMPRESSION:|1. |2. |3. |4. |5. |6. |7. |8. |9."


def download_single_report(patient: str, study: str):
    """
    Downloads single report specified by patient and study.

    Args:
        patient (str): e.g. "p10000935"
        study (str): e.g. "s56522600"
    """
    link = f"https://physionet.org/files/mimic-cxr/2.0.0/files/{patient[:3]}/{patient}/{study}.txt?download"
    webbrowser.open(link)


def convert_mimic_cxr_report_to_single_str(study: str):
    """
    Each study in the MIMIC-CXR dataset has an accompanying report in text file format, e.g. "s56522600.txt"

    These text files usually have the sections: examination, indication, technique, comparison, findings and impression.

    To get the final report, we concatenate the findings and impression sections.

    To do this, we follow these steps:
        1. The findings and impression sections follow each other and are always the last 2 sections of a report.
        Therefore, once we find the findings section marked by the capitalized words "FINDINGS:", we concatenate the
        subsequent, remaining lines of the text file as the report.
        2. There are some cases where the findings section is missing, and there is only an impression section.
        Therefore, if 1. fails, then we find the impression section marked by the capitalized words "IMPRESSION:"
        and concatenate the subsequent, remaining lines as the report.
        3. In very rare cases, there is neither a findings nor a impression section. But these reports tend to have
        their actual findings in the last paragraph of the report (without it being explicitly called findings).
        Therefore, in this case, we take the last paragraph as the report.
        4. Also, in very rase cases, the impression section is followed up by a notification or recommendation section.
        A notification could e.g. be: "Findings discussed with Dr.___ by Dr.___ ___ telephone at 8:05am" and a recommendation
        could be "Recommend obtaining PA and lateral chest radiograph.". These sections are removed.

    Args:
        study (str): e.g. "s56522600.txt"
    Returns:
        report (str): single str that contains information of the findings and impression sections
    """
    path_to_report_txt_file = os.path.join(path_to_folder_with_mimic_cxr_reports, study)
    with open(path_to_report_txt_file) as f:
        lines = f.readlines()

    report = None

    for num_line, line in enumerate(lines):
        if "FINDINGS" in line:
            remaining_lines = lines[num_line:]
            # follow step 1 (see doc string)
            report = get_report(remaining_lines)
            break
        elif "IMPRESSION" in line:
            remaining_lines = lines[num_line:]
            # follow step 2 (see doc string)
            # if we find "IMPRESSION" in a line, then that means there is no "FINDINGS" section in the report,
            # since "FINDINGS" always comes before "IMPRESSION"
            report = get_report(remaining_lines)
            break

    # follow step 3 (see doc string)
    if report is None:
        remaining_lines = get_last_paragraph(lines)
        report = get_report(remaining_lines)

    return report


def get_report(remaining_lines):
    # concatenate the lines to 1 string
    report = " ".join(remaining_lines)

    # see step 4 of doc string
    report = remove_notification_recommendation(report)

    # remove substrings
    report = re.sub(SUBSTRING_TO_REMOVE_FROM_REPORT, "", report, flags=re.DOTALL)

    # remove unnecessary whitespaces
    report = " ".join(report.split())

    report = remove_duplicate_sentences(report)

    return report


def remove_notification_recommendation(report):
    for keyword in ["NOTIFICATION", "RECOMMENDATION"]:
        if keyword in report:
            start_index = report.find(keyword)
            report = report[:start_index]

    return report


def remove_duplicate_sentences(report):
    # remove the last period
    if report[-1] == ".":
        report = report[:-1]

    # dicts are insertion ordered as of Python 3.6
    sentence_dict = {sentence: None for sentence in report.split(". ")}

    report = ". ".join(sentence for sentence in sentence_dict)

    # add last period
    return report + "."


def get_last_paragraph(lines):
    """
    Determine the lines that make up the last paragraph by:
        1. iterating through the lines in reverse
        2. finding first line that is only "\n" (after being stripped of spaces)
        3. but also making sure that this line came after lines that had text in them,
        since theoretically the very first line of the (reversed) iteration could be a linebreak
    """
    text_found = False
    for num_line, line in enumerate(reversed(lines)):
        if re.search("[a-zA-Z]", line):
            text_found = True
        if text_found and line.strip(" ") == "\n":
            remaining_lines = lines[-num_line:]
            return remaining_lines
