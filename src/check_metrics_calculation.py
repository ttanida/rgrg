from pycocoevalcap.meteor.meteor import Meteor
import re


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
    by lowercasing and separating punctuations from words.
    """
    sents_or_reports_converted = {}
    for num, text in enumerate(sents_or_reports):
        sents_or_reports_converted[str(num)] = [
            re.sub(" +", " ", text.replace(".", " .")).lower()
        ]

    return sents_or_reports_converted


gen_sents_path = "/u/home/tanida/region-guided-chest-x-ray-report-generation/src/full_model/evaluate_bbox_variations/gen_sents.txt"
ref_sents_path = "/u/home/tanida/region-guided-chest-x-ray-report-generation/src/full_model/evaluate_bbox_variations/ref_sents.txt"

with open(gen_sents_path) as f:
    lines = f.readlines()
    gen_sents = [line[:-1] for line in lines]

print(len(gen_sents))
for sent in gen_sents:
    if "\n" in sent:
        print("hello")



#     gen_sents = [line.rstrip() for line in lines]

# with open(ref_sents_path) as f:
#     lines = f.readlines()
#     ref_sents = [line.rstrip() for line in lines]

# for sent in gen_sents:
#     if "\n" in sent:
#         print("hello")

# print(len(gen_sents[:-5]))
# print(len(ref_sents))

# gen_sents = convert_for_pycoco_scorer(gen_sents[:-5])
# ref_sents = convert_for_pycoco_scorer(ref_sents)

# meteor_scorer = Meteor()
# result = meteor_scorer.compute_score(ref_sents, gen_sents)
# print(len(result))
