import evaluate
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize
import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

import re
import tokenizers

gts = {
    "5": ["this is a test hello world"],
    "6": ["this is an example sentence"]
}

res = {
    "5": ["this is not a test hello there"],
    "6": ["testing out the bleu score"]
}


def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)
    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, _ = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


print(compute_scores(gts, res))


# bleu = Bleu(4)
# meteor = Meteor()
tokenizer = PTBTokenizer()

pred_1_str = "this is a test hello world"
ref_1_str = "this is not a test hello there"

pred_2_str = "this is an example sentence"
ref_2_str = "testing out the bleu score"

predictions, labels = {}, {}

predictions["5"] = [{"caption": pred_1_str}]
labels["5"] = [{"caption": ref_1_str}]
predictions["6"] = [{"caption": pred_2_str}]
labels["6"] = [{"caption": ref_2_str}]

print(predictions)
print(labels)

print("==============")
predictions = tokenizer.tokenize(predictions)
labels = tokenizer.tokenize(labels)
print(predictions)
print(labels)

# score, scores = bleu.compute_score(labels, predictions)
# print(score)
# # print(score[0])
# print()
# print(scores)

# print()

bleu_1 = evaluate.load("bleu")
bleu_2 = evaluate.load("bleu")
bleu_3 = evaluate.load("bleu")
bleu_4 = evaluate.load("bleu")
meteor_huggingface = evaluate.load("meteor")
rouge_huggingface = evaluate.load("rouge")

bleu_1_result = bleu_1.compute(predictions=[pred_1_str, pred_2_str], references=[ref_1_str, ref_2_str], max_order=1)["bleu"]
bleu_2_result = bleu_2.compute(predictions=[pred_1_str, pred_2_str], references=[ref_1_str, ref_2_str], max_order=2)["bleu"]
bleu_3_result = bleu_3.compute(predictions=[pred_1_str, pred_2_str], references=[ref_1_str, ref_2_str], max_order=3)["bleu"]
bleu_4_result = bleu_4.compute(predictions=[pred_1_str, pred_2_str], references=[ref_1_str, ref_2_str], max_order=4)["bleu"]
meteor_result = meteor_huggingface.compute(predictions=[pred_1_str, pred_2_str], references=[ref_1_str, ref_2_str])["meteor"]
rouge_result = rouge_huggingface.compute(predictions=[pred_1_str, pred_2_str], references=[ref_1_str, ref_2_str])
print(bleu_1_result)
print(bleu_2_result)
print(bleu_3_result)
print(bleu_4_result)
print(f"meteor: {meteor_result}")
print(f"rouge: {rouge_result}")

#######
#######

bleu = Bleu(4)
rouge = Rouge()
# predictions, labels = {}, {}

# pred_1 = [re.sub(' +', ' ', pred_1_str.replace(".", " ."))]
# ref_1 = [re.sub(' +', ' ', ref_1_str.replace(".", " ."))]
# pred_2 = [re.sub(' +', ' ', pred_2_str.replace(".", " ."))]
# ref_2 = [re.sub(' +', ' ', ref_2_str.replace(".", " ."))]

# predictions["5"] = pred_1
# labels["5"] = ref_1
# predictions["6"] = pred_2
# labels["6"] = ref_2

print()
score, scores = bleu.compute_score(labels, predictions)
print(score)
print()
score, scores = rouge.compute_score(labels, predictions)
print(score)
print()
# print(scores)

# # import torch
# # from torchmetrics.text.bert import BERTScore

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # bleu_1 = evaluate.load("bleu")
# # bleu_2 = evaluate.load("bleu")
# # bleu_3 = evaluate.load("bleu")
# # bleu_4 = evaluate.load("bleu")
# # meteor = evaluate.load("meteor")

# # print("Sleeping 5")
# # time.sleep(5)
# # print("Sleeping finished")
# # bertscore_torchmetric = BERTScore(device=device)
# # bert_score = evaluate.load("bertscore")

# # rouge = evaluate.load("rouge")
# # meteor = evaluate.load("meteor")

# # gen_sentences = ["I like to go swimming", "you have to pass", "you have to pass", "The cardiomediastinal silhouette is normal.", "This is a test."]
# # ref_sentences = ["the ball is blue", "you have to fail", "you need to pass the exam", "The cardiomediastinal and hilar silhouettes are normal.", "This is a test."]
# # gen_sentences = ["There is mild elevation. The heart size is normal. Lung volumes are low."]
# # ref_sentences = ["There is mild elevation. Lung volumes are low. The heart size is normal."]

# # bleu_1.add_batch(predictions=gen_sentences, references=ref_sentences)
# # bleu_2.add_batch(predictions=gen_sentences, references=ref_sentences)
# # bleu_3.add_batch(predictions=gen_sentences, references=ref_sentences)
# # bleu_4.add_batch(predictions=["The lungs are clear. There is no pleural effusion or pneumothorax. No acute cardiopulmonary process."], references=["Pulmonary vasculature is normal. Lungs are clear. No pleural effusion or pneumothorax is present. No acute cardiopulmonary abnormality."])
# # bert_score.add_batch(predictions=gen_sentences, references=ref_sentences)

# # rouge.add_batch(predictions=gen_sentences, references=ref_sentences)
# # meteor.add_batch(predictions=gen_sentences, references=ref_sentences)

# # bleu_1_result = bleu_1.compute(max_order=1)["bleu"]
# # bleu_2_result = bleu_2.compute(max_order=2)["bleu"]
# # bleu_3_result = bleu_3.compute(max_order=3)["bleu"]
# # bleu_4_result = bleu_4.compute(max_order=4)["bleu"]
# # print(bleu_4_result)
# # "The lungs are clear without consolidation, effusion, or edema."
# # "The cardiomediastinal and hilar silhouettes are normal."

# # gen_sents = ["There is no focal consolidation, effusion, or pneumothorax.", "The cardiac silhouette is normal.", "The spine is normal."]
# # ref_sents = ["No large effusion or pneumothorax.", "The heart size is normal.", "No abnormality in the spine."]

# # scores = [
# #     single_meteor_score(
# #         word_tokenize(ref), word_tokenize(pred)
# #     )
# #     for ref, pred in zip(ref_sents, gen_sents)
# # ]

# # print("mean", np.mean(scores))

# # bert_score_torchmetrics = bertscore_torchmetric(preds=gen_sents, target=ref_sents)['f1']
# # bert_score_result_distil = bert_score.compute(predictions=gen_sents, references=ref_sents, model_type="distilbert-base-uncased", device=device)
# # bert_score_result_roberta = bert_score.compute(lang="en", predictions=gen_sents, references=ref_sents)

# # print(f"bert_score torchmetrics f1: {bert_score_torchmetrics}")
# # print(f"bert_score torchmetrics f1: {type(bert_score_torchmetrics)}")
# # print(f"bert_score torchmetrics f1 average: {np.array(bert_score_torchmetrics).mean()}")
# # print(f"bert_score torchmetrics f1 average: {type(np.array(bert_score_torchmetrics).mean())}")
# # print(f"bert_score torchmetrics f1 average: {type(float(np.array(bert_score_torchmetrics).mean()))}")
# # # print(f"bert_score: {bert_score_result_distil}")
# # print(f"bert_score_roberta f1: {bert_score_result_roberta['f1']}")

# # result = meteor.compute(predictions=gen_sents, references=ref_sents)
# # print(result)


# # rouge_result = rouge.compute(rouge_types=["rougeL"], use_aggregator=True)["rougeL"]
# # print(float(rouge_result[1][2]))
# # rouge_result = [result[2] for result in rouge_result]
# # print(rouge_result)

# # print(f"bleu_1: {bleu_1_result}")
# # print(f"bleu_2: {bleu_2_result}")
# # print(f"bleu_3: {bleu_3_result}")
# # print(f"bleu_4: {bleu_4_result}")

# # print(f"rouge precision: {rouge_result[0]}")
# # print(f"rouge recall: {rouge_result[1]}")
# # print(f"rouge f: {rouge_result[2]}")  # <- the one that is actually reported
# # print(f"meteor: {meteor_result}")
