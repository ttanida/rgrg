import evaluate
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bleu_1 = evaluate.load("bleu")
# bleu_2 = evaluate.load("bleu")
# bleu_3 = evaluate.load("bleu")
bleu_4 = evaluate.load("bleu")

# print("Sleeping 5")
# time.sleep(5)
# print("Sleeping finished")
# bert_score = evaluate.load("bertscore")

# rouge = evaluate.load("rouge")
# meteor = evaluate.load("meteor")

# gen_sentences = ["I like to go swimming", "you have to pass", "you have to pass", "The cardiomediastinal silhouette is normal.", "This is a test."]
# ref_sentences = ["the ball is blue", "you have to fail", "you need to pass the exam", "The cardiomediastinal and hilar silhouettes are normal.", "This is a test."]
# gen_sentences = ["There is mild elevation. The heart size is normal. Lung volumes are low."]
# ref_sentences = ["There is mild elevation. Lung volumes are low. The heart size is normal."]

# bleu_1.add_batch(predictions=gen_sentences, references=ref_sentences)
# bleu_2.add_batch(predictions=gen_sentences, references=ref_sentences)
# bleu_3.add_batch(predictions=gen_sentences, references=ref_sentences)
# bleu_4.add_batch(predictions=["The lungs are clear. There is no pleural effusion or pneumothorax. No acute cardiopulmonary process."], references=["Pulmonary vasculature is normal. Lungs are clear. No pleural effusion or pneumothorax is present. No acute cardiopulmonary abnormality."])
# bert_score.add_batch(predictions=gen_sentences, references=ref_sentences)

# rouge.add_batch(predictions=gen_sentences, references=ref_sentences)
# meteor.add_batch(predictions=gen_sentences, references=ref_sentences)

# bleu_1_result = bleu_1.compute(max_order=1)["bleu"]
# bleu_2_result = bleu_2.compute(max_order=2)["bleu"]
# bleu_3_result = bleu_3.compute(max_order=3)["bleu"]
# bleu_4_result = bleu_4.compute(max_order=4)["bleu"]
# print(bleu_4_result)
# "The lungs are clear without consolidation, effusion, or edema."
# "The cardiomediastinal and hilar silhouettes are normal."

# gen_sent = 'There is no focal consolidation, effusion, or pneumothorax.'
# ref_sent = 'No large effusion or pneumothorax.'
# bert_score_result_distil = bert_score.compute(lang="en", predictions=[gen_sent], references=[ref_sent], model_type="distilbert-base-uncased", device=device)
# bert_score_result_roberta = bert_score.compute(lang="en", predictions=[gen_sent], references=[ref_sent])

# print(f"bert_score: {bert_score_result_distil['f1'][0]}")
# print(f"bert_score_roberta: {bert_score_result_roberta['f1']}")


# rouge_result = rouge.compute(rouge_types=["rougeL"], use_aggregator=True)["rougeL"]
# print(float(rouge_result[1][2]))
# rouge_result = [result[2] for result in rouge_result]
# print(rouge_result)

# print(f"bleu_1: {bleu_1_result}")
# print(f"bleu_2: {bleu_2_result}")
# print(f"bleu_3: {bleu_3_result}")
# print(f"bleu_4: {bleu_4_result}")

# print(f"rouge precision: {rouge_result[0]}")
# print(f"rouge recall: {rouge_result[1]}")
# print(f"rouge f: {rouge_result[2]}")  # <- the one that is actually reported
# print(f"meteor: {meteor_result}")
