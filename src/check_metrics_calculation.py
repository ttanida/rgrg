import evaluate
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bleu_1 = evaluate.load("bleu")
# bleu_2 = evaluate.load("bleu")
# bleu_3 = evaluate.load("bleu")
# bleu_4 = evaluate.load("bleu")
bert_score = evaluate.load("bertscore")
print("Sleeping 5")
time.sleep(5)
print("Sleeping finished")
bert_score = evaluate.load("bertscore")

# rouge = evaluate.load("rouge")
# meteor = evaluate.load("meteor")

# gen_sentences = ["I like to go swimming", "you have to pass", "you have to pass", "The cardiomediastinal silhouette is normal.", "This is a test."]
# ref_sentences = ["the ball is blue", "you have to fail", "you need to pass the exam", "The cardiomediastinal and hilar silhouettes are normal.", "This is a test."]
# gen_sentences = ["There is mild elevation. The heart size is normal. Lung volumes are low."]
# ref_sentences = ["There is mild elevation. Lung volumes are low. The heart size is normal."]

# bleu_1.add_batch(predictions=gen_sentences, references=ref_sentences)
# bleu_2.add_batch(predictions=gen_sentences, references=ref_sentences)
# bleu_3.add_batch(predictions=gen_sentences, references=ref_sentences)
# bleu_4.add_batch(predictions=gen_sentences, references=ref_sentences)
# bert_score.add_batch(predictions=gen_sentences, references=ref_sentences)

# rouge.add_batch(predictions=gen_sentences, references=ref_sentences)
# meteor.add_batch(predictions=gen_sentences, references=ref_sentences)

# bleu_1_result = bleu_1.compute(max_order=1)["bleu"]
# bleu_2_result = bleu_2.compute(max_order=2)["bleu"]
# bleu_3_result = bleu_3.compute(max_order=3)["bleu"]
# bleu_4_result = bleu_4.compute(max_order=4)["bleu"]
# "The lungs are clear without consolidation, effusion, or edema."
# "The cardiomediastinal and hilar silhouettes are normal."

# gen_sent = " pulmonary pulmonary acute acute pulmonary acute pulmonary acute acute acute acute acute pulmonary acute acute acute acute acute acute acuteThe card acute acute acute acute acute acute acute acute acute acute pulmonary pulmonary acute pulmonary acute acute acute acute pulmonary pulmonary acute pulmonary acute pulmonary acute acute the pulmonary acute pulmonary acute pulmonary pulmonary pulmonary acute pulmonary acute acute acute acute acute acute acute pulmonary acute pulmonary acute acute pulmonary acute pulmonary pulmonary acute acute pulmonary acute acute pulmonary acute acute pulmonary acute acute pulmonary acute acute pulmonary acute acute acute pulmonary acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute pulmonary acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute pulmonary pulmonary pulmonary acute acute acute acute acute acute acute acute acute acute acute acute pulmonary pulmonaryThe acute acute acute acute acute pulmonary acute pulmonaryTheThe pulmonaryTheThe pulmonary acute acute acute acute acute acute acute acute acute acute pulmonaryTheTheTheThe pulmonary acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute pulmonaryTheTheTheTheThe the ."
# ref_sent = " the pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary acute pulmonary pulmonary acute pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary pulmonary acute acute pulmonary pulmonary acute pulmonary pulmonary pulmonary pulmonary pulmonary acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute acute the card the card the cardiac the the the the the the acute the acute acute acute the acute the pulmonary the acute the acute pulmonary the card the the cardiac the the acute the the acute the the the pulmonary the pulmonary the the pulmonary the pulmonary the pulmonary the acute the the acute the pulmonary the the acute the acute the the pulmonary the the the the pulmonary the pulmonary the the the acute the acute the pulmonary the pulmonary pulmonary pulmonary the the the the pulmonary the the the the the pulmonary the the the the the the the the acute pulmonary the the pulmonary the the the the the the the the acute the pulmonary the the pulmonary pulmonary the the the the the the the the the the the the the the the the the the the the the acute the card the the the the the the the the the the acute the the the the the acute the the the the acute the The."
# bert_score_result_distil = bert_score.compute(lang="en", predictions=[gen_sent], references=[ref_sent], model_type="distilbert-base-uncased")
# bert_score_result_roberta = bert_score.compute(lang="en", predictions=[gen_sent], references=[ref_sent])

# print(f"bert_score_distil: {bert_score_result_distil['f1'][0]}")
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
