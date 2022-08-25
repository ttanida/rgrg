import evaluate

bleu_1 = evaluate.load("bleu")
bleu_2 = evaluate.load("bleu")
bleu_3 = evaluate.load("bleu")
bleu_4 = evaluate.load("bleu")
bert_score = evaluate.load("bertscore")

gen_sentences = ["I like to go swimming", "you have to pass"]
ref_sentences = ["the ball is blue", "you have to fail"]

bleu_1.add_batch(predictions=gen_sentences, references=ref_sentences)
bleu_2.add_batch(predictions=gen_sentences, references=ref_sentences)
bleu_3.add_batch(predictions=gen_sentences, references=ref_sentences)
bleu_4.add_batch(predictions=gen_sentences, references=ref_sentences)
bert_score.add_batch(predictions=gen_sentences, references=ref_sentences)

bleu_1_result = bleu_1.compute(max_order=1)["bleu"]
bleu_2_result = bleu_2.compute(max_order=2)["bleu"]
bleu_3_result = bleu_3.compute(max_order=3)["bleu"]
bleu_4_result = bleu_4.compute(max_order=4)["bleu"]
bert_score_result = bert_score.compute(lang="en")

print(f"bleu_1: {bleu_1_result}")
print(f"bleu_2: {bleu_2_result}")
print(f"bleu_3: {bleu_3_result}")
print(f"bleu_4: {bleu_4_result}")
print(f"bert_score: {bert_score_result}")
