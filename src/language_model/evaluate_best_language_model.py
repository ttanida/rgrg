import os

import evaluate
from datasets import Dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.language_model.language_model import LanguageModel
from src.language_model.custom_image_word_dataset import CustomImageWordDataset
from src.language_model.custom_collator import CustomCollatorWithPadding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_best_weights = "/u/home/tanida/runs/decoder_model/run_3/weights/val_loss_18.717_epoch_2.pth"
generated_sentences_folder_path = "/u/home/tanida/runs/decoder_model/run_3/weights/generated_sentences_for_best_weight"

PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.2
PERCENTAGE_OF_VAL_SET_TO_USE = 0.2
BATCH_SIZE = 16
NUM_WORKERS = 12
NUM_SENTENCES_TO_GENERATE = 5000
NUM_BEAMS = 4
MAX_NUM_TOKENS_GENERATE = 300
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 5


def write_metrics_to_file(metrics_sentence_generation, metrics_with_scores_dict):
    metrics_txt_file = os.path.join(generated_sentences_folder_path, "metrics")
    with open(metrics_txt_file, "w") as f:
        f.write("Sentence generation:\n")
        for metric, score in metrics_sentence_generation.items():
            f.write(f"\t{metric}: {score:.5f}\n")

        for reference_phrase_subset, metrics_with_scores in metrics_with_scores_dict.items():
            f.write(f"\n{reference_phrase_subset}:\n")
            for metric_name, score in metrics_with_scores.items():
                f.write(f"\t{metric_name}: {score:.5f}\n")


def print_out_metrics(metrics_sentence_generation, metrics_with_scores_dict):
    print("\nSentence generation:")
    for metric, score in metrics_sentence_generation.items():
        print(f"\t{metric}: {score:.5f}")

    for reference_phrase_subset, metrics_with_scores in metrics_with_scores_dict.items():
        print(f"\n{reference_phrase_subset}:")
        for metric_name, score in metrics_with_scores.items():
            print(f"\t{metric_name}: {score:.5f}")


def write_sentences_to_file(
        gen_and_ref_sentences_to_save_to_file,
        generated_sentences_folder_path):
    generated_sentences_txt_file = os.path.join(generated_sentences_folder_path, "generated_sentences")

    # generated_sentences is a list of str
    generated_sentences = gen_and_ref_sentences_to_save_to_file["generated_sentences"]

    # reference_sentences is a list of list of str
    reference_sentences = gen_and_ref_sentences_to_save_to_file["reference_sentences"]

    with open(generated_sentences_txt_file, "w") as f:
        for gen_sent, ref_sent in zip(generated_sentences, reference_sentences):
            f.write(f"Generated sentence: {gen_sent}\n")
            f.write(f"Reference sentence: {ref_sent[0]}\n\n")


def remove_empty_reference_sentences(generated_sentences, reference_sentences, is_abnormal_list):
    filtered_generated_sentences = []
    filtered_reference_sentences = []
    filtered_is_abnormal_list = []

    for gen_sentence, ref_sentence, is_abnormal in zip(generated_sentences, reference_sentences, is_abnormal_list):
        # only append non-empty reference sentences (and corresponding generated sentences and is_abnormal boolean variable)
        if ref_sentence[0] != "#":
            filtered_generated_sentences.append(gen_sentence)
            filtered_reference_sentences.append(ref_sentence)
            filtered_is_abnormal_list.append(is_abnormal)

    return filtered_generated_sentences, filtered_reference_sentences, filtered_is_abnormal_list


def compute_scores_for_sentence_generation(tp, fp, fn, tn, generated_sentences, reference_sentences):
    for gen_sentence, ref_sentence in zip(generated_sentences, reference_sentences):
        ref_sentence = ref_sentence[0]  # since it's originally a list of str
        # if a sentence == "#", it means it is empty
        if ref_sentence != "#" and gen_sentence != "#":
            tp += 1
        elif ref_sentence == "#" and gen_sentence != "#":
            fp += 1
        elif ref_sentence != "#" and gen_sentence == "#":
            fn += 1
        elif ref_sentence == "#" and gen_sentence == "#":
            tn += 1

    return tp, fp, fn, tn


def evaluate_model_on_metrics(model, val_dl, tokenizer):
    # compute metrics for all non-empty reference sentencens
    metrics_with_scores_all = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}
    metrics_with_scores_all["bert_score"] = evaluate.load("bertscore")

    # compute metrics for all non-empty reference sentences with findings (e.g. “There is pneumothorax.”)
    metrics_with_scores_with_findings = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}
    metrics_with_scores_with_findings["bert_score"] = evaluate.load("bertscore")

    # compute metrics for all non-empty reference sentences without findings (e.g. “There is no pneumothorax.”)
    metrics_with_scores_without_findings = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}
    metrics_with_scores_without_findings["bert_score"] = evaluate.load("bertscore")

    gen_and_ref_sentences_to_save_to_file = {
        "generated_sentences": [],
        "reference_sentences": []
    }

    # since generating sentences takes a long time (generating sentences for 29 regions takes around 8 seconds),
    # we only generate NUM_SENTENCES_TO_GENERATE sentences
    num_batches_to_process = NUM_SENTENCES_TO_GENERATE // BATCH_SIZE

    # to compute f1 / precision / recall score for if sentence was generated if there was a reference sentence
    tp = 0  # there was a reference, and also a sentence generated
    fp = 0  # there was no reference, but a sentence generated
    fn = 0  # there was a reference, but no sentence was generated
    tn = 0  # there was no reference, and also no generated sentence was generated

    model.eval()
    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=num_batches_to_process):
            if num_batch >= num_batches_to_process:
                break

            # reference_phrases is a list of list of str
            _, _, image_hidden_states, reference_sentences, is_abnormal_list = batch.values()

            image_hidden_states = image_hidden_states.to(device, non_blocking=True)  # shape (batch_size x image_hidden_dim) (with image_hidden_dim = 1024)

            beam_search_output = model.generate(image_hidden_states, max_length=MAX_NUM_TOKENS_GENERATE, num_beams=NUM_BEAMS, early_stopping=True)

            # generated_sentences is a list of str
            generated_sentences = tokenizer.batch_decode(beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            tp, fp, fn, tn = compute_scores_for_sentence_generation(tp, fp, fn, tn, generated_sentences, reference_sentences)

            # remove all empty reference sentences (and corresponding generated sentences and also remove corresponding boolean values in is_abnormal_list)
            filt_generated_sentences, filt_reference_sentences, filt_is_abnormal_list = remove_empty_reference_sentences(generated_sentences, reference_sentences, is_abnormal_list)

            if num_batch < NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE:
                gen_and_ref_sentences_to_save_to_file["generated_sentences"].extend(filt_generated_sentences)
                gen_and_ref_sentences_to_save_to_file["reference_sentences"].extend(filt_reference_sentences)

            # compute metrics for all non-empty reference sentences
            for score in metrics_with_scores_all.values():
                score.add_batch(predictions=filt_generated_sentences, references=filt_reference_sentences)

            # compute metrics for all non-empty reference sentences with abnormal findings and without abnormal findings
            gen_sentences_with_findings = []
            ref_sentences_with_findings = []
            gen_sentences_without_findings = []
            ref_sentences_without_findings = []

            for gen_sent, ref_sent, is_abnormal in zip(filt_generated_sentences, filt_reference_sentences, filt_is_abnormal_list):
                if is_abnormal:
                    gen_sentences_with_findings.append(gen_sent)
                    ref_sentences_with_findings.append(ref_sent)
                else:
                    gen_sentences_without_findings.append(gen_sent)
                    ref_sentences_without_findings.append(ref_sent)

            if len(ref_sentences_with_findings) != 0:
                for score in metrics_with_scores_with_findings.values():
                    score.add_batch(predictions=gen_sentences_with_findings, references=ref_sentences_with_findings)

            if len(ref_sentences_without_findings) != 0:
                for score in metrics_with_scores_without_findings.values():
                    score.add_batch(predictions=gen_sentences_without_findings, references=ref_sentences_without_findings)

    write_sentences_to_file(gen_and_ref_sentences_to_save_to_file, generated_sentences_folder_path)

    # compute final scores for sentence generation
    precision_sent_generation = tp / (tp + fp)
    recall_sentence_generation = tp / (tp + fn)
    f1_sent_generation = 2 * (precision_sent_generation * recall_sentence_generation) / (precision_sent_generation + recall_sentence_generation)

    metrics_sentence_generation = {}
    metrics_sentence_generation["precision_sent_generation"] = precision_sent_generation
    metrics_sentence_generation["recall_sentence_generation"] = recall_sentence_generation
    metrics_sentence_generation["f1_sent_generation"] = f1_sent_generation

    # collect metrics_with_scores dicts in 1 dict
    metrics_with_scores_dict = {"all": metrics_with_scores_all, "with_findings": metrics_with_scores_with_findings, "without_findings": metrics_with_scores_without_findings}

    for key, metrics_with_scores in metrics_with_scores_dict.items():
        temp = {}
        for score_name, score in metrics_with_scores.items():
            if score_name.startswith("bleu"):
                result = score.compute(max_order=int(score_name[-1]))
                temp[f"{score_name}"] = result["bleu"]
            else:  # bert_score
                result = score.compute(lang="en", device=device)
                avg_precision = np.array(result["precision"]).mean()
                avg_recall = np.array(result["recall"]).mean()
                avg_f1 = np.array(result["f1"]).mean()

                temp["bertscore_precision"] = avg_precision
                temp["bertscore_recall"] = avg_recall
                temp["bertscore_f1"] = avg_f1

        metrics_with_scores_dict[key] = temp

    return metrics_sentence_generation, metrics_with_scores_dict


def get_data_loaders(tokenizer, val_dataset_complete):
    custom_collate_val = CustomCollatorWithPadding(tokenizer=tokenizer, padding="longest", is_val=True, has_is_abnormal_column=True)

    val_loader = DataLoader(
        val_dataset_complete,
        collate_fn=custom_collate_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return val_loader


def get_tokenized_datasets(tokenizer, raw_val_dataset):
    def tokenize_function(example):
        phrase = example["phrases"]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"
        if len(phrase) == 0:
            phrase_with_special_tokens = bos_token + "#" + eos_token
        else:
            phrase_with_special_tokens = bos_token + phrase + eos_token
        return tokenizer(phrase_with_special_tokens, truncation=True, max_length=1024)

    # don't set batched=True, otherwise phrases that are empty will not be processed correctly
    # tokenized datasets will consist of the columns "phrases", "input_ids", "attention_mask", "is_abnormal"
    tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

    return tokenized_val_dataset


def get_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_datasets_with_phrases_and_is_abnormal():
    # path to the csv files specifying the val set
    path_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"
    datasets_as_dfs = {dataset: os.path.join(path_chest_imagenome_customized, dataset) + ".csv" for dataset in ["valid"]}

    # only read in the phrases and is_abnormal boolean variable
    usecols = ["phrases", "is_abnormal"]
    datasets_as_dfs = {
        dataset: pd.read_csv(csv_file_path, usecols=usecols, keep_default_na=False)
        for dataset, csv_file_path in datasets_as_dfs.items()
    }

    total_num_samples_val = len(datasets_as_dfs["valid"])

    # compute new number of samples for val
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    # limit the datasets to those new numbers
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

    return raw_val_dataset


def main():
    model = LanguageModel()
    model.load_state_dict(torch.load(path_to_best_weights))
    model.eval()
    model.to(device, non_blocking=True)

    # get the dataset with the raw phrases and is_abnormal boolean variable
    raw_val_dataset = get_datasets_with_phrases_and_is_abnormal()

    tokenizer = get_tokenizer()

    # tokenized_val_dataset has the columns "phrases", "input_ids", "attention_mask", "is_abnormal"
    tokenized_val_dataset = get_tokenized_datasets(tokenizer, raw_val_dataset)

    val_dataset_complete = CustomImageWordDataset("val", tokenized_val_dataset)

    val_loader = get_data_loaders(tokenizer, val_dataset_complete)

    metrics_sentence_generation, metrics_with_scores_dict = evaluate_model_on_metrics(model, val_loader, tokenizer)

    print_out_metrics(metrics_sentence_generation, metrics_with_scores_dict)
    write_metrics_to_file(metrics_sentence_generation, metrics_with_scores_dict)


if __name__ == "__main__":
    main()
