import os

import evaluate
from datasets import Dataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm

from gpt2 import DecoderModel
from custom_image_word_dataset import CustomImageWordDataset
from custom_collator import CustomCollatorWithPadding

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


def evaluate_model_on_metrics(model, val_dl, tokenizer):
    metrics_with_scores = {f"bleu_{i}": evaluate.load("bleu") for i in range(1, 5)}
    metrics_with_scores["bert_score"] = evaluate.load("bertscore")

    gen_and_ref_sentences_to_save_to_file = {
        "generated_sentences": [],
        "reference_sentences": []
    }

    # since generating sentences takes a long time (generating sentences for 36 regions takes around 8 seconds),
    # we only generate NUM_SENTENCES_TO_GENERATE sentences
    num_batches_to_process = NUM_SENTENCES_TO_GENERATE // BATCH_SIZE

    model.eval()
    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=num_batches_to_process):
            if num_batch >= num_batches_to_process:
                break

            # reference_phrases is a list of list of str
            _, _, image_hidden_states, reference_sentences = batch.values()

            image_hidden_states = image_hidden_states.to(device, non_blocking=True)  # shape (batch_size x image_hidden_dim) (with image_hidden_dim = 1024)

            beam_search_output = model.generate(image_hidden_states, max_length=MAX_NUM_TOKENS_GENERATE, num_beams=NUM_BEAMS, early_stopping=True)

            # generated_sentences is a list of str
            generated_sentences = tokenizer.batch_decode(beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            if num_batch < NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE:
                gen_and_ref_sentences_to_save_to_file["generated_sentences"].extend(generated_sentences)
                gen_and_ref_sentences_to_save_to_file["reference_sentences"].extend(reference_sentences)

            for score in metrics_with_scores.values():
                score.add_batch(predictions=generated_sentences, references=reference_sentences)

    write_sentences_to_file(gen_and_ref_sentences_to_save_to_file, generated_sentences_folder_path)

    metrics_with_final_scores = {}
    for score_name, score in metrics_with_scores.items():
        if score_name[:4] == "bleu":
            result = score.compute(max_order=int(score_name[-1]))
            metrics_with_final_scores[score_name] = result["bleu"]
        else:  # bert_score
            result = score.compute(lang="en", device=device)
            avg_precision = np.array(result["precision"]).mean()
            avg_recall = np.array(result["recall"]).mean()
            avg_f1 = np.array(result["f1"]).mean()

            metrics_with_final_scores["bertscore_precision"] = avg_precision
            metrics_with_final_scores["bertscore_recall"] = avg_recall
            metrics_with_final_scores["bertscore_f1"] = avg_f1

    return metrics_with_final_scores




def get_data_loaders(tokenizer, val_dataset_complete):
    custom_collate_val = CustomCollatorWithPadding(tokenizer=tokenizer, padding="longest", is_val=True)

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
    # tokenized datasets will consist of the columns "phrases", "input_ids", "attention_mask"
    tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

    return tokenized_val_dataset


def get_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_datasets_with_phrases():
    # path to the csv files specifying the train, val, test sets
    path_chest_imagenome_customized = "/u/home/tanida/datasets/chest-imagenome-dataset-customized-full"
    datasets_as_dfs = {dataset: os.path.join(path_chest_imagenome_customized, dataset) + ".csv" for dataset in ["valid"]}

    # only read in the phrases
    usecols = ["phrases"]
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
    model = DecoderModel()
    model.load_state_dict(torch.load(path_to_best_weights))
    model.eval()
    model.to(device, non_blocking=True)

    # get the dataset with the raw phrases before tokenization
    raw_val_dataset = get_datasets_with_phrases()

    tokenizer = get_tokenizer()

    # tokenize the raw dataset
    tokenized_val_dataset = get_tokenized_datasets(tokenizer, raw_val_dataset)

    val_dataset_complete = CustomImageWordDataset("val", tokenized_val_dataset)

    val_loader = get_data_loaders(tokenizer, val_dataset_complete)

    metrics = evaluate_model_on_metrics(model, val_loader, tokenizer)


if __name__ == "__main__":
    main()
