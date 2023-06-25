"""
Specify the checkpoint_path, images_paths and generated_reports_txt_path in the main function
before running this script.

If you encounter any spacy-related errors, try upgrading spacy to version 3.5.3 and spacy-transformers to version 1.2.5
pip install -U spacy
pip install -U spacy-transformers
"""

from collections import defaultdict

import albumentations as A
import cv2
import evaluate
import spacy
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.train_full_model import get_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERTSCORE_SIMILARITY_THRESHOLD = 0.9
IMAGE_INPUT_SIZE = 512
MAX_NUM_TOKENS_GENERATE = 300
NUM_BEAMS = 4
mean = 0.471  # see get_transforms in src/dataset/compute_mean_std_dataset.py
std = 0.302


def write_generated_reports_to_txt(images_paths, generated_reports, generated_reports_txt_path):
    with open(generated_reports_txt_path, "w") as f:
        for image_path, report in zip(images_paths, generated_reports):
            f.write(f"Image path: {image_path}\n")
            f.write(f"Generated report: {report}\n\n")
            f.write("=" * 30)
            f.write("\n\n")


def remove_duplicate_generated_sentences(generated_report, bert_score, sentence_tokenizer):
    def check_gen_sent_in_sents_to_be_removed(gen_sent, similar_generated_sents_to_be_removed):
        for lists_of_gen_sents_to_be_removed in similar_generated_sents_to_be_removed.values():
            if gen_sent in lists_of_gen_sents_to_be_removed:
                return True

        return False

    # since different (closely related) regions can have the same generated sentence, we first remove exact duplicates

    # use sentence tokenizer to separate the generated sentences
    gen_sents = sentence_tokenizer(generated_report).sents

    # convert spacy.tokens.span.Span object into str by using .text attribute
    gen_sents = [sent.text for sent in gen_sents]

    # remove exact duplicates using a dict as an ordered set
    # note that dicts are insertion ordered as of Python 3.7
    gen_sents = list(dict.fromkeys(gen_sents))

    # there can still be generated sentences that are not exact duplicates, but nonetheless very similar
    # e.g. "The cardiomediastinal silhouette is normal." and "The cardiomediastinal silhouette is unremarkable."
    # to remove these "soft" duplicates, we use bertscore

    # similar_generated_sents_to_be_removed maps from one sentence to a list of similar sentences that are to be removed
    similar_generated_sents_to_be_removed = defaultdict(list)

    for i in range(len(gen_sents)):
        gen_sent_1 = gen_sents[i]

        for j in range(i + 1, len(gen_sents)):
            if check_gen_sent_in_sents_to_be_removed(gen_sent_1, similar_generated_sents_to_be_removed):
                break

            gen_sent_2 = gen_sents[j]
            if check_gen_sent_in_sents_to_be_removed(gen_sent_2, similar_generated_sents_to_be_removed):
                continue

            bert_score_result = bert_score.compute(
                lang="en", predictions=[gen_sent_1], references=[gen_sent_2], model_type="distilbert-base-uncased"
            )

            if bert_score_result["f1"][0] > BERTSCORE_SIMILARITY_THRESHOLD:
                # remove the generated similar sentence that is shorter
                if len(gen_sent_1) > len(gen_sent_2):
                    similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                else:
                    similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

    generated_report = " ".join(
        sent
        for sent in gen_sents
        if not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
    )

    return generated_report


def convert_generated_sentences_to_report(generated_sents_for_selected_regions, bert_score, sentence_tokenizer):
    generated_report = " ".join(sent for sent in generated_sents_for_selected_regions)

    generated_report = remove_duplicate_generated_sentences(generated_report, bert_score, sentence_tokenizer)
    return generated_report


def get_report_for_image(model, image_tensor, tokenizer, bert_score, sentence_tokenizer):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.generate(
            image_tensor.to(device, non_blocking=True),
            max_length=MAX_NUM_TOKENS_GENERATE,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

    beam_search_output, _, _, _ = output

    generated_sents_for_selected_regions = tokenizer.batch_decode(
        beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )  # list[str]

    generated_report = convert_generated_sentences_to_report(
        generated_sents_for_selected_regions, bert_score, sentence_tokenizer
    )  # str

    return generated_report


def get_image_tensor(image_path):
    # cv2.imread by default loads an image with 3 channels
    # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # shape (3056, 2544)

    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    transform = val_test_transforms(image=image)
    image_transformed = transform["image"]  # shape (1, 512, 512)
    image_transformed_batch = image_transformed.unsqueeze(0)  # shape (1, 1, 512, 512)

    return image_transformed_batch


def get_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
    )

    # if there is a key error when loading checkpoint, try uncommenting down below
    # since depending on the torch version, the state dicts may be different
    # checkpoint["model"]["object_detector.rpn.head.conv.weight"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.weight")
    # checkpoint["model"]["object_detector.rpn.head.conv.bias"] = checkpoint["model"].pop("object_detector.rpn.head.conv.0.0.bias")
    model = ReportGenerationModel(pretrain_without_lm_model=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device, non_blocking=True)
    model.eval()

    del checkpoint

    return model


def main():
    checkpoint_path = ".../___.pt"
    model = get_model(checkpoint_path)

    print("Model instantiated.")

    # paths to the images that we want to generate reports for
    images_paths = [
        ".../___.jpg",
        ".../___.jpg",
        ".../___.jpg",
    ]

    generated_reports_txt_path = ".../___.txt"
    generated_reports = []

    bert_score = evaluate.load("bertscore")
    sentence_tokenizer = spacy.load("en_core_web_trf")
    tokenizer = get_tokenizer()

    # if you encounter a spacy-related error, try upgrading spacy to version 3.5.3 and spacy-transformers to version 1.2.5
    # pip install -U spacy
    # pip install -U spacy-transformers

    for image_path in tqdm(images_paths):
        image_tensor = get_image_tensor(image_path)  # shape (1, 1, 512, 512)
        generated_report = get_report_for_image(model, image_tensor, tokenizer, bert_score, sentence_tokenizer)
        generated_reports.append(generated_report)

    write_generated_reports_to_txt(images_paths, generated_reports, generated_reports_txt_path)


if __name__ == "__main__":
    main()
