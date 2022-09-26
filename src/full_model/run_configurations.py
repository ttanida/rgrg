"""
Define configurations for training run.

There are 3 different settings for the PRETRAIN_WITHOUT_LM_MODEL and PRETRAIN_LM_MODEL flags:

(1) PRETRAIN_WITHOUT_LM_MODEL = True and PRETRAIN_LM_MODEL = False

In this case, only the object detector and the 2 binary classifiers are trained in the full model,
with the language model (as the last component) being fully excluded from the model architecture.
This setting is for pre-training the 2 binary classifiers, since it's assumed that the object detector
was already trained separately in object_detector/training_script_object_detector.py

(2) PRETRAIN_WITHOUT_LM_MODEL = False and PRETRAIN_LM_MODEL = True

In this case, all 4 components (object detector + 2 binary classifiers + language model) are included in the model architecture,
but only the language model weights are updated. This setting is for pre-training the language model.

(3) PRETRAIN_WITHOUT_LM_MODEL = False and PRETRAIN_LM_MODEL = False

In this case, all 4 components are included in the model architecture, and all 4 are trained.
This setting is for training the full model.

--------
Ideally, the 3 settings should be gone through in order, i.e. first pre-training the binary classifiers,
then pre-training the language model, then training the full model.

Note that the setting of PRETRAIN_WITHOUT_LM_MODEL = True and PRETRAIN_LM_MODEL = True is undefined and thus should not be used.
"""
RUN = 29
RUN_COMMENT = """Pre-train full model minus language model."""
PRETRAIN_WITHOUT_LM_MODEL = True
PRETRAIN_LM_MODEL = False
IMAGE_INPUT_SIZE = 512
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 0.05
BATCH_SIZE = 8
EFFECTIVE_BATCH_SIZE = 64  # batch size after gradient accumulation
NUM_WORKERS = 10
EPOCHS = 20
LR = 1e-4
# how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
# EVALUATE_EVERY_K_BATCHES should be divisible by ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
EVALUATE_EVERY_K_BATCHES = 2400
PATIENCE_LR_SCHEDULER = 10  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1
THRESHOLD_LR_SCHEDULER = 1e-3  # threshold for measuring the new optimum, to only focus on significant changes
FACTOR_LR_SCHEDULER = 0.5
COOLDOWN_LR_SCHEDULER = 5
NUM_BEAMS = 4
MAX_NUM_TOKENS_GENERATE = 300
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 10  # save num_batches_of_... worth of generated sentences with their gt reference phrases to a txt file
NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE = 10  # save num_batches_of_... worth of generated reports with their gt reference reports to a txt file
NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION = 100  # for evaluation of bleu, rouge-l and meteor
NUM_IMAGES_TO_PLOT = 8
BERTSCORE_SIMILARITY_THRESHOLD = 0.9  # threshold for discarding generated sentences that are too similar
WEIGHT_OBJECT_DETECTOR_LOSS = 1
WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS = 5
WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS = 5
WEIGHT_LANGUAGE_MODEL_LOSS = 2
