# define configurations for training run
RUN = 16
RUN_COMMENT = """Train full model for 2 epoch with pre-trained weights from run 14 (weights of 5 for binary classifiers), CA enabled. Weight of 2 for language model loss."""
PRETRAIN_WITHOUT_LM_MODEL = False
IMAGE_INPUT_SIZE = 512
NORMALITY_POOL_SIZE = 500
AGGREGATE_ATTENTION_NUM = 6
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 0.05
BATCH_SIZE = 2
EFFECTIVE_BATCH_SIZE = 64  # batch size after gradient accumulation
NUM_WORKERS = 10
EPOCHS = 2
LR = 1e-3
# how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
# EVALUATE_EVERY_K_BATCHES should be divisible by ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
EVALUATE_EVERY_K_BATCHES = 4800
PATIENCE_LR_SCHEDULER = 5  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1
THRESHOLD_LR_SCHEDULER = 1e-3  # threshold for measuring the new optimum, to only focus on significant changes
NUM_BEAMS = 4
MAX_NUM_TOKENS_GENERATE = 300
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 10  # save num_batches_of_... worth of generated sentences with their gt reference phrases to a txt file
NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE = 10  # save num_batches_of_... worth of generated reports with their gt reference reports to a txt file
NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION = 150  # for evaluation of bleu, rouge-l and meteor
NUM_IMAGES_TO_PLOT = 8
BERTSCORE_SIMILARITY_THRESHOLD = 0.9  # threshold for discarding generated sentences that are too similar
