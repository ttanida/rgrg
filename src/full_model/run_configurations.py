# define configurations for training run
RUN = 2
# can be useful to add additional information to run_config.txt file
RUN_COMMENT = """Train full model with ResNet object detector and decoder that can only handle 1024 input dimension. Therefore use nn_for_modifying_region_features_dimension to bridge the gap."""
IMAGE_INPUT_SIZE = 512
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 0.2
BATCH_SIZE = 2
NUM_WORKERS = 12
EPOCHS = 20
LR = 1e-4
EVALUATE_EVERY_K_STEPS = 5000  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE_LR_SCHEDULER = 5  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1
THRESHOLD_LR_SCHEDULER = 1e-3  # threshold for measuring the new optimum, to only focus on significant changes
NUM_BEAMS = 4
MAX_NUM_TOKENS_GENERATE = 300
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 6  # save num_batches_of_... worth of generated sentences with their gt reference phrases to a txt file
NUM_SENTENCES_TO_GENERATE_FOR_EVALUATION = 300  # for evaluation of BLEU/BERTScore
NUM_IMAGES_TO_PLOT = 6
