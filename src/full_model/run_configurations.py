# define configurations for training run
RUN = 6
# can be useful to add additional information to run_config.txt file
RUN_COMMENT = """(Pre-)train full model minus language model. Change backbone to ResNet-50 pretrained on ImageNet. Use dynamic contrastive attention and gradient accumulation."""
PRETRAIN_WITHOUT_LM_MODEL = True
IMAGE_INPUT_SIZE = 512
NORMALITY_POOL_SIZE = 500
AGGREGATE_ATTENTION_NUM = 6
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 0.4
BATCH_SIZE = 16
EFFECTIVE_BATCH_SIZE = 64  # batch size after gradient accumulation
NUM_WORKERS = 8
EPOCHS = 20
LR = 1e-3
EVALUATE_EVERY_K_BATCHES = 500  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE_LR_SCHEDULER = 10  # number of evaluations to wait for val loss to reduce before lr is reduced by 1e-1
THRESHOLD_LR_SCHEDULER = 1e-3  # threshold for measuring the new optimum, to only focus on significant changes
NUM_BEAMS = 4
MAX_NUM_TOKENS_GENERATE = 300
NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE = 10  # save num_batches_of_... worth of generated sentences with their gt reference phrases to a txt file
NUM_SENTENCES_TO_GENERATE_FOR_EVALUATION = 300  # for evaluation of BLEU/BERTScore
NUM_IMAGES_TO_PLOT = 8
