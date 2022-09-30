NUM_EPOCHS = 8        #Number of epochs to train for
BATCH_SIZE = 18       #Change this depending on GPU memory
NUM_WORKERS = 4       #A value of 0 means the main process loads the data
LEARNING_RATE = 2e-5
LOG_EVERY = 200       #iterations after which to log status during training
VALID_NITER = 2000    #iterations after which to evaluate model and possibly save (if dev performance is a new max)
PRETRAIN_PATH = None  #path to pretrained model, such as BlueBERT or BioBERT
PAD_IDX = 0           #padding index as required by the tokenizer 

#CONDITIONS is a list of all 14 medical observations 
CONDITIONS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
              'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
              'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
              'Support Devices', 'No Finding']
CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}
