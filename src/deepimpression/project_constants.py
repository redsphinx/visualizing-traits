import numpy as np

# random seed
SEED = 6

# which gpu
DEVICE = '0'

# training parameters
BATCH_SIZE = 32
VAL_BATCH_SIZE = 200
EPOCHS = 900

# folder structure stuff
# NUMBER_TRAINING_FOLDERS = 75
# for testing, comment when not testing:
NUMBER_TRAINING_FOLDERS = 2
NUMBER_VALIDATION_FOLDERS = 25

# image specific stuff
SIDE = 192
NUM_VIDEOS = 20
NUM_FRAMES_PER_VIDEO = 10