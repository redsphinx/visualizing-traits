ON_GPU = False
VAR_LOCAL = True

if VAR_LOCAL:
    TRAIN_DATA = '/home/gabi/PycharmProjects/visualizing-traits/data/training'
    TRAIN_LABELS = '/home/gabi/PycharmProjects/visualizing-traits/data/training/annotation_training.pkl'
else:
    TRAIN_DATA = '/vol/ccnlab-scratch1/gabras/chalearn_train_aligned'
    TRAIN_LABELS = '/vol/ccnlab-scratch1/gabras/chalearn_train/annotation_training.pkl'
