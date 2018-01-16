ON_GPU = False
VAR_LOCAL = True

if VAR_LOCAL:
    TRAIN_DATA = '/home/gabi/PycharmProjects/visualizing-traits/data/training'
    TRAIN_LABELS = '/home/gabi/PycharmProjects/visualizing-traits/data/training/annotation_training.pkl'
    TEST_DATA = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed'
    TEST_LABELS = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/annotation_test.pkl'
    PRE_TRAINED = '/home/gabi/PycharmProjects/visualizing-traits/model/model'
    LOG = '/home/gabi/PycharmProjects/visualizing-traits/data/log.txt'
else:
    TRAIN_DATA = '/vol/ccnlab-scratch1/gabras/chalearn_train_aligned_all'
    TRAIN_LABELS = '/vol/ccnlab-scratch1/gabras/chalearn_train/annotation_training.pkl'
    TEST_DATA = '/vol/ccnlab-scratch1/gabras/chalearn_compressed'
    TEST_LABELS = '/vol/ccnlab-scratch1/gabras/chalearn_compressed/annotation_test.pkl'
    PRE_TRAINED = ''
    LOG = '/vol/ccnlab-scratch1/gabras/log.txt'
