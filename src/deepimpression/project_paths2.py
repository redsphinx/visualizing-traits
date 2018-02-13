ON_GPU = False
VAR_LOCAL = True


if VAR_LOCAL:
    TRAIN_DATA = '/home/gabi/PycharmProjects/visualizing-traits/data/training'
    TRAIN_LABELS = '/home/gabi/PycharmProjects/visualizing-traits/data/training/annotation_training.pkl'
    # TEST_DATA = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed'
    TEST_DATA = '/home/gabi/PycharmProjects/visualizing-traits/data/chalearn_test_aligned'
    TEST_LABELS = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/annotation_test.pkl'

    # model location
    # PRE_TRAINED = '/home/gabi/PycharmProjects/visualizing-traits/model/model'
    PRE_TRAINED = '/home/gabi/PycharmProjects/visualizing-traits/data/models/deepimpression_e_599'

    LOG = '/home/gabi/PycharmProjects/visualizing-traits/data/log.txt'
    CHALEARN_JPGS = '/home/gabi/PycharmProjects/visualizing-traits/data/chalearn_aligned_jpgs'
    MODEL_SAVES = '/home/gabi/PycharmProjects/visualizing-traits/data/models'
    TEST_LOG = 'shitty_test_log.txt'
    TRAIN_LOG = 'shitty_train_log.txt'
    VALIDATION_LOG = 'shitty_val_log.txt'


else:
    TRAIN_DATA = '/vol/ccnlab-scratch1/gabras/chalearn_train_aligned_all'
    VALIDATION_DATA = '/vol/ccnlab-scratch1/gabras/chalearn_validation_aligned/val-1'

    TRAIN_LABELS = '/scratch2/gabi/chalearn_aligned_jpgs/annotation_training.pkl'  # only on hinton
    # TEST_DATA = '/vol/ccnlab-scratch1/gabras/chalearn_compressed'
    TEST_DATA = '/vol/ccnlab-scratch1/gabras/chalearn_test_aligned'
    TEST_LABELS = '/vol/ccnlab-scratch1/gabras/chalearn_compressed/annotation_test.pkl'
    PRE_TRAINED = '/home/gabras/visualizing-traits/model/model'
    # LOG = '/vol/ccnlab-scratch1/gabras/log.txt'

    CHALEARN_JPGS = '/scratch2/gabi/chalearn_aligned_jpgs'  # only available from hinton
    CHALEARN_VALIDATION_JPGS = '/scratch2/gabi/chalearn_validation_jpgs'

    MODEL_SAVES = '/vol/ccnlab-scratch1/gabras/models'

    VALIDATION_LABELS = '/vol/ccnlab-scratch1/gabras/chalearn_validation/annotation_validation.pkl'

    VALIDATION_LOG = '/vol/ccnlab-scratch1/gabras/logs/validation_log.txt'

    TEST_LOG = '/vol/ccnlab-scratch1/gabras/logs/test_log.txt'

    TRAIN_LOG = '/vol/ccnlab-scratch1/gabras/logs/train_log.txt'