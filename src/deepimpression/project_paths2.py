ON_GPU = False
VAR_LOCAL = True


if VAR_LOCAL:
    TRAIN_DATA = '/home/gabi/PycharmProjects/visualizing-traits/data/training'
    TRAIN_LABELS = '/home/gabi/PycharmProjects/visualizing-traits/data/training/annotation_training.pkl'
    # TEST_DATA = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed'
    TEST_DATA = '/home/gabi/PycharmProjects/visualizing-traits/data/chalearn_test_aligned'
    TEST_LABELS = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/annotation_test.pkl'

    VALIDATION_LABELS = '/media/gabi/DATADRIVE1/datasets/chalearn_validation/annotation_validation.pkl'

    # model location
    # PRE_TRAINED = '/home/gabi/PycharmProjects/visualizing-traits/model/model'
<<<<<<< HEAD
<<<<<<< HEAD
    # PRE_TRAINED = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/deepimpression_e_899'
    PRE_TRAINED = '/home/gabi/Downloads/deepimpression_e_899'
    # PRE_TRAINED = '/media/gabi/DATADRIVE1/datasets/chalearn_face_models/transpose/deepimpression_e_899'
=======
    PRE_TRAINED = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/deepimpression_e_899'
>>>>>>> cba6b702bc9ed7939779da4260b5ecf4897eb9ab
=======
    PRE_TRAINED = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/deepimpression_e_899'
>>>>>>> cba6b702bc9ed7939779da4260b5ecf4897eb9ab

    LOG = '/home/gabi/PycharmProjects/visualizing-traits/data/log.txt'
    CHALEARN_JPGS = '/home/gabi/PycharmProjects/visualizing-traits/data/chalearn_aligned_jpgs'
    MODEL_SAVES = '/home/gabi/PycharmProjects/visualizing-traits/data/models'
    TEST_LOG = 'shitty_test_log.txt'
    TRAIN_LOG = 'shitty_train_log.txt'
    VALIDATION_LOG = 'shitty_val_log.txt'

<<<<<<< HEAD
<<<<<<< HEAD
    LUC_LABELS = '/media/gabi/DATADRIVE1/datasets/luc_pepper/ground_truth.csv'
=======
    LUC_LABELS = '/home/gabi/Downloads/ground_truth.csv'
>>>>>>> cba6b702bc9ed7939779da4260b5ecf4897eb9ab
=======
    LUC_LABELS = '/home/gabi/Downloads/ground_truth.csv'
>>>>>>> cba6b702bc9ed7939779da4260b5ecf4897eb9ab
    LUC_VIDEOS = '/media/gabi/DATADRIVE1/datasets/luc_pepper/participants_aligned'

    LUC_LOG = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/log.txt'

<<<<<<< HEAD
<<<<<<< HEAD
    LUC_TRAIT_LOG = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/rand_ord_trait_899_2.csv'
    LUC_PRED_ID = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/rand_ord_pred_id_899_2.csv'

    LUC_TRANSPOSE_TRAIT_LOG = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/transpose_trait_log_3.csv'
    LUC_TRANSPOSE_PRED_ID = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/transpose_pred_id_3.csv'
    LUC_RAND_ID = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/rand_id.csv'
=======
>>>>>>> cba6b702bc9ed7939779da4260b5ecf4897eb9ab
=======
>>>>>>> cba6b702bc9ed7939779da4260b5ecf4897eb9ab

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

    TRAIN_PRETRAINED = '/vol/ccnlab-scratch1/gabras/models/deepimpression_e_399'