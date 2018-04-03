VAR_LOCAL = True

if VAR_LOCAL:
    PREDICTOR = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    # for train data
    # DATA_PATH = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_train'
    # BASE_SAVE_LOCATION = '/home/gabi/PycharmProjects/visualizing-traits/data/training'
    # for test data
    # DATA_PATH = '/media/gabi/DATADRIVE1/datasets/chalearn_first_impressions_17'
    # DATA_PATH = '/media/gabi/DATADRIVE1/datasets/luc_pepper/Pepper'
<<<<<<< HEAD
<<<<<<< HEAD
    # DATA_PATH = '/home/gabi/kdenlive/luc'
    DATA_PATH = '/home/gabi/Documents/temp_datasets/img_align_celeba'
    # BASE_SAVE_LOCATION = '/media/gabi/DATADRIVE1/datasets/chalearn_test_aligned'
    # for the 1 video in test that needs to be aligned
    # BASE_SAVE_LOCATION = '/home/gabi/PycharmProjects/visualizing-traits/'
    # BASE_SAVE_LOCATION = '/media/gabi/DATADRIVE1/datasets/luc_pepper/participants_aligned'
    BASE_SAVE_LOCATION = '/home/gabi/Documents/temp_datasets/caleba_align_crop_224'
=======
=======
>>>>>>> cba6b702bc9ed7939779da4260b5ecf4897eb9ab
    DATA_PATH = '/home/gabi/kdenlive/luc'
    # BASE_SAVE_LOCATION = '/media/gabi/DATADRIVE1/datasets/chalearn_test_aligned'
    # for the 1 video in test that needs to be aligned
    # BASE_SAVE_LOCATION = '/home/gabi/PycharmProjects/visualizing-traits/'
    BASE_SAVE_LOCATION = '/media/gabi/DATADRIVE1/datasets/luc_pepper/participants_aligned'
<<<<<<< HEAD
>>>>>>> cba6b702bc9ed7939779da4260b5ecf4897eb9ab
=======
>>>>>>> cba6b702bc9ed7939779da4260b5ecf4897eb9ab
    TEMPLATE = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/cropped_landmark_template.txt'
else:
    # PREDICTOR = '/home/gabras/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    PREDICTOR = '/vol/ccnlab-scratch1/gabras/predictor-landmarks/shape_predictor_68_face_landmarks.dat'
    # training
    # DATA_PATH = '/vol/ccnlab-scratch1/gabras/chalearn_train/'
    # BASE_SAVE_LOCATION = '/vol/ccnlab-scratch1/gabras/chalearn_train_aligned'
    # testing
    # DATA_PATH = '/vol/ccnlab-scratch1/gabras/chalearn_compressed/'
    # BASE_SAVE_LOCATION = '/vol/ccnlab-scratch1/gabras/chalearn_test_aligned'
    # validation
    DATA_PATH = '/vol/ccnlab-scratch1/gabras/chalearn_validation'
    BASE_SAVE_LOCATION = '/vol/ccnlab-scratch1/gabras/chalearn_validation_aligned'
    TEMPLATE = '/vol/ccnlab-scratch1/gabras/predictor-landmarks/cropped_landmark_template.txt'
    # TEMPLATE = '/home/gabras/visualizing-traits/src/align_face/cropped_landmark_template.txt'
