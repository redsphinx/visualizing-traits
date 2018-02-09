VAR_LOCAL = True

if VAR_LOCAL:
    PREDICTOR = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    # for train data
    # DATA_PATH = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_train'
    # BASE_SAVE_LOCATION = '/home/gabi/PycharmProjects/visualizing-traits/data/training'
    # for test data
    DATA_PATH = '/media/gabi/DATADRIVE1/datasets/chalearn_first_impressions_17'
    BASE_SAVE_LOCATION = '/media/gabi/DATADRIVE1/datasets/chalearn_test_aligned'
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
