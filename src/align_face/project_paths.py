VAR_LOCAL = True

if VAR_LOCAL:
    PREDICTOR = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    # for train data
    # DATA_PATH = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_train'
    # BASE_SAVE_LOCATION = '/home/gabi/Documents/tight_crop_everything/chalearn_training'
    # BASE_SAVE_LOCATION = '/home/gabi/PycharmProjects/visualizing-traits/data/training'
    # for test data
    # DATA_PATH = '/media/gabi/DATADRIVE1/datasets/chalearn_first_impressions_17'
    # DATA_PATH = '/media/gabi/DATADRIVE1/datasets/luc_pepper/Pepper'
    # DATA_PATH = '/home/gabi/kdenlive/luc'
    DATA_PATH = '/home/gabi/Documents/temp_datasets/img_align_celeba'
    # DATA_PATH = '/media/gabi/DATADRIVE1/datasets/beata_data/anouk/edits/webcam_selected_15_seconds'
    # BASE_SAVE_LOCATION = '/media/gabi/DATADRIVE1/datasets/chalearn_test_aligned'
    # for the 1 video in test that needs to be aligned
    # BASE_SAVE_LOCATION = '/home/gabi/PycharmProjects/visualizing-traits/'
    # BASE_SAVE_LOCATION = '/media/gabi/DATADRIVE1/datasets/luc_pepper/participants_aligned'
    # BASE_SAVE_LOCATION = '/home/gabi/Documents/temp_datasets/caleba_align_crop_224'
    BASE_SAVE_LOCATION = '/home/gabi/Documents/tight_crop_everything/celeba'
    # BASE_SAVE_LOCATION = '/media/gabi/DATADRIVE1/datasets/beata_data/anouk/edits/webcam_selected_15_seconds_aligned'
    # TEMPLATE = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/cropped_landmark_template.txt'
    TEMPLATE = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/landmarks_tight_crop_resize.txt'
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
    TIGHT_TEMPLATE = '/vol/ccnlab-scratch1/gabras/predictor-landmarks/cropped_landmark_template_tight.txt'
    # TEMPLATE = '/home/gabras/visualizing-traits/src/align_face/cropped_landmark_template.txt'