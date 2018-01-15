VAR_LOCAL = True

if VAR_LOCAL:
    PREDICTOR = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    DATA_PATH = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_train'
    BASE_SAVE_LOCATION = '/home/gabi/PycharmProjects/visualizing-traits/data/training'
    TEMPLATE = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/cropped_landmark_template.txt'
else:
    PREDICTOR = '/home/gabras/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    DATA_PATH = '/vol/ccnlab-scratch1/gabras/chalearn_train/'
    BASE_SAVE_LOCATION = '/vol/ccnlab-scratch1/gabras/chalearn_train_aligned'
    TEMPLATE = '/home/gabras/visualizing-traits/src/align_face/cropped_landmark_template.txt'