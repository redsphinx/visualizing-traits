VAR_LOCAL = True

if VAR_LOCAL:
    PREDICTOR = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    DATA_PATH = '/home/gabi/Documents/temp_datasets/chalearn_fi_17_compressed/%s'
    BASE_SAVE_LOCATION = '/home/gabi/PycharmProjects/visualizing-traits/data/training'
else:
    PREDICTOR = '/home/gabras/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    DATA_PATH = '/vol/ccnlab-scratch1/gabras/chalearn_train'
    BASE_SAVE_LOCATION = '/vol/ccnlab-scratch1/gabras/chalearn_train_aligned'
