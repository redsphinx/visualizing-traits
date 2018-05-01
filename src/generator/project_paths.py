VAR_LOCAL = True

if VAR_LOCAL:
    # CELEB_FACES_FC6 = '/home/gabi/Documents/temp_datasets/celeba_fc6_features.txt'
    # CELEB_FACES_FC6_TRAIN = '/home/gabi/Documents/temp_datasets/celeba_fc6_features_train.txt'
    # CELEB_FACES_FC6_TEST = '/home/gabi/Documents/temp_datasets/celeba_fc6_features_test.txt'

    CELEB_FACES_FC6_TRAIN = '/home/gabi/Documents/tight_crop_everything/celeba_vggface_features/fc6_train.txt'
    CELEB_FACES_FC6_TEST = '/home/gabi/Documents/tight_crop_everything/celeba_vggface_features/fc6_test.txt'

    CELEB_DATA_ALIGNED = '/home/gabi/Documents/temp_datasets/caleba_align_crop_224'
    RECONSTRUCTION_FOLDER = '/home/gabi/Documents/temp_datasets/celeba_reconstruction'
    # ORIGINAL = '/home/gabi/Documents/temp_datasets/celeba_sample_10'
    MODEL_SAVES = '/home/gabi/Documents/temp_datasets/generator_models'
    FC6_TEST_H5 = '/home/gabi/PycharmProjects/visualizing-traits/src/generator/test.h5'
    FC6_TRAIN_H5 = '/home/gabi/PycharmProjects/visualizing-traits/src/generator/train.h5'
    # FC6_TEST_H5 = '/home/gabi/Documents/tight_crop_everything/celeba_vggface_features/test.h5'
    # FC6_TRAIN_H5 = '/home/gabi/Documents/tight_crop_everything/celeba_vggface_features/train.h5'
    # TRAIN_LOG = 'train_log.txt'
    TEST_RECONSTRUCTION_FOLDER = '/home/gabi/Documents/temp_datasets/test_celeba_reconstruction'
else:
    pass