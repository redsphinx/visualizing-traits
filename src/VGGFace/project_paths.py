VAR_LOCAL = True

if VAR_LOCAL:
    VGGFACE_CAFFE_MODEL = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE.caffemodel'
    VGGFACE_CAFFE_PROTO = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE_deploy.prototxt'
    CAFFE_PATH = '/home/gabi/Documents/caffe/python'
    # CELEB_FACES = '/home/gabi/Documents/temp_datasets/caleba_align_crop_224'
    # CELEB_FACES_FC6_TRAIN = '/home/gabi/Documents/temp_datasets/celeba_fc6_features_train.txt'
    # CELEB_FACES_FC6_TEST = '/home/gabi/Documents/temp_datasets/celeba_fc6_features_test.txt'
    CELEB_FACES = '/home/gabi/Documents/tight_crop_everything/celeba'
    CELEB_FACES_FC6_TRAIN = '/home/gabi/Documents/tight_crop_everything/celeba_vggface_features/fc6_train.txt'
    CELEB_FACES_FC6_TEST = '/home/gabi/Documents/tight_crop_everything/celeba_vggface_features/fc6_test.txt'
    CELEB_FACES_FC6_ALL = '/home/gabi/Documents/tight_crop_everything/celeba_vggface_features'

else:
    pass

