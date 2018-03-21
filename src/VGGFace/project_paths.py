VAR_LOCAL = True

if VAR_LOCAL:
    VGGFACE_CAFFE_MODEL = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE.caffemodel'
    VGGFACE_CAFFE_PROTO = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE_deploy.prototxt'
    CAFFE_PATH = '/home/gabi/Documents/caffe/python'
    CELEB_FACES = '/home/gabi/Documents/temp_datasets/caleba_align_crop_224'
    CELEB_FACES_FC6 = '/home/gabi/Documents/temp_datasets/celeba_fc6_features.txt'
else:
    pass