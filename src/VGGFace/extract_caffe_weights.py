# just checking if the caffe model is working using caffe
import sys
from scipy import ndimage
import numpy as np
import os
import project_paths as pp

# VGGFACE_CAFFE_MODEL = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE.caffemodel'
# VGGFACE_CAFFE_PROTO = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE_deploy.prototxt'
# CAFFE_PATH = '/home/gabi/Documents/caffe/python'
# CELEB_FACES = '/home/gabi/Documents/temp_datasets/caleba_align_crop_224'
# CELEB_FACES_FC6 = '/home/gabi/Documents/temp_datasets/celeba_fc6_features.txt'

sys.path.append(pp.CAFFE_PATH)
import caffe

caffe_model = caffe.Net(pp.VGGFACE_CAFFE_PROTO, pp.VGGFACE_CAFFE_MODEL, caffe.TEST)

list_faces = os.listdir(pp.CELEB_FACES)

# for i in range(len(list_faces)):
for i in range(10):
    name = os.path.join(pp.CELEB_FACES, list_faces[i])
    data = ndimage.imread(name).astype(np.float32)
    transformer = caffe.io.Transformer({'data': caffe_model.blobs['data'].data.shape})
    # RGB order
    mean_image = np.array([129.1863,104.7624,93.5940])

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean_image)            # subtract the dataset-mean value in each channel
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    transformed_image = transformer.preprocess('data', data)
    caffe_model.blobs['data'].data[...] = transformed_image
    output = caffe_model.forward()
    fc6_features = caffe_model.blobs['fc6'].data[0]
    # output_prob = output['fc6'][0]

    # write to features to file
    if not os.path.exists(pp.CELEB_FACES_FC6):
        _ = open(pp.CELEB_FACES_FC6, 'w')
        _.close()

    with open(pp.CELEB_FACES_FC6, 'a') as my_file:
        for j in range(4096):
            if j != 4095:
                if j == 0:
                    my_file.write('%s,%f,' % (list_faces[i], fc6_features[j]))
                else:
                    my_file.write('%f,' % fc6_features[j])
            else:
                my_file.write('%f\n' % fc6_features[j])


# data
# labels = list(np.genfromtxt('/media/gabi/DATADRIVE1/datasets/VGGFace/vgg_face_dataset/only_names.txt', dtype=str))
# example = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/adam.jpg'
# example_data = ndimage.imread(example).astype(np.float32)