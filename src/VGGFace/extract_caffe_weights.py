# just checking if the caffe model is working using caffe
import sys
from scipy import ndimage
import numpy as np
import os
import project_paths as pp
import time
import util
import tqdm


# VGGFACE_CAFFE_MODEL = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE.caffemodel'
# VGGFACE_CAFFE_PROTO = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE_deploy.prototxt'
# CAFFE_PATH = '/home/gabi/Documents/caffe/python'
# CELEB_FACES = '/home/gabi/Documents/temp_datasets/caleba_align_crop_224'
# CELEB_FACES_FC6 = '/home/gabi/Documents/temp_datasets/celeba_fc6_features.txt'

sys.path.append(pp.CAFFE_PATH)
import caffe

caffe_model = caffe.Net(pp.VGGFACE_CAFFE_PROTO, pp.VGGFACE_CAFFE_MODEL, caffe.TEST)
is_train = True


def extract_classic():
    caffe_model = caffe.Net(pp.VGGFACE_CAFFE_PROTO, pp.VGGFACE_CAFFE_MODEL, caffe.TEST)
    list_faces = os.listdir(pp.CELEB_FACES)

    num_test = 30000
    if is_train:
        list_faces = list_faces[num_test:]
        feature_path = pp.CELEB_FACES_FC6_TRAIN
    else:
        list_faces = list_faces[0:num_test]
        feature_path = pp.CELEB_FACES_FC6_TEST


    if not os.path.exists(feature_path):
        _ = open(feature_path, 'w')
        _.close()

    tt = time.time()
    for i in tqdm.tqdm(range(len(list_faces))):
    # for i in range(100):
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
        with open(feature_path, 'a') as my_file:
            for j in range(4096):
                if j != 4095:
                    if j == 0:
                        my_file.write('%s,%f,' % (list_faces[i], fc6_features[j]))
                    else:
                        my_file.write('%f,' % fc6_features[j])
                else:
                    my_file.write('%f\n' % fc6_features[j])

    print('total time: %f minutes' % ((time.time() - tt) / 60))


# if is_train:
#     name_write_to = 'train.txt'
# else:
#     name_write_to = 'test.txt'
# write_to = os.path.join(pp.CELEB_FACES_FC6_ALL, name_write_to)
# f = open(write_to, 'w')
# f.close()


def extract_parallel(list_faces):
    name = os.path.join(pp.CELEB_FACES, list_faces)
    data = ndimage.imread(name).astype(np.float32)
    transformer = caffe.io.Transformer({'data': caffe_model.blobs['data'].data.shape})
    # RGB order
    mean_image = np.array([129.1863, 104.7624, 93.5940])

    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean_image)  # subtract the dataset-mean value in each channel
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    transformed_image = transformer.preprocess('data', data)
    caffe_model.blobs['data'].data[...] = transformed_image
    output = caffe_model.forward()
    fc6_features = caffe_model.blobs['fc6'].data[0]

    # write to features to file
    with open(write_to, 'a') as my_file:
        for j in range(4096):
            if j != 4095:
                if j == 0:
                    my_file.write('%s,%f,' % (list_faces, fc6_features[j]))
                else:
                    my_file.write('%f,' % fc6_features[j])
            else:
                my_file.write('%f\n' % fc6_features[j])


# data
# labels = list(np.genfromtxt('/media/gabi/DATADRIVE1/datasets/VGGFace/vgg_face_dataset/only_names.txt', dtype=str))
# example = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/adam.jpg'
# example_data = ndimage.imread(example).astype(np.float32)

tt = time.time()
# util.parallel_extract(is_train, [0, 30000], extract_parallel, number_processes=5)
extract_classic()
print('total time: %f minutes' % ((time.time() - tt) / 60))
# list_f = util.get_list_faces(is_train, [0,10])
# extract_parallel(list_f)
