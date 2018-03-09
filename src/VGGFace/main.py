from chainer.links.caffe import CaffeFunction as cf
from chainer import Variable
import chainer
import numpy as np
import project_paths as pp
from scipy import ndimage
from chainer.links.model.vision.vgg import VGG16Layers
import os


VGGFACE_CAFFE_MODEL = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE.caffemodel'
VGGFACE_CAFFE_PROTO = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE_deploy.prototxt'
CAFFE_PATH = '/home/gabi/Documents/caffe/python'


def trying_chainer_caffefunction():
    func = cf(VGGFACE_CAFFE_MODEL)

    labels = list(np.genfromtxt('/media/gabi/DATADRIVE1/datasets/VGGFace/names.txt', dtype=str))

    example = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/willow.jpg'
    example_data = ndimage.imread(example).astype(np.float32)
    s = np.shape(example_data)
    new_data = np.zeros(s).astype(np.float32)
    # RGB to BGR
    # for r in range(224):
    #     for c in range(224):
    #         new_data[r][c][2] = example_data[r][c][0]  # R
    #         new_data[r][c][0] = example_data[r][c][2]  # B
    # channels first
    example_data = np.reshape(example_data, (s[2], s[0], s[1]))
    # example_data = np.reshape(new_data, (s[2], s[0], s[1]))
    example_data = np.expand_dims(example_data, 0)

    x = Variable(example_data)

    with chainer.using_config('train', False):
        # y, = func(x)
        y, = func(inputs={'data': x}, outputs=['prob'])
    y = y.data[0]
    # y = y.data
    i = np.argmax(y)
    print(labels[i])


def trying_chainer_caffefunction_2():
    something = VGG16Layers(pretrained_model=False)
    caffeVGG_as_chainerVGG = 'chainerVGGFace'
    something.convert_caffemodel_to_npz(VGGFACE_CAFFE_MODEL, caffeVGG_as_chainerVGG)


# issues with this: some proto thingy dunno. Do this on the terminal!
def trying_caffe_caffe():
    import sys
    sys.path.append(CAFFE_PATH)
    import caffe
    caffe_model = caffe.Net(VGGFACE_CAFFE_PROTO, VGGFACE_CAFFE_MODEL, caffe.TEST)


def make_chainerVGGFace_fc8_layer():
    caffeVGG_as_chainerVGG = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/chainerVGGFace'

    # load model
    model = VGG16Layers(pretrained_model=False)
    model.fc8.out_size = 2622
    model.fc8.b = np.zeros(2622, dtype=np.float32)
    model.fc8.W = np.zeros((2622, 4096), dtype=np.float32)
    chainer.serializers.load_npz(caffeVGG_as_chainerVGG, model)
    print('loading successful')
    labels = list(np.genfromtxt('/media/gabi/DATADRIVE1/datasets/VGGFace/names.txt', dtype=str))
    example = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/alan.jpg'
    example_data = ndimage.imread(example).astype(np.float32)
    s = np.shape(example_data)
    example_data = np.reshape(example_data, (s[2], s[0], s[1]))
    example_data = np.expand_dims(example_data, 0)

    # x = Variable(example_data)

    with chainer.using_config('train', False):
        y = model(example_data)
    y = y['prob'].data[0]
    # y = y.data
    i = np.argmax(y)
    print(labels[i])
    # save as chainer model
    # model_name = 'chainer_VGGFace_w_proper_fc8.model'
    # chainer.serializers.save_npz(model_name, model)
    # print('model saved')


def test_chainer_VGGFace_w_proper_fc8():
    model_path = 'chainer_VGGFace_w_proper_fc8.model'
    model = chainer.serializers.load_npz(file=model_path, obj=self)

    # get example data
    labels = list(np.genfromtxt('/media/gabi/DATADRIVE1/datasets/VGGFace/names.txt', dtype=str))
    example = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/alan.jpg'
    example_data = ndimage.imread(example).astype(np.float32)
    s = np.shape(example_data)
    # new_data = np.zeros(s).astype(np.float32)
    # RGB to BGR
    # for r in range(224):
    #     for c in range(224):
    #         new_data[r][c][2] = example_data[r][c][0]  # R
    #         new_data[r][c][0] = example_data[r][c][2]  # B
    # channels first
    example_data = np.reshape(example_data, (s[2], s[0], s[1]))
    # example_data = np.reshape(new_data, (s[2], s[0], s[1]))
    example_data = np.expand_dims(example_data, 0)

    x = Variable(example_data)

    with chainer.using_config('train', False):
        y = model(x)
    y = y.data[0]
    # y = y.data
    i = np.argmax(y)
    print(labels[i])


# make_chainerVGGFace_fc8_layer()

def fix_label_names_VGGFace():
    labels_path = '/media/gabi/DATADRIVE1/datasets/VGGFace/vgg_face_dataset/names.txt'
    list_names = list(np.genfromtxt(labels_path, dtype=str))
    list_only_names = [list_names[i].strip().split('.txt')[0] for i in range(len(list_names))]

    file_name = '/media/gabi/DATADRIVE1/datasets/VGGFace/vgg_face_dataset/only_names.txt'
    if not os.path.exists(file_name):
        _ = open(file_name, 'w')
        _.close()

    with open(file_name, 'a') as my_file:
        for i in range(len(list_only_names)):
            my_file.write('%s\n' % list_only_names[i])


fix_label_names_VGGFace()
