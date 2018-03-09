from chainer.links.model.vision.vgg import VGG16Layers
import sys
from scipy import ndimage
import numpy as np
import chainer


VGGFACE_CAFFE_MODEL = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE.caffemodel'
VGGFACE_CAFFE_PROTO = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE_deploy.prototxt'
CAFFE_PATH = '/home/gabi/Documents/caffe/python'

sys.path.append(CAFFE_PATH)
import caffe

caffe_model = caffe.Net(VGGFACE_CAFFE_PROTO, VGGFACE_CAFFE_MODEL, caffe.TEST)

fc8_w = caffe_model.params['fc8'][0].data
fc8_b = caffe_model.params['fc8'][1].data

model = VGG16Layers(pretrained_model=False)
model.fc8.out_size = 2622
model.fc8.b = fc8_b
model.fc8.W = fc8_w

caffeVGG_as_chainerVGG = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/chainerVGGFace'
chainer.serializers.load_npz(caffeVGG_as_chainerVGG, model)

labels = list(np.genfromtxt('/media/gabi/DATADRIVE1/datasets/VGGFace/vgg_face_dataset/only_names.txt', dtype=str))
example = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/ak.png'
example_data = ndimage.imread(example).astype(np.float32)
s = np.shape(example_data)
example_data = np.reshape(example_data, (s[2], s[0], s[1]))
example_data = np.expand_dims(example_data, 0)

with chainer.using_config('train', False):
    y = model(example_data)
y = y['prob'].data[0]
i = np.argmax(y)
print(labels[i])
