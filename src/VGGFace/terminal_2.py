# check if the weights are the same
from chainer.links.model.vision.vgg import VGG16Layers
import sys
from scipy import ndimage
import numpy as np
import chainer


################################
# Port caffe weights to chainer
################################

VGGFACE_CAFFE_MODEL = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE.caffemodel'
VGGFACE_CAFFE_PROTO = '/media/gabi/DATADRIVE1/datasets/VGGFace/VGG_FACE_deploy.prototxt'
CAFFE_PATH = '/home/gabi/Documents/caffe/python'

sys.path.append(CAFFE_PATH)
import caffe

caffe_model = caffe.Net(VGGFACE_CAFFE_PROTO, VGGFACE_CAFFE_MODEL, caffe.TEST)
chainer_model = VGG16Layers(pretrained_model=False)

# conv1_1
conv1_1_w = caffe_model.params['conv1_1'][0].data
conv1_1_b = caffe_model.params['conv1_1'][1].data

# cov1_1
conv1_2_w = caffe_model.params['conv1_2'][0].data
conv1_2_b = caffe_model.params['conv1_2'][1].data

# fc 6
fc6_w = caffe_model.params['fc6'][0].data
fc6_b = caffe_model.params['fc6'][1].data
# chainer_model.fc6.b = fc6_b
# chainer_model.fc6.W = fc6_w

# fc7
fc7_w = caffe_model.params['fc7'][0].data
fc7_b = caffe_model.params['fc7'][1].data
# chainer_model.fc7.b = fc7_b
# chainer_model.fc7.W = fc7_w

# fc8
fc8_w = caffe_model.params['fc8'][0].data
fc8_b = caffe_model.params['fc8'][1].data
chainer_model.fc8.out_size = 2622
chainer_model.fc8.b = fc8_b
chainer_model.fc8.W = fc8_w

# check if weights are the same

caffeVGG_as_chainerVGG = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/chainerVGGFace'
chainer.serializers.load_npz(caffeVGG_as_chainerVGG, chainer_model)

n_conv1_1_w = chainer_model.conv1_1.W.data
n_conv1_1_b = chainer_model.conv1_1.b.data
n_conv1_2_w = chainer_model.conv1_2.W.data
n_conv1_2_b = chainer_model.conv1_2.b.data
# n_fc6_b = chainer_model.fc6.b.data
# n_fc6_w = chainer_model.fc6.W.data
# n_fc7_b = chainer_model.fc7.b.data
# n_fc7_w = chainer_model.fc7.W.data


print('conv1_1_w', n_conv1_1_w == conv1_1_w)
print('conv1_1_b', n_conv1_1_b == conv1_1_b)
print('conv1_2_w', n_conv1_2_w == conv1_2_w)
print('conv1_2_b', n_conv1_2_b == conv1_2_b)

# print('fc6_b', n_fc6_b == fc6_b)
# print('fc6_w', n_fc6_w == fc6_w)
# print('fc7_b', n_fc7_b == fc7_b)
# print('fc7_w', n_fc7_w == fc7_w)

################################
# Load test data
################################

labels = list(np.genfromtxt('/media/gabi/DATADRIVE1/datasets/VGGFace/vgg_face_dataset/only_names.txt', dtype=str))
example = '/home/gabi/PycharmProjects/visualizing-traits/src/VGGFace/alan.jpg'
example_data = ndimage.imread(example).astype(np.float32)
# subtract channel mean from each channel
mean_image = [129.1863,104.7624,93.5940]
for i in range(3):
    m = np.ones((224, 224)) * mean_image[i]
    example_data[:, :, i] = example_data[:, :, i] - m

s = np.shape(example_data)
# RGB to BGR
example_data = example_data[:, :, [2, 1, 0]]

# channels first + switch width and height
example_data = np.transpose(example_data, axes=(2, 1, 0))
# example_data = np.reshape(example_data, (s[2], s[0], s[1]))

example_data = np.expand_dims(example_data, 0)

################################
# See if it works
################################

with chainer.using_config('train', False):
    y = chainer_model(example_data)

y = y['prob'].data[0]
i = np.argmax(y)
print(labels[i])
