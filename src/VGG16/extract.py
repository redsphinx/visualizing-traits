import numpy as np
import os
from scipy import ndimage
from PIL import Image
import time
import h5py as h5
from chainer.links.model.vision.vgg import VGG16Layers

t = time.time()

ON_GPU = False
CELEBA_JPGS = '/home/gabi/Documents/tight_crop_everything/celeba'
VGG16_CONV3_3_FEATURES_H5 = '/home/gabi/Documents/tight_crop_everything/VGG16_relu3_3_features.h5'


if ON_GPU:
    vgg16 = VGG16Layers().to_gpu()
else:
    vgg16 = VGG16Layers()

all_celeba = os.listdir(CELEBA_JPGS)

action = 'a' if os.path.exists(VGG16_CONV3_3_FEATURES_H5) else 'w'
with h5.File(VGG16_CONV3_3_FEATURES_H5, action) as my_file:
    for i in range(len(all_celeba)):
        print('progress', i, len(all_celeba))
        p = os.path.join(CELEBA_JPGS, all_celeba[i])
        im = ndimage.imread(p).astype(np.float32)
        im[:, :, 0] -= 123.68
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 103.939
        im = np.asarray(Image.fromarray(im, mode='RGB').resize((224, 224), Image.ANTIALIAS), dtype=np.float32)
        relu3_3 = vgg16(np.expand_dims(im.transpose((2, 0, 1)), 0), layers=['conv3_3'])['conv3_3'].data[0]
        my_file.create_dataset(name=all_celeba[i], data=relu3_3)

print('time', (time.time() - t)/60)