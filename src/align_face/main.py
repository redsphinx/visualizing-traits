# author:    redsphinx

import sys
sys.path.append('/home/gabi/PycharmProjects/visualizing-traits/src')
import deepimpression.util as du
import os
import numpy as np
import cv2
from PIL import Image
import time
import psutil
import pylab
import matplotlib.pyplot as plt
from scipy import misc
import imageio
import skvideo.io


data_path = '/home/gabi/PycharmProjects/visualizing-traits/data/1uC-2TZqplE.003.mp4'

def show_frames_1():
    vid = imageio.get_reader(data_path,  'ffmpeg')
    frames = 3
    for num in range(frames):
        image = vid.get_data(num)
        fig = pylab.figure()
        fig.suptitle('image #{}'.format(num), fontsize=20)
        pylab.imshow(image)
    pylab.show()


def show_frames_2():
    video_capture = skvideo.io.vread(data_path)
    frames = 10
    # video_shape = np.shape(video_capture)
    # video_capture = np.reshape(video_capture[0:frames], (frames, video_shape[-1], video_shape[1], video_shape[2]), 'float32')
    # video_shape = np.shape(video_capture)
    # print('video shape: ', video_shape)

    video_capture = np.array(video_capture[0:frames], dtype=np.uint8)

    img = Image.fromarray(video_capture[0], mode='RGB')
    print(type(img))
    img.save('shit.jpg')



show_frames_2()
