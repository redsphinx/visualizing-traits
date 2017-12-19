import math
import numpy as np
from skimage import data
from skimage import transform as tf
from PIL import Image
from scipy.misc import imsave, imshow
from redis import Redis
from rq import Queue


def similarity():
    tform = tf.SimilarityTransform(scale=1, rotation=math.pi/2,
                                   translation=(0, 1))
    print(tform.params)

    text = data.text()

    tform = tf.SimilarityTransform(scale=1, rotation=math.pi/4,
                                   translation=(text.shape[0]/2, -100))

    rotated = tf.warp(text, tform)
    back_rotated = tf.warp(rotated, tform.inverse)

    imshow(rotated)

    imshow(back_rotated)
