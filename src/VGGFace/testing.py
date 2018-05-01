import numpy as np
from PIL import Image
from scipy import ndimage


def get_order():
    p = '/home/gabi/Downloads/shuffle_order.txt'
    l = np.genfromtxt(p, dtype=int, delimiter=',')
    # l = list(l[0:110592])
    l = list(l[110593:])
    return l

# get_order()

def image_to_array():
    example = 'jeff.jpg'
    e = ndimage.imread(example).astype(np.uint8)
    e = Image.fromarray(e, mode='RGB')
    e = e.resize((192, 192), Image.ANTIALIAS)
    e = np.array(e)
    e = Image.fromarray(e, mode='RGB')
    e.show()
    print('asdf')


image_to_array()