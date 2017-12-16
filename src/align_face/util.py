import time
import os


def safe_mkdir(my_path):
    if not os.path.exists(my_path):
        os.mkdir(my_path)


def safe_makedirs(my_path):
    if not os.path.exists(my_path):
        os.makedirs(my_path)


def get_time(t0):
    print('time: %s' % str(time.time() - t0))


def make_dirs(x):
    base_save_location = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_only_faces_aligned'
    data_path = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/%s' % x
    list_dirs = os.listdir(data_path)

    for i in list_dirs:
        p = os.path.join(base_save_location, x, i)
        safe_makedirs(p)

