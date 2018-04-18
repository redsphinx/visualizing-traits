import time
import os
from multiprocessing import Pool
import project_paths as pp


def parallel_extract(is_train, range_, func, number_processes=10):
    # func has to be extract()
    pool = Pool(processes=number_processes)

    list_faces = os.listdir(pp.CELEB_FACES)
    num_test = 30000
    if is_train:
        list_faces = list_faces[num_test:]
    else:
        list_faces = list_faces[0:num_test]

    if len(list_faces) > range_[0]:
        if len(list_faces) > range_[1]:
            list_faces = list_faces[range_[0]:range_[1]]
        else:
            range_[1] = len(list_faces)
            print('range_[1] is out of bounds, setting range_[1] to %d' % len(list_faces))
            list_faces = list_faces[range_[0]:range_[1]]
    else:
        print('range_[0] is out of bounds, keeping original list')

    pool.apply_async(func)
    pool.map(func, list_faces)


def get_list_faces(is_train, range_):
    list_faces = os.listdir(pp.CELEB_FACES)
    num_test = 30000
    if is_train:
        list_faces = list_faces[num_test:]
    else:
        list_faces = list_faces[0:num_test]

    if len(list_faces) > range_[0]:
        if len(list_faces) > range_[1]:
            list_faces = list_faces[range_[0]:range_[1]]
        else:
            range_[1] = len(list_faces)
            print('range_[1] is out of bounds, setting range_[1] to %d' % len(list_faces))
            list_faces = list_faces[range_[0]:range_[1]]
    else:
        print('range_[0] is out of bounds, keeping original list')

    return list_faces
