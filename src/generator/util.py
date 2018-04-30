import subprocess
import project_paths as pp
import numpy as np
import project_constants as pc
import os
from scipy import ndimage
from PIL import Image
import chainer
from shlex import split
import pandas as pd
import h5py as h5
import tqdm
from matplotlib.pyplot import plot as pyplot
from multiprocessing import Pool


def csv_pandas(path, num):
    f = pd.read_csv(path, skiprows=num, nrows=1)
    print(f)


def get_number_of_features(path):
    thing = subprocess.check_output(["wc", "-l", path])
    thing = int(thing.split(" ")[0])
    return thing


def sed_line(path, num):
    # num > 0
    thing = subprocess.check_output(["sed", "%sq;d" % num, path])
    return thing


def tail_head(path, num):
    p1 = subprocess.Popen(split("tail -n+%s %s" % (num, path)), stdout=subprocess.PIPE)
    p2 = subprocess.Popen(split("head -n1"), stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    thing, err = p2.communicate()
    return thing


def save_features_as_h5(path, h5_path):
    num_features = get_number_of_features(path)
    action = 'a' if os.path.exists(h5_path) else 'w'
    with h5.File(h5_path, action) as my_file:
        for i in tqdm.tqdm(range(num_features)):
            f = sed_line(path, i+1).strip().split(',')
            nam = f[0]
            if nam == '':
                print('nam is ''', f)
            # dat = [float(_) for _ in f[1:]]
            dat = []
            for j in range(len(f[1:])):
                _ = f[1:][j].split('.jpg')[0]
                try:
                    __ = float(_)
                except (ValueError, TypeError):
                    _ = 0.0
                    print('weird number, setting to 0.0')
                dat.append(float(_))

            try:
                my_file.create_dataset(name=nam, data=dat)
            except ValueError:
                print('name is: ', nam)


# h5p = pp.FC6_TRAIN_H5
# p = pp.CELEB_FACES_FC6_TRAIN
# h5p = pp.FC6_TEST_H5
# p = pp.CELEB_FACES_FC6_TEST
# save_features_as_h5(p, h5p)


def get_features_in_batches(step, train=True):
    features = np.zeros((pc.BATCH_SIZE, 4096))
    names = [''] * pc.BATCH_SIZE
    # names = np.asarray(names)

    if train:
        faces = pp.CELEB_FACES_FC6_TRAIN
    else:
        faces = pp.CELEB_FACES_FC6_TEST

    b = step * pc.BATCH_SIZE
    e = b + pc.BATCH_SIZE

    cnt = 0

    for i in range(b, e):
        f = sed_line(faces, i).strip().split(',')
        names[cnt] = f[0]
        features[cnt] = [float(i) for i in f[1:]]
        cnt += 1

    return names, features


def get_features_h5_in_batches(keys, train):
    features = np.zeros((pc.BATCH_SIZE, 4096))
    if train:
        h5_path = pp.FC6_TRAIN_H5
    else:
        h5_path = pp.FC6_TEST_H5

    h5_file = h5.File(h5_path, 'r')

    for i in range(len(keys)):
        k = keys[i]
        features[i] = h5_file[k][:]

    return features


def to_correct_input(features):
    new_features = np.zeros((pc.BATCH_SIZE, 1, 64, 64), dtype=np.float32)
    for i in range(pc.BATCH_SIZE):
        _ = np.reshape(features[i], (64, 64))
        new_features[i] = np.asarray(_, dtype=np.float32)
    return new_features


# def get_features(train=True):
#     if train:
#         faces = pp.CELEB_FACES_FC6_TRAIN
#     else:
#         faces = pp.CELEB_FACES_FC6_TEST
#
#     features = list(np.genfromtxt(faces, dtype=str))
#     new_features = np.zeros((len(features), 1, 1, 64, 64), dtype=np.float32)
#     names = [''] * 10
#     for i in range(len(features)):
#         tmp = features[i].split(',')[1:]
#         tmp = [float(j) for j in tmp]
#         tmp = np.reshape(tmp, (64, 64))
#         new_features[i] = np.asarray(tmp, dtype=np.float32)
#         names[i] = features[i].split(',')[0]
#     return new_features, names


def get_names_h5_file(path):
    f = h5.File(path, 'r')
    k = f.keys()
    k = [str(_) for _ in k]
    return k


def get_labels(names):
    labels = np.zeros((pc.BATCH_SIZE, 32*32*3))
    # labels = np.zeros((10, 1072,1072,3))
    for i in range(len(names)):
        p = os.path.join(pp.CELEB_DATA_ALIGNED, names[i])
        img = ndimage.imread(p).astype(np.uint8)
        tmp_img = Image.fromarray(img, mode='RGB')
        # n = os.path.join(pp.ORIGINAL, names[i])
        # n = os.path.join(pp.CELEB_DATA_ALIGNED, names[i])
        # tmp_img.save(n)
        # tmp_img = tmp_img.resize((1072, 1072), Image.ANTIALIAS)
        tmp_img = tmp_img.resize((32, 32), Image.ANTIALIAS)
        img = np.array(tmp_img)
        img = np.ndarray.flatten(img)
        labels[i] = img.astype(np.float32)

    # labels = np.transpose(labels, (0, 3, 1, 2))
    return labels


def save_image(arr, name, epoch, location):
    arr = arr.data[0]
    # shape_arr = np.shape(arr)
    r1 = np.reshape(arr, (32, 32, 3))
    # r1 = np.transpose(arr, (1, 2, 0))
    r1 = np.asarray(r1, dtype=np.uint8)
    r1 = Image.fromarray(r1, mode='RGB')
    r1 = r1.resize((128, 128), Image.ANTIALIAS)
    nam = name.split('.')[0]
    # new_name = os.path.join(pp.RECONSTRUCTION_FOLDER, '%s_%d.jpg' % (nam, epoch))
    new_name = os.path.join(location, '%s_%d.jpg' % (nam, epoch))
    # r1.show()
    r1.save(new_name)


def save_model(model, epoch):
    model_name = os.path.join(pp.MODEL_SAVES, 'generator_e_%d' % epoch)
    chainer.serializers.save_npz(model_name, model)
    print('model saved')


def make_ones(generator):
    ones = []
    for i in range(pc.BATCH_SIZE):
        tmp = generator.xp.array([1])
        ones.append(tmp)
    ones = generator.xp.asarray(ones, dtype=np.int32)
    ones = chainer.Variable(ones)
    return ones

def make_zeros(generator):
    ones = []
    for i in range(pc.BATCH_SIZE):
        tmp = generator.xp.array([0])
        ones.append(tmp)
    ones = generator.xp.asarray(ones, dtype=np.int32)
    ones = chainer.Variable(ones)
    return ones


def update_information(information, step, generator_loss, l1, l2):
    information = np.transpose(information, (1, 0))
    information[step][0] = generator_loss
    information[step][1] = l1
    information[step][2] = l2
    information = np.transpose(information, (1, 0))
    return information


def plot_everything(information, fig, lines, axis, prev_max, step):
    num = len(lines)
    for i in range(num):
        line = lines[i]
        # line[0].__setattr__('_y', information[i])
        line.set_ydata(information[i])
    information_max = np.max(information)
    if information_max < prev_max:
        information_max = prev_max
    axis.set_ylim((0, int(information_max)+1))
    axis.set_xlim((0, int(step)))
    fig.canvas.draw()
    return information_max


