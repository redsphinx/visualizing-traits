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
from multiprocessing import Pool
from src.align_face.face_utils.helpers import get_template_landmark
from src.align_face.util2 import resize_template
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.misc import imshow


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


def loop_de_loop():
    save_location = '/scratch2/gabi/VGGFACE/data'
    keywords = 'celeba_fc6_features_train_tight'
    cnt = 1
    for i in os.listdir(save_location):
        if keywords in i:
            print('file', cnt, 10)
            h5p = pp.FC6_TRAIN_H5
            p = os.path.join(save_location, i)
            save_features_as_h5(p, h5p)

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


def get_features_h5_in_batches(keys, train, which_features=None):
    features = np.zeros((pc.BATCH_SIZE, 4096))

    if which_features == 'vgg16':
        h5_path = pp.VGG16_RELU_H5
    else:
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
    labels_32 = np.zeros((pc.BATCH_SIZE, 32 * 32 * 3), dtype=np.float32)
    labels_224 = np.zeros((pc.BATCH_SIZE, 3, 224, 224), dtype=np.float32)
    # labels = np.zeros((10, 1072,1072,3))
    for i in range(len(names)):
        p = os.path.join(pp.CELEB_DATA_ALIGNED, names[i])
        img = ndimage.imread(p)
        img_copy = img.astype(np.float32)
        # remove mean
        img_copy[:, :, 0] -= 123.68
        img_copy[:, :, 1] -= 116.779
        img_copy[:, :, 2] -= 103.939

        labels_224[i] = np.transpose(img_copy, (2, 0, 1)).astype(np.float32)

        tmp_img = Image.fromarray(img, mode='RGB')
        # n = os.path.join(pp.ORIGINAL, names[i])
        # n = os.path.join(pp.CELEB_DATA_ALIGNED, names[i])
        # tmp_img.save(n)
        # tmp_img = tmp_img.resize((1072, 1072), Image.ANTIALIAS)
        tmp_img = tmp_img.resize((32, 32), Image.ANTIALIAS)
        img = np.array(tmp_img)
        img = np.ndarray.flatten(img)
        labels_32[i] = img.astype(np.float32)

    # labels = np.transpose(labels, (0, 3, 1, 2))
    return labels_32, labels_224


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


def fix_prediction_for_vgg16(prediction):
    pred = np.zeros((pc.BATCH_SIZE, 3, 224, 224)).astype(np.float32)
    for pr in range(pc.BATCH_SIZE):
        im = np.array(Image.fromarray(np.reshape(prediction.data[pr], (32, 32, 3)).astype(np.uint8)).resize((224, 224),
                                                                                                            Image.ANTIALIAS),
                      dtype=np.float32)
        im[:, :, 0] -= 123.68
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 103.939
        pred[pr] = im.transpose((2, 0, 1)).astype(np.float32)
    return pred


def manual_make_convex_hull(w=32, h=32):
    template = resize_template(pp.TEMPLATE, (w, h))
    canvas = np.ones((33, 33))
    # canvas[:, :, 0] = 255
    # canvas[:, :, 1] = 255
    # canvas[:, :, 2] = 255
    for p in template:
        x, y = p
        canvas[y, x] = 0#[255, 255, 255]
    plt.figure(figsize=(10,10))
    im = plt.imshow(canvas, interpolation='none', vmin=0, vmax=1, aspect='equal')
    ax = plt.gca();
    ax = plt.gca();
    x, y = np.meshgrid(range(33), range(33))

    # Major ticks
    ax.set_xticks(np.arange(0, 33, 1));
    ax.set_yticks(np.arange(0, 33, 1));

    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, 33, 1));
    ax.set_yticklabels(np.arange(0, 33, 1));

    # Minor ticks
    ax.set_xticks(np.arange(-.5, 33, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 33, 1), minor=True);

    # font size
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)

    for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
        if canvas[y_val][x_val] == 0:
            c = str(x_val)+','+str(y_val)
            ax.text(x_val, y_val, c, va='center', ha='center', fontsize=9, color='white')

    # Gridlines based on minor ticks
    # ax.grid(which='both', color='w', linestyle='-', linewidth=1)
    print('')


def manual_get_index_boundary(template):
    convex = pc.CONVEX_HULL_32
    index_list = []
    for i in range(len(convex)):
        for j, x in enumerate(template):
            if convex[i][0] == x[0] and convex[i][1] == x[1]:
                print(convex[i], x)
                index_list.append(j)
    print(index_list, len(index_list))  # this is the value of pc.CONVEX_HULL_INDEX
    return index_list


def get_boundary(template):
    new_template = []
    for i in pc.CONVEX_HULL_INDEX:
        new_template.append(template[i])
    return new_template


def get_L_sti_mask():
    h = 32
    w = 32
    template = resize_template(pp.TEMPLATE, (w, h))
    template = get_boundary(template)
    # template = ConvexHull(template).simplices
    template = Polygon(template)
    assert template.is_valid

    mask = []
    # images.shape = (batchsize, 3, w, h)
    for x in range(w):
        for y in range(h):
            point = Point(y, x)
            if not template.contains(point):
                mask.append(False)
            else:
                mask.append(True)
                # mask.append([y, x])
    mask = np.asarray(mask).reshape((w, h))
    mask = np.array([mask, mask, mask])
    mask = mask.transpose((1, 2, 0))
    return mask


def apply_mask(images, mask):
    # images has shape (batchsize, 32*32*3)
    # images needs to be shape (batchsize, 32, 32, 3)
    images = images.reshape((pc.BATCH_SIZE, 32, 32, 3))
    for i in range(pc.BATCH_SIZE):
        masked_image = images[i] * mask
        # masked_image = images[i][~np.array(mask)]
        # masked_image = ma.array(images[i], mask)
        images[i] = masked_image
        # imshow(masked_image)
    return images


