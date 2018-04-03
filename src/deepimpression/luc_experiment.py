from util import load_model, predict_trait, find_video_test, track_prediction, get_accuracy
from training_util import get_random_frame
import pickle as pkl
import os
import project_paths2 as pp
import project_constants as pc
import numpy as np
import chainer
# from scipy.stats import linregress
from sklearn import linear_model
import statsmodels.api as sm
import csv
from scipy.misc import imresize
from scipy.stats import normaltest, ttest_ind
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def get_luc_truth_labels():
    labels = np.zeros((111, 5))
    r = 0

    with open(pp.LUC_LABELS, 'r') as my_file:
        reader = csv.reader(my_file)
        for row in reader:
            if row[0][0] == '0' or row[0][0] == '1':
                for i in range(5):
                    labels[r][i] = float(row[i])
                r += 1

    return labels


def try_again(func):
    try:
        func
    except KeyError:
        print('key error')
        try_again(func)


def main():
    model = load_model()
    model.validation = False

    labels = get_luc_truth_labels()

    annotation_test_keys = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
    all_video_names = os.listdir(pp.LUC_VIDEOS)

    rep = 1

    len_frames = len(all_video_names)
    y_tmp = np.zeros((len_frames, len(annotation_test_keys)), dtype=np.float32)
    target_tmp = np.zeros((len_frames, len(annotation_test_keys)), dtype=np.float32)

    # make zero audios
    sample_length = 50176
    shape_audio = (1, 1, 1, sample_length)
    audios = np.zeros(shape=shape_audio, dtype='float32')

    # get the random ordering
    p_order = '/home/gabi/Downloads/shuffle_order.txt'
    shuffle_order = np.genfromtxt(p_order, dtype=int, delimiter=',')
    shuffle_order = list(shuffle_order[110593:])

    # repeat 10 times
    for i in range(rep):
        print('rep: %d out of %d' % (i, rep) )
        for ind in range(len_frames):
            # print('ind: ', ind)
            video_id = all_video_names[ind]
            video_name = int(video_id.strip().split('_')[0].split('#')[-1])
            # print('video name: ', video_name)
            path_video = os.path.join(pp.LUC_VIDEOS, video_id)
            # print('path video: ', path_video)
            frame = get_random_frame(path_video)
            frame_shape = np.shape(frame)
            # NOTE: TRANSPOSE
            # frame = np.transpose(frame, (2, 0, 1))
            # # reshape
            # frame = np.reshape(frame, (3, frame_shape[0], frame_shape[1]))
            # shuffle, reshape
            frame = np.ndarray.flatten(frame)
            frame = frame[shuffle_order]
            frame = np.reshape(frame, (3, 192, 192))
            frame = np.expand_dims(frame, 0)
            # prediction
            with chainer.using_config('train', False):
                prediction = model([audios, frame])

            y_tmp[ind] = prediction.data
            # y_tmp[ind] = 0.5
            target_tmp[ind] = labels[video_name - 1]

        # calculate validation loss
        # y_tmp.astype(np.float32)
        # target_tmp.astype(np.float32)
        # loss = chainer.functions.mean_absolute_error(y_tmp, target_tmp)
        # print('loss model: ', loss)

        # get average prediction per trait per video
        # y_avg = np.mean(y_tmp, axis=0)

        # check if log file exists
        # tp = '/home/gabi/pycharmprojects/visualizing-traits/data/luc/trait_log.txt'
        # tp = pp.LUC_TRANSPOSE_TRAIT_LOG
        tp = pp.LUC_TRAIT_LOG
        if not os.path.exists(tp):
            _ = open(tp, 'w')
            _.close()

        # save loss
        try:
            with open(tp, 'a') as my_file:
                # line = 'model: %s, loss model: %s\n' % (pp.pre_trained, str(loss))
                for j in range(len_frames):
                    line = '%s,%f,%f,%f,%f,%f\n' % (all_video_names[j], y_tmp[j][0], y_tmp[j][1], y_tmp[j][2], y_tmp[j][3], y_tmp[j][4])
                    my_file.write(line)
                    # line = 'model: ', pp.pre_trained,' loss model: ', loss, '\n'
                print('write successful!')
        except:
            pass


def get_error_per_id():
    labels = get_luc_truth_labels()
    tp = pp.LUC_TRAIT_LOG
    tp2 = pp.LUC_PRED_ID
    # tp = pp.LUC_TRANSPOSE_TRAIT_LOG
    # tp2 = pp.LUC_TRANSPOSE_PRED_ID

    r = 0
    with open(tp, 'r') as my_file:
        reader = csv.reader(my_file)
        for row in reader:
            # name = row[0]
            tmp_pred = [float(row[i]) for i in range(1,6)]
            id = int(row[0].strip().split('_')[0].split('#')[-1])
            tmp_truth = labels[id - 1]
            avg_loss = np.mean(np.abs(tmp_pred - tmp_truth))
            line = '%s,%f\n' % (row[0], avg_loss)
            print(line)

            if not os.path.exists(tp2):
                _ = open(tp2, 'w')
                _.close()

            with open(tp2, 'a') as f:
                f.write(line)
            r += 1


def get_error_rand_per_id():
    labels = get_luc_truth_labels()

    r = 0
    # use the file only to get the video names.
    with open(pp.LUC_TRAIT_LOG, 'r') as my_file:
        reader = csv.reader(my_file)
        for row in reader:
            # name = row[0]
            tmp_pred = [0.5, 0.5, 0.5, 0.5, 0.5]
            id = int(row[0].strip().split('_')[0].split('#')[-1])
            tmp_truth = labels[id - 1]
            avg_loss = np.mean(np.abs(tmp_pred - tmp_truth))
            line = '%s,%f\n' % (row[0], avg_loss)
            print(line)

            if not os.path.exists(pp.LUC_RAND_ID):
                _ = open(pp.LUC_RAND_ID, 'w')
                _.close()

            with open(pp.LUC_RAND_ID, 'a') as f:
                f.write(line)
            r += 1


def test_norm(p, nam):
    # test_norm(p=pp.LUC_RAND_ID, nam='random_error_distribution')

    data = [0.0] * 111

    r = 0
    with open(p, 'r') as my_file:
        reader = csv.reader(my_file)
        for row in reader:
            data[r] = float(row[1])
            r += 1

    k2, p_val = normaltest(data)
    print('k2', k2)
    print('p-val', p_val)

    mean = np.mean(data)
    print('average loss: %f' % mean)
    std = np.std(data)

    # the histogram of the data
    b = 20
    n, bins, patches = plt.hist(data, bins=b, normed=True)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mean, std)
    l = plt.plot(bins, y, 'r--', linewidth=2)

    plt.xlabel('%s' % nam)
    plt.savefig('%s_%d' % (nam, b))
    plt.show()


def sig_test():
    p1 = pp.LUC_RAND_ID
    p2 = pp.LUC_PRED_ID
    # p2 = pp.LUC_TRANSPOSE_PRED_ID

    data_p1 = np.array([0.0] * 111)
    data_p2 = np.array([0.0] * 111)

    r = 0
    with open(p1, 'r') as my_file:
        reader = csv.reader(my_file)
        for row in reader:
            data_p1[r] = float(row[1])
            r += 1
    r = 0
    with open(p2, 'r') as my_file:
        reader = csv.reader(my_file)
        for row in reader:
            data_p2[r] = float(row[1])
            r += 1

    t, p = ttest_ind(data_p2, data_p1, equal_var=False)
    print('t: ', t)
    print('p: ', p)


def super_main():
    main()
    get_error_per_id()
    p = pp.LUC_PRED_ID
    n = 'prediction_error_distribution_rand_order_899_2'
    test_norm(p, n)
    sig_test()


def get_avg_loss(file_name):
    data = [0.0] * 111

    r = 0
    with open(file_name, 'r') as my_file:
        reader = csv.reader(my_file)
        for row in reader:
            data[r] = float(row[1])
            r += 1

    avg_loss = np.mean(data)
    print('average loss: %f' % avg_loss)

# main()
# get_error_per_id()
# test_norm(p=pp.LUC_TRANSPOSE_PRED_ID, nam='trps_prediction_error_distribution')
# test_norm(p=pp.LUC_PRED_ID, nam='prediction_error_distribution')
# sig_test()

super_main()

# fn = '/home/gabi/PycharmProjects/visualizing-traits/data/luc/rand_ord_pred_id_3.csv'
# get_avg_loss(fn)