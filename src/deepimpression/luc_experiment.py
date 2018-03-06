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


def main():
    model = load_model()
    model.validation = False

    labels = np.zeros((111, 5))
    r = 0

    with open(pp.LUC_LABELS, 'r') as my_file:
        reader = csv.reader(my_file)
        for row in reader:
            if row[0][0] == '0' or row[0][0] == '1':
                for i in range(5):
                    labels[r][i] = float(row[i])
                r += 1

    annotation_test_keys = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
    all_video_names = os.listdir(pp.LUC_VIDEOS)

    len_frames = len(all_video_names)
    y_tmp = np.zeros((len_frames, len(annotation_test_keys)), dtype=np.float32)
    target_tmp = np.zeros((len_frames, len(annotation_test_keys)), dtype=np.float32)

    # make zero audios
    sample_length = 50176
    shape_audio = (1, 1, 1, sample_length)
    audios = np.zeros(shape=shape_audio, dtype='float32')

    for ind in range(len_frames):
        print('ind: ', ind)
        video_id = all_video_names[ind]
        video_name = int(video_id.strip().split('_')[0].split('#')[-1])
        print('video name: ', video_name)
        path_video = os.path.join(pp.LUC_VIDEOS, video_id)
        print('path video: ', path_video)
        frame = get_random_frame(path_video)
        frame_shape = np.shape(frame)
        # reshape
        frame = np.reshape(frame, (3, frame_shape[0], frame_shape[1]))
        frame = np.expand_dims(frame, 0)
        # prediction
        # with chainer.using_config('train', False):
        #     prediction = model([audios, frame])

        # prediction = chainer.cuda.to_cpu(prediction.data)

        # y_tmp[ind] = prediction.data
        y_tmp[ind] = 0.5
        target_tmp[ind] = labels[video_name - 1]

    # calculate validation loss
    y_tmp.astype(np.float32)
    target_tmp.astype(np.float32)
    loss = chainer.functions.mean_absolute_error(y_tmp, target_tmp)
    print('model: ', pp.PRE_TRAINED, ' loss model: ', loss)

    # check if log file exists
    if not os.path.exists(pp.LUC_LOG):
        _ = open(pp.LUC_LOG, 'w')
        _.close()

    # save loss
    try:
        with open(pp.LUC_LOG, 'a') as my_file:
            line = 'model: %s, loss model: %s\n' % (pp.PRE_TRAINED, str(loss))
            # line = 'model: ', pp.PRE_TRAINED,' loss model: ', loss, '\n'
            my_file.write(line)
    except:
        pass


for i in range(5):
    main()
