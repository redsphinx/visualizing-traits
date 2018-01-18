import audiovisual_stream
import chainer.serializers
import librosa
import numpy as np
import skvideo.io
import os
# import psutil
from project_paths2 import ON_GPU
import project_paths2 as pp
from PIL import Image
import subprocess
import time


def load_audio(data):
    audio = librosa.load(data, 16000)[0][None, None, None, :]
    return audio


def load_model(load_trained=True):
    print('loading model')
    if ON_GPU:
        model = audiovisual_stream.ResNet18().to_gpu(device='0')
    else:
        model = audiovisual_stream.ResNet18()

    # model = audiovisual_stream.ResNet18()
    if load_trained:
        chainer.serializers.load_npz(pp.PRE_TRAINED, model)

    return model


def load_video(data):
    print('loading data')
    video_capture = skvideo.io.vread(data)

    frames = 50
    video_capture = video_capture[:frames]

    video_shape = np.shape(video_capture)
    print('video shape: ', video_shape)
    # frames = video_shape[0]
    video_capture = np.reshape(video_capture, (frames, video_shape[-1], video_shape[1], video_shape[2]), 'float32')
    # video_capture = np.reshape(video_capture, (frames, video_shape[1], video_shape[2], video_shape[-1]), 'float32')
    # return video_capture
    video = np.array(video_capture, 'float32')
    return video


def predict_trait(data, model):
    print('predicting trait')
    x = [load_audio(data), load_video(data)]
    print(type(x))
    try:
        print(np.shape(x))
    except:
        print("can't")
    print('now really predicting. this will take a while.')
    with chainer.using_config('train', False):
        return model(x)
        # return model(x)


def find_video(video_id):
    print('finding video: ', video_id)
    # ----------------------------------------------------------------------------
    # base_path_1 = 'chalearn_fi_17_compressed/test-1'
    base_path_1 = os.path.join(pp.TEST_DATA, 'test-1')
    # base_path_2 = 'chalearn_fi_17_compressed/test-2'
    base_path_2 = os.path.join(pp.TEST_DATA, 'test-2')
    # ----------------------------------------------------------------------------
    video = None

    not_found = True
    while not_found:
        for item in [base_path_1, base_path_2]:
            all_dirs = os.listdir(item)
            for d in all_dirs:
                all_vids = os.listdir(os.path.join(item, d))
                for vid in all_vids:
                    if vid == video_id:
                        video = os.path.join(item, d, vid)
                        not_found = False
    return video


def track_prediction(video_id, prediction, target):
    print('tracking prediction')
    output_path = 'performance_chalearn.txt'
    if not os.path.exists(output_path):
        # create file
        output_file = open(output_path, 'w')
        output_file.close()

    with open(output_path, 'a') as my_file:
        my_file.write('%s,\n%s,\n%s,\n%s\n' % (video_id, str(prediction), str(target), str(target - prediction)))


def get_accuracy(output_path):
    result_per_trait = np.zeros((2000, 5), dtype=float)
    result_total = np.zeros(2000, dtype=float)

    with open(output_path, 'r') as my_file:
        all_lines = my_file.readlines()

    for ind in range(2000):
        diff = all_lines[ind * 4 + 3]
        diff = diff.split('\n')[0].split('[')[-1].split(']')[0]
        diff = np.fromstring(diff, sep=' ', dtype=float)
        diff = np.abs(diff)
        print(ind * 4 + 3, diff, type(diff), np.shape(diff))
        result_per_trait[ind] = 1 - diff
        result_total[ind] = np.mean(1 - diff)

    result_per_trait = np.mean(result_per_trait, axis=0)
    print('average accuracy per trait:', result_per_trait)

    result_total = np.mean(result_total)
    print('average accuracy per video:', result_total)


def mp4_to_jpgs(video_path, save_path):
    # video_path = '//home/gabi/PycharmProjects/visualizing-traits/data/training/training80_03/PuVy3akfzNI.000.mp4'
    # save_path = '/home/gabi/PycharmProjects/visualizing-traits/data/mp4_to_jpgs_4'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    video_capture = skvideo.io.vread(video_path)
    for i in range(video_capture.shape[0]):
        raw = video_capture[i]
        img = Image.fromarray(raw, mode='RGB')
        name = '%03d.jpg' % i
        name = os.path.join(save_path, name)
        img.save(name)


def mp4_to_wav(video_path, save_path):
    name_audio = os.path.join(save_path, 'audio.wav')
    command = "ffmpeg -loglevel panic -i %s -ab 160k -ac 2 -ar 44100 -vn -y %s" % (video_path, name_audio)
    subprocess.call(command, shell=True)


# for all the videos, save as jpgs and wav
def folders_mp4_to_jpgs():
    # check if important folders exist, else make them
    if not os.path.exists(pp.TRAIN_DATA):
        print('train data does not exist:\n%s' % pp.TRAIN_DATA)
        return
    if not os.path.exists(pp.CHALEARN_JPGS):
        os.mkdir(pp.CHALEARN_JPGS)

    for i in os.listdir(pp.TRAIN_DATA):
        l1 = os.path.join(pp.TRAIN_DATA, i)
        if os.path.isdir(l1):
            nl1 = os.path.join(pp.CHALEARN_JPGS, i)
            if not os.path.exists(nl1):
                os.mkdir(nl1)

            # videos
            t = time.time()
            for j in os.listdir(l1):
                video_path = os.path.join(l1, j)
                j_name = j.split('.mp4')[0]
                new_video_folder = os.path.join(nl1, j_name)
                # save jpgs
                mp4_to_jpgs(video_path, new_video_folder)
                # save wav
                mp4_to_wav(video_path, new_video_folder)
            print('time: %f seconds' % (time.time() - t))


# folders_mp4_to_jpgs()
