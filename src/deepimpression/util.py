import audiovisual_stream
import chainer.serializers
import librosa
import numpy as np
import skvideo.io
import os
import psutil


def load_audio(data):
    return librosa.load(data, 16000)[0][None, None, None, :]


def load_model():
    print('loading model')
    model = audiovisual_stream.ResNet18().to_gpu(device='0')
    # model = audiovisual_stream.ResNet18()

    # maybe here?
    chainer.serializers.load_npz('./model', model)

    return model


def load_video(data):
    print('loading data')
    video_capture = skvideo.io.vread(data)

    frames = 50
    video_capture = video_capture[:frames]

    video_shape = np.shape(video_capture)
    print('video shape: ', video_shape)
    # frames = video_shape[0]
    # video_capture = np.reshape(video_capture, (frames, video_shape[-1], video_shape[1], video_shape[2]), 'float32')
    # video_capture = np.reshape(video_capture, (frames, video_shape[1], video_shape[2], video_shape[-1]), 'float32')
    # return video_capture
    return np.array(video_capture, 'float32')


def predict_trait(data, model):
    print('predicting trait')
    x = [load_audio(data), load_video(data)]
    print('now really predicting. this will take a while.')
    with chainer.using_config('train', False):
        return model(x)
        # return model(x)


def find_video(video_id):
    print('finding video: ', video_id)
    # ----------------------------------------------------------------------------
    base_path_1 = 'chalearn_fi_17_compressed/test-1'
    base_path_2 = 'chalearn_fi_17_compressed/test-2'
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

# get_accuracy('data/performance_chalearn.txt')



