import numpy as np
import subprocess
from random import randint
import random
import skvideo.io
import time
from PIL import Image
import librosa
import pickle as pkl
import os
import project_constants as pc
import project_paths2 as pp


def get_random_frame_times(fps, seed, at_time, seconds):
    if seconds is None:
        seconds = 15
    else:
        seconds = seconds
    total_frames = fps * seconds
    random.seed(seed)
    random_number = randint(0, total_frames)
    each_frame = 1/(fps*1.0)
    if at_time is None:
        big_seconds = random_number * each_frame
    else:
        if not isinstance(at_time, int):
            print('error: at_time parameter must be int or None')
            return None, None
        else:
            if at_time > 15 | at_time < 0:
                print('error: at_time parameter must be between 0 and 15')
                return None, None
            else:
                big_seconds = at_time

    time_12 = '00'
    time_34 = '00'
    t_56_int = int(big_seconds)
    t_56_float = '%0.2f' % (big_seconds - t_56_int)
    t_56_float = int(t_56_float.split('.')[-1])
    time_56 = '%02d.%d' % (t_56_int, t_56_float)
    begin_time = '%s:%s:%s' % (time_12, time_34, time_56)
    end_time = '00:00:0%0.3f' % each_frame
    return begin_time, end_time


def get_random_frame(video_path, seed=None, at_time=None, seconds=None):
    meta_data = skvideo.io.ffprobe(video_path)
    h = int(meta_data['video']['@height'])
    w = int(meta_data['video']['@width'])
    fps = str(meta_data['video']['@avg_frame_rate'])
    fps = int(fps.split('/')[0][:2])
    begin_time, end_time = get_random_frame_times(fps, seed, at_time, seconds)
    command = "ffmpeg -loglevel panic -ss %s -t %s -i %s -r %s.0 -f image2pipe -pix_fmt rgb24 -vcodec rawvideo -" % (begin_time, end_time, video_path, fps)
    pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    img = pipe.stdout.read(h*w*3)
    img = np.fromstring(img, dtype='uint8')

    # some videos are shorter than 15 seconds, try to grab a random frame from first 5 seconds instead
    if np.size(img) == 0:
        print('recursion')
        img = get_random_frame(video_path, seed, at_time, seconds=1)

    img = img.reshape((h, w, 3))
    # im = Image.fromarray(img, mode='RGB')
    # im.show()
    return img


def get_random_audio_clip(video_path):
    audio = librosa.load(video_path, 16000)[0][None, None, None, :]
    sample_length = 50176
    audio_length = np.shape(audio)[-1]
    if audio_length < sample_length:
        # TODO: pad with zeros
        missing = sample_length - audio_length
        audio = audio[:, :, :, 0:audio_length]
        aud = np.reshape(audio, audio_length)
        aud = list(aud)
        aud += [0] * missing
        aud = np.array(aud)
        aud = np.reshape(aud, (1, 1, 1, sample_length))
        audio = aud
        del aud
    else:
        clip_here = randint(0, audio_length-sample_length)
        audio = audio[:, :, :, clip_here:clip_here+sample_length]
    return audio


def extract_frame_and_audio(video_path, get_audio=True):
    # get frame
    frame = get_random_frame(video_path)
    # get audio
    audio = None
    if get_audio:
        audio = get_random_audio_clip(video_path)
    return np.array(frame, 'float32'), np.array(audio, 'float32')


def get_names():
    # return random path to video and the label of that video in order
    with open(pp.TRAIN_LABELS, 'r') as my_file:
        annotation_train = pkl.load(my_file)

    # ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']
    annotation_train_keys = annotation_train.keys()
    number_of_classes = len(annotation_train_keys)

    list_names = []
    array_labels = np.zeros((pc.BATCH_SIZE, number_of_classes))

    for b in range(pc.BATCH_SIZE):
        # TODO: put all videos in same folder
        folder_number = randint(1, pc.NUMBER_TRAINING_FOLDERS)
        name_without_video = os.path.join(pp.TRAIN_DATA, 'training80_%02d' % folder_number)
        all_videos_here = os.listdir(name_without_video)
        random_number = randint(0, len(all_videos_here)-1)
        name_video = all_videos_here[random_number]
        path_video = os.path.join(name_without_video, name_video)
        list_names.append(path_video)

        for i in range(number_of_classes):
            k = annotation_train_keys[i]
            array_labels[b][i] = annotation_train[k][name_video]

    return list_names, array_labels
