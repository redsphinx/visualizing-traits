import numpy as np
import subprocess
from random import randint
import skvideo.io
import time
from PIL import Image
import librosa


def get_random_frame_number(fps):
    seconds = 15
    total_frames = fps * seconds
    random_number = randint(0, total_frames)
    each_frame = 1/(fps*1.0)
    big_seconds = random_number * each_frame
    time_12 = '00'
    time_34 = '00'
    t_56_int = int(big_seconds)
    t_56_float = '%0.2f' % (big_seconds - t_56_int)
    t_56_float = int(t_56_float.split('.')[-1])
    time_56 = '%02d.%d' % (t_56_int, t_56_float)
    begin_time = '%s:%s:%s' % (time_12, time_34, time_56)
    end_time = '00:00:0%0.3f' % each_frame
    return begin_time, end_time


def get_random_frame(video_path):
    # video_path = '/home/gabi/Documents/temp_datasets/chalearn_fi_faces_aligned_center/test-1/test80_01/1uC-2TZqplE.003.mp4'
    meta_data = skvideo.io.ffprobe(video_path)
    h = int(meta_data['video']['@height'])
    w = int(meta_data['video']['@width'])
    fps = str(meta_data['video']['@avg_frame_rate'])
    fps = int(fps.split('/')[0][:2])
    begin_time, end_time = get_random_frame_number(fps)
    # save_path = 'test.jpg'
    # command = "ffmpeg -loglevel panic -y -ss %s -t %s -i %s -r %s.0 %s" % (begin_time, end_time, video_path, fps, save_path)
    command = "ffmpeg -loglevel panic -ss %s -t %s -i %s -r %s.0 -f image2pipe -pix_fmt rgb24 -vcodec rawvideo -" % (begin_time, end_time, video_path, fps)
    pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    img = pipe.stdout.read(h*w*3)
    img = np.fromstring(img, dtype='uint8')
    img = img.reshape((h, w, 3))
    # im = Image.fromarray(img, mode='RGB')
    # im.show()
    return img


def get_random_audio_clip(video_path):
    audio = librosa.load(video_path, 16000)[0][None, None, None, :]
    sample_length = 50176
    audio_length = np.shape(audio)[-1]
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
    return frame, audio
