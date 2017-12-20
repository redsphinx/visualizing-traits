import numpy as np
import subprocess
from random import randint
import skvideo.io
import time


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


def save_random_frame():
    video_path = '/home/gabi/Documents/temp_datasets/chalearn_fi_faces_aligned_center/test-1/test80_01/1uC-2TZqplE.003.mp4'
    save_path = 'test.jpg'
    meta_data = skvideo.io.ffprobe(video_path)
    fps = str(meta_data['video']['@avg_frame_rate'])
    fps = int(fps.split('/')[0][:2])
    begin_time, end_time = get_random_frame_number(fps)
    # command = "ffmpeg -loglevel panic -y -ss %s -t %s -i %s -r %s.0 %s" % (begin_time, end_time, video_path, fps, save_path)

    command = "ffmpeg -loglevel panic -y -ss %s -t %s -i %s -r %s.0 %s" % (begin_time, end_time, video_path, fps, save_path)


    subprocess.call(command, shell=True)

# 'ffmpeg -ss 00:00:09.43 -t 00:00:00.033 -i /home/gabi/Documents/temp_datasets/chalearn_fi_faces_aligned_center/test-1/test80_01/1uC-2TZqplE.003.mp4 -r 30.0 test.jpg'


t = time.time()
save_random_frame()
print('time: %s seconds' % str((time.time() - t) / 60))
