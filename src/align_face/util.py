import time
import os
from multiprocessing import Pool
import subprocess
import skvideo.io
import numpy as np


def safe_mkdir(my_path):
    if not os.path.exists(my_path):
        os.mkdir(my_path)


def safe_makedirs(my_path):
    if not os.path.exists(my_path):
        os.makedirs(my_path)


def get_time(t0):
    print('time: %s' % str(time.time() - t0))


def make_folder_dirs(x):
    base_save_location = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_faces_aligned_center'
    data_path = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/%s' % x
    list_dirs = os.listdir(data_path)

    for i in list_dirs:
        p = os.path.join(base_save_location, x, i)
        safe_makedirs(p)


def get_path_videos(name_folder):
    top_folder_path = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed'
    top_folder_path = os.path.join(top_folder_path, name_folder)

    video_folders = os.listdir(top_folder_path)
    video_path_list = []

    for f in video_folders:
        video_folder_path = os.path.join(top_folder_path, f)
        videos = os.listdir(video_folder_path)
        for v in videos:
            video_path = os.path.join(video_folder_path, v)
            video_path_list.append(video_path)

    print('length list: %s videos' % str(len(video_path_list)))

    return video_path_list


def parallel_align(which_folder, func, number_processes=30):
    # func has to be align_faces_in_video
    pool = Pool(processes=number_processes)
    list_path_all_videos = get_path_videos(which_folder)
    make_folder_dirs(which_folder)
    pool.apply_async(func)
    pool.map(func, list_path_all_videos)


def add_audio(audio_path, video_path):
    command = "ffmpeg -i %s -ab 160k -ac 2 -ar 44100 -vn audio.wav" % audio_path
    subprocess.call(command, shell=True)

    name_video = video_path.split('/')[-1].split('.mp4')[0]

    command = "ffmpeg -i %s -i %s -codec copy -shortest %s_output.avi" % (video_path, 'audio.wav', name_video)
    subprocess.call(command, shell=True)


def open_avi(data_path):
    if os.path.exists(data_path):
        video_capture = skvideo.io.vread(data_path)
        print(np.shape(video_capture))
        meta_data = skvideo.io.ffprobe(data_path)
        fps = str(meta_data['video']['@avg_frame_rate'])
        fps = int(fps.split('/')[0][:2])
        print('fps: %s' % fps)
        print('all good')


# open_avi('/home/gabi/PycharmProjects/visualizing-traits/data/testing/13kjwEtSyXc.003.mp4')
# open_avi('/home/gabi/PycharmProjects/visualizing-traits/data/testing/1uC-2TZqplE.003.avi')
