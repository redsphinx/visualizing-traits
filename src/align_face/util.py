import time
import os
from multiprocessing import Pool
import subprocess
import skvideo.io
import numpy as np
from redis import Redis
from rq import Queue


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


def parallel_align(which_folder, func, number_processes=5):
    # func has to be align_faces_in_video
    pool = Pool(processes=number_processes)
    list_path_all_videos = get_path_videos(which_folder)[0:number_processes]
    make_folder_dirs(which_folder)
    pool.apply_async(func)
    pool.map(func, list_path_all_videos)


def add_audio(vid_name, name_audio, avi_vid_name):
    command = "ffmpeg -loglevel panic -i %s -i %s -codec copy -shortest -y %s" % (vid_name, name_audio,
                                                                                  avi_vid_name)
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


def avi_to_mp4(old_path, new_path):
    command = "ffmpeg -loglevel panic -i %s -strict -2 %s" % (old_path, new_path)
    subprocess.call(command, shell=True)


def remove_file(file_path):
    forbidden = ['/', '/home', '/home/gabi', '*', '']
    if file_path in forbidden:
        print('ERROR: deleting this file will lead to catastrophic error')
    else:
        command = "mv %s /tmp" % file_path
        subprocess.call(command, shell=True)


def redis_stuff(which_folder, func):
    list_path_all_videos = get_path_videos(which_folder)[0:5]
    make_folder_dirs(which_folder)
    q = Queue(connection=Redis())
    q.enqueue(func, list_path_all_videos)
