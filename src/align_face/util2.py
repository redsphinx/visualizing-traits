import time
import os
from multiprocessing import Pool
import subprocess
import skvideo.io
import numpy as np
# from redis import Redis
# from rq import Queue
import project_paths as pp


def safe_mkdir(my_path):
    if not os.path.exists(my_path):
        os.mkdir(my_path)


def safe_makedirs(my_path):
    if not os.path.exists(my_path):
        os.makedirs(my_path)


def get_time(t0):
    print('time: %s' % str(time.time() - t0))


def make_folder_dirs(x):
    base_save_location = pp.BASE_SAVE_LOCATION
    data_path = os.path.join(pp.DATA_PATH, x)
    list_dirs = os.listdir(data_path)

    for i in list_dirs:
        p = os.path.join(base_save_location, x, i)
        safe_makedirs(p)


def get_path_videos(name_folder):
    # top_folder_path = '/home/gabi/Documents/temp_datasets/chalearn_fi_17_compressed'
    top_folder_path = pp.DATA_PATH
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


def parallel_align(which_folder, range_, func, number_processes=10):
    # func has to be align_faces_in_video
    pool = Pool(processes=number_processes)
    list_path_all_videos = get_path_videos(which_folder)
    list_path_all_videos.sort()
    list_path_all_videos = list_path_all_videos[range_[0]:range_[1]]
    # make folder in sa
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


def avi_to_mp4(old_path, new_path):
    command = "ffmpeg -loglevel panic -i %s -strict -2 %s" % (old_path, new_path)
    subprocess.call(command, shell=True)


def remove_file(file_path):
    forbidden = ['/', '/home', '/home/gabi', '*', '']
    if file_path in forbidden:
        print('ERROR: removing this file will lead to catastrophic error')
    else:
        command = "mv %s /tmp" % file_path
        subprocess.call(command, shell=True)


# def redis_stuff(which_folder, func):
#     list_path_all_videos = get_path_videos(which_folder)[0:5]
#     make_folder_dirs(which_folder)
#     q = Queue(connection=Redis())
#     q.enqueue(func, list_path_all_videos)


def move_files():
    folders = ['test80_08', 'test80_09', 'test80_10']

    from_path = '/vol/ccnlab-scratch1/gabras/chalearn_faces_test/test-1-missing-videos/test-1-missing-videos'
    to_path = '/vol/ccnlab-scratch1/gabras/chalearn_faces_test/test-1'
    check_path = '/vol/ccnlab-scratch1/gabras/chalearn_compressed/test-1'

    for i in range(len(folders)):
        f = folders[i]
        ch_pa = os.path.join(check_path, f)
        videos_that_should_be = os.listdir(ch_pa)
        dest = os.path.join(to_path, f)
        for j in range(len(videos_that_should_be)):
            video_name = videos_that_should_be[j]
            is_video_there = os.path.join(dest, video_name)
            is_video_available = os.path.join(from_path, video_name)
            if not os.path.exists(is_video_there):
                if os.path.exists(is_video_available):
                    command = "mv %s %s" % (is_video_available, dest)
                    subprocess.call(command, shell=True)


def find_largest_face(face_rectangles):
    number_rectangles = len(face_rectangles)

    if number_rectangles == 0:
        return None
    elif number_rectangles == 1:
        return face_rectangles[0]
    else:
        largest = 0
        which_rectangle = None
        for i in range(number_rectangles):
            r = face_rectangles[i]
            # it's a square so only one side needs to be checked
            width = r.right() - r.left()
            if width > largest:
                largest = width
                which_rectangle = i
        # print('rectangle %d is largest with a side of %d' % (which_rectangle, largest))
        return face_rectangles[which_rectangle]


def tight_template():
    current_template = pp.TEMPLATE
