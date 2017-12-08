# author:    redsphinx

import numpy as np
import time
import tqdm
from PIL import Image
import psutil
import skvideo.io
import os
from imutils.face_utils import FaceAligner
# from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2
import imageio
from multiprocessing import Pool


def destroy_frame():
    for proc in psutil.process_iter():
        if proc.name() == "display":
            proc.kill()


def show_frames(data_path, frames):
    data_path = '/home/gabi/PycharmProjects/visualizing-traits/data/1Lv72Si4GnY.000./test_2.mp4'
    video_capture = skvideo.io.vread(data_path)
    video_capture = np.array(video_capture, dtype=np.uint8)

    for i in range(frames):
        frame = Image.fromarray(video_capture[0], mode='RGB')
        print('shape: ', np.shape(frame))
        frame.show()
        time.sleep(0.8)
        destroy_frame()


def safe_mkdir(my_path):
    if not os.path.exists(my_path):
        os.mkdir(my_path)


def safe_makedirs(my_path):
    if not os.path.exists(my_path):
        os.makedirs(my_path)


def get_time(t0):
    print('time: %s' % str(time.time() - t0))


def align_face(image):
    pred_path = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pred_path)
    # fa = FaceAligner(predictor, desiredFaceWidth=256)
    fa = FaceAligner(predictor, desiredFaceWidth=96)

    # resize it, and convert it to grayscale
    # image = imutils.resize(image, width=800)
    image = imutils.resize(image, width=400)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 2)

    faceAligned = None

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        # (x, y, w, h) = rect_to_bb(rect)
        # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

    return faceAligned


def make_dirs(x):
    base_save_location = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_only_faces_aligned'
    data_path = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/%s' % x
    list_dirs = os.listdir(data_path)

    for i in list_dirs:
        p = os.path.join(base_save_location, x, i)
        safe_makedirs(p)


def align_faces_in_video(data_path, frames=None):
    print('aligning: %s' % data_path)
    base_save_location = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_only_faces_aligned'

    # tmp = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/test-1/test80_01/3gmc2kLV4Bo.003.mp4'

    which_test = data_path.strip().split('/')[6]
    which_video_folder = data_path.strip().split('/')[7]

    save_location = os.path.join(base_save_location, which_test, which_video_folder)

    # if not os.path.exists(save_location):
    #     os.makedirs(save_location)

    if os.path.exists(data_path):
        video_capture = skvideo.io.vread(data_path)
        video_capture = np.array(video_capture, dtype=np.uint8)
        name_video = data_path.split('/')[-1].split('mp4')[0]

        if frames is None:
            frames = np.shape(video_capture)[0]

        # new_height, new_width, channels = 256, 256, 3
        new_height, new_width, channels = 96, 96, 3

        new_video_array = np.zeros((frames, new_height, new_width, channels), dtype='uint8')

        no_face_counter = 0

        # for i in tqdm.tqdm(range(frames)):
        for i in range(frames):
            frame = video_capture[i]
            new_frame = align_face(frame)

            # if no face detected, copy face from previous frame
            if new_frame is None:
                new_frame = new_video_array[i - 1]
                no_face_counter += 1
                # print('\nno face detected %d' % no_face_counter)

            new_frame = np.array(new_frame, dtype='uint8')
            new_video_array[i] = new_frame

            # for saving individual frames as jpg
            # new_image = Image.fromarray(new_frame, mode='RGB')
            # new_image_save_path = os.path.join(save_location, '%04d.jpg' % i)
            # new_image.save(new_image_save_path)

        vid_name = os.path.join(save_location, '%s.mp4' % name_video)
        # TODO: add the audio track
        imageio.mimwrite(vid_name, new_video_array, fps=30.)

    else:
        print('Error: data_path does not exist')


def get_number_of_videos_per_folder():
    top_folder_path = '/media/gabi/DATADRIVE1/datasets/chalearn_first_impressions_17'
    video_suffix = 'mp4'

    level_1 = os.listdir(top_folder_path)
    print('level 0: %s' % top_folder_path.split('/')[-1])
    for l1 in level_1:
        l1_path = os.path.join(top_folder_path, l1)
        if os.path.isdir(l1_path):
            print('- level 1: %s' % l1)
            level_2 = os.listdir(l1_path)
            for l2 in level_2:
                l2_path = os.path.join(l1_path, l2)
                if os.path.isdir(l2_path):
                    level_3 = os.listdir(l2_path)
                    video_count = 0
                    for l3 in level_3:
                        if l3.strip().split('.')[-1] == video_suffix:
                            video_count += 1
                    print('-- level 2: %s   %d videos' % (l2, video_count))


def align_faces_in_folder():
    folder_path = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed'
    save_location_top = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_only_faces_aligned'

    safe_mkdir(save_location_top)
    level_1 = os.listdir(folder_path)
    for l1 in level_1:
        l1_path = os.path.join(folder_path, l1)
        save_location_l1 = os.path.join(save_location_top, l1)
        iszip = l1.strip().split('.')[-1]
        if os.path.isdir(l1_path) and not iszip == 'zip':
            safe_mkdir(save_location_l1)
            level_2 = os.listdir(l1_path)
            for l2 in level_2:
                l2_path = os.path.join(l1_path, l2)
                save_location_l2 = os.path.join(save_location_l1, l2)
                if os.path.isdir(l2_path):
                    level_3 = os.listdir(l2_path)
                    safe_mkdir(save_location_l2)
                    for l3 in level_3:
                        if l3.strip().split('.')[-1] == 'mp4':
                            l3_path = os.path.join(l2_path, l3)
                            align_faces_in_video(l3_path, save_location_l2, frames=10)


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


def parallel_align_1(which_folder):
    pool = Pool(processes=40)
    list_path_all_videos = get_path_videos(which_folder)[0:80]
    make_dirs(which_folder)
    pool.apply_async(align_faces_in_video)
    pool.map(align_faces_in_video, list_path_all_videos)


# t = time.time()
# parallel_align_1('test-1')
# get_time(t)
# make_dirs('test-2')
