# author:    redsphinx

import numpy as np
import skvideo.io
import os
from face_utils.facealigner import FaceAligner
import face_utils.helpers as h
import util
import dlib
import cv2
import imageio
from multiprocessing import Pool
import librosa


def align_face(image, desired_face_width=96, mode='center', radius='fixed'):
    """
    Given an image, return processed image where face is aligned according to chosen mode
    :param image:
    :return:
    """
    # create the facial landmark predictor
    predictor = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor)
    # initialize dlib's face detector (HOG-based)
    detector = dlib.get_frontal_face_detector()
    # create the face aligner
    fa = FaceAligner(predictor, desiredFaceWidth=desired_face_width)

    # resize it, and convert it to grayscale
    # image = imutils.resize(image, width=800)
    image = h.resize(image, width=400)
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
        util.safe_makedirs(p)


def align_faces_in_video(data_path, frames=None):
    print('aligning: %s' % data_path)
    base_save_location = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_only_faces_aligned'

    # tmp = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/test-1/test80_01/3gmc2kLV4Bo.003.mp4'

    which_test = data_path.strip().split('/')[-3]
    which_video_folder = data_path.strip().split('/')[-2]

    save_location = os.path.join(base_save_location, which_test, which_video_folder)

    # if not os.path.exists(save_location):
    #     os.makedirs(save_location)

    if os.path.exists(data_path):
        # returns array with shape (459, 256, 456, 3)
        video_capture = skvideo.io.vread(data_path)
        print(np.shape(video_capture))
        video_capture = np.array(video_capture, dtype=np.uint8)
        # returns an array with shape (1, 1, 1, 244800)
        audio_capture = librosa.load(data_path, 16000)[0][None, None, None, :]
        print(np.shape(audio_capture))

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

        vid_name = os.path.join(save_location, '%s.mp4' % name_video)
        # TODO: add the audio track
        imageio.mimwrite(vid_name, new_video_array, fps=30.)

    else:
        print('Error: data_path does not exist')


align_faces_in_video('/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/test-1/test80_01/3gmc2kLV4Bo.003.mp4')


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


def parallel_align(which_folder):
    pool = Pool(processes=10)
    list_path_all_videos = get_path_videos(which_folder)[0:80]
    make_dirs(which_folder)
    pool.apply_async(align_faces_in_video)
    pool.map(align_faces_in_video, list_path_all_videos)

