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


def align_face(image, radius=None, desired_face_width=96, mode='center'):
    """
    Given an image, return processed image where face is aligned according to chosen mode.
    :param image:
    :param radius:
    :param desired_face_width:
    :param mode:
    :return: image of aligned face, radius of face if radius is not None
    """
    # create the facial landmark predictor
    predictor = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor)

    # initialize dlib's face detector (HOG-based)
    detector = dlib.get_frontal_face_detector()

    # create the face aligner
    fa = FaceAligner(predictor, desiredFaceWidth=desired_face_width)

    # resize it, and convert it to gray scale
    image = h.resize(image, width=200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the gray scale image
    face_rectangles = detector(gray, 2)

    # initialize variables
    face_aligned = None

    # loop over the face detections
    for rectangle in face_rectangles:
        if mode == 'eyes_mass':
            face_aligned = fa.align(image, gray, rectangle)
        elif mode == 'eyes_geometric':
            face_aligned = fa.align_geometric_eyes(image, gray, rectangle)
        elif mode == 'center':
            face_aligned, radius = fa.align_center(image, gray, rectangle, radius)
        elif mode == 'template_affine':
            face_aligned = fa.align_to_template_affine(image, gray, rectangle)

    return face_aligned, radius


def align_faces_in_video(data_path, frames=None):
    """
    Align face in video.
    :param data_path:
    :param frames:
    :return:
    """
    # base_save_location = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_only_faces_aligned'
    base_save_location = '/home/gabi/PycharmProjects/visualizing-traits/data/chalearn'

    which_test = data_path.strip().split('/')[-3]
    which_video_folder = data_path.strip().split('/')[-2]

    save_location = os.path.join(base_save_location, which_test, which_video_folder)

    if os.path.exists(data_path):
        video_capture = skvideo.io.vread(data_path)
        video_capture = np.array(video_capture, dtype=np.uint8)
        audio_capture = librosa.load(data_path, 16000)[0][None, None, None, :]

        if frames is None:
            frames = np.shape(video_capture)[0]

        side = 96
        channels = 3

        new_video_array = np.zeros((frames, side, side, channels), dtype='uint8')

        no_face_counter = 0
        the_radius = 0

        for i in range(frames):
            frame = video_capture[i]
            new_frame, radius = align_face(frame, radius=the_radius, desired_face_width=side)
            if i == 0:
                the_radius = radius

            # if no face detected, copy face from previous frame
            if new_frame is None:
                new_frame = new_video_array[i - 1]
                no_face_counter += 1

            new_frame = np.array(new_frame, dtype='uint8')
            new_video_array[i] = new_frame

        name_video = data_path.split('/')[-1].split('mp4')[0]
        vid_name = os.path.join(save_location, '%s.mp4' % name_video)
        # TODO: add the audio track
        # TODO: get FPS from video
        fps = 30.
        imageio.mimwrite(vid_name, new_video_array, fps=fps)

    else:
        print('Error: data_path does not exist')


# data_path = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/test-1/test80_01/3gmc2kLV4Bo.003.mp4'
dp = '/media/gabi/345148f0-e089-41d9-8570-eb01be812c35/home/gabi/Documents/datasets/chalearn_fi_17_compressed/test-1/test80_01/2Z8Xi_DTlpI.000.mp4'
align_faces_in_video(dp)


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
    h.make_dirs(which_folder)
    pool.apply_async(align_faces_in_video)
    pool.map(align_faces_in_video, list_path_all_videos)

