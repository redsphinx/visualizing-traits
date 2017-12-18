# author:    redsphinx

import numpy as np
import skvideo.io
import os
from face_utils.facealigner import FaceAligner
import face_utils.helpers as h
import util
import tqdm
import dlib
import cv2
import imageio
from multiprocessing import Pool
import librosa
import subprocess
import time


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
        elif mode == 'similarity':
            face_aligned = fa.align_to_template_similarity(image, gray, rectangle)

    return face_aligned, radius


def align_faces_in_video(data_path, frames=None, audio=False):
    """
    Align face in video.
    :param data_path:
    :param frames:
    :param audio:
    :return:
    """
    # base_save_location = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_only_faces_aligned'
    # base_save_location = '/home/gabi/PycharmProjects/visualizing-traits/data/chalearn'
    base_save_location = '/home/gabi/PycharmProjects/visualizing-traits/data/testing'

    # which_test = data_path.strip().split('/')[-3]
    # which_video_folder = data_path.strip().split('/')[-2]
    # save_location = os.path.join(base_save_location, which_test, which_video_folder)

    save_location = base_save_location

    if os.path.exists(data_path):
        video_capture = skvideo.io.vread(data_path)
        meta_data = skvideo.io.ffprobe(data_path)
        # not necessary
        fps = str(meta_data['video']['@avg_frame_rate'])
        fps = int(fps.split('/')[0][:2])
        print('fps: %s' % fps)

        video_capture = np.array(video_capture, dtype=np.uint8)
        # audio_capture = librosa.load(data_path, 16000)[0][None, None, None, :]
        # audio_capture = librosa.load(data_path)
        # ac = audio_capture[0][None, None, None, :]
        if audio:
            command = "ffmpeg -i %s -ab 160k -ac 2 -ar 44100 -vn audio.wav" % data_path
            subprocess.call(command, shell=True)

        # TODO: delete the audio file

        if frames is None:
            frames = np.shape(video_capture)[0]

        side = 96
        channels = 3

        new_video_array = np.zeros((frames, side, side, channels), dtype='uint8')

        no_face_counter = 0
        the_radius = 0

        for i in tqdm.tqdm(range(frames)):
            frame = video_capture[i]
            new_frame, radius = align_face(frame, radius=the_radius, desired_face_width=side, mode='similarity')
            if i == 0:
                the_radius = radius

            # if no face detected, copy face from previous frame
            if new_frame is None:
                new_frame = new_video_array[i - 1]
                no_face_counter += 1

            new_frame = np.array(new_frame, dtype='uint8')
            new_video_array[i] = new_frame

        name_video = data_path.split('/')[-1].split('.mp4')[0]
        vid_name = os.path.join(save_location, '%s_align_center.mp4' % name_video)
        imageio.mimwrite(vid_name, new_video_array, fps=fps)
        if audio:
            # TODO: add the audio track
            time.sleep(1)
            command = "ffmpeg -i %s -i %s -codec copy -shortest %s_output.avi" % (vid_name, 'audio.wav', name_video)
            subprocess.call(command, shell=True)

    else:
        print('Error: data_path does not exist')


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


def add_audio(audio_path, video_path):
    command = "ffmpeg -i %s -ab 160k -ac 2 -ar 44100 -vn audio.wav" % audio_path
    subprocess.call(command, shell=True)

    name_video = video_path.split('/')[-1].split('.mp4')[0]

    command = "ffmpeg -i %s -i %s -codec copy -shortest %s_output.avi" % (video_path, 'audio.wav', name_video)
    subprocess.call(command, shell=True)


# ap = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/test-1/test80_01/glgfB3vFewc.004.mp4'
# vp = '/home/gabi/PycharmProjects/visualizing-traits/data/testing/glgfB3vFewc.004_align_center.mp4'
ap = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/test-1/test80_01/E3z1D7CKoOA.004.mp4'
# add_audio(ap, vp)
align_faces_in_video(ap, frames=None)
