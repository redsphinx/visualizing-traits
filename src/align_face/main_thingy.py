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
# import librosa
import subprocess
import time
from scipy import ndimage


def align_face(image, desired_face_width, radius=None, mode='center'):
    """
    Given an image, return processed image where face is aligned according to chosen mode.
    :param image:
    :param desired_face_width:
    :param radius:
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

    # for debugging
    if not isinstance(image, np.ndarray):
        if isinstance(image, str):
            if os.path.exists(image):
                image = ndimage.imread(image)
            else:
                print('incorrect path: %s' % image)

    # resize it, and convert it to gray scale
    # image = h.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the gray scale image
    face_rectangles = detector(gray, 2)

    # initialize variables
    face_aligned = None

    # find largest face rectangle
    largest_face_rectangle = util.find_largest_face(face_rectangles)

    # do alignment
    if mode == 'eyes_mass':
        face_aligned = fa.align(image, gray, largest_face_rectangle)
    elif mode == 'eyes_geometric':
        face_aligned = fa.align_geometric_eyes(image, gray, largest_face_rectangle)
    elif mode == 'center':
        face_aligned, radius = fa.align_center(image, gray, largest_face_rectangle, radius)
    elif mode == 'template_affine':
        face_aligned = fa.align_to_template_affine(image, gray, largest_face_rectangle)
    elif mode == 'similarity':
        face_aligned = fa.align_to_template_similarity(image, gray, largest_face_rectangle)

    return face_aligned, radius


# img = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/face_utils/arya2face.jpg'
# img = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/face_utils/ARYA.jpg'
# img = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/face_utils/arya.jpeg'
img = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/backup_face_2.jpg'
# dfw = 96
dfw = 198
m = 'similarity'

align_face(img, dfw, mode=m)


def align_faces_in_video(data_path, frames=None, audio=True, side=96):
    """
    Align face in video.
    :param data_path:
    :param frames:
    :param audio:
    :param side:
    :return:
    """
    base_save_location = '/home/gabi/Documents/temp_datasets/chalearn_fi_faces_aligned_center'
    # base_save_location = '/home/gabi/PycharmProjects/visualizing-traits/data/testing'

    # relevant when testing is over
    which_test = data_path.strip().split('/')[-3]
    which_video_folder = data_path.strip().split('/')[-2]
    save_location = os.path.join(base_save_location, which_test, which_video_folder)

    # uncomment for testing
    # save_location = base_save_location

    if os.path.exists(data_path):
        video_capture = skvideo.io.vread(data_path)
        meta_data = skvideo.io.ffprobe(data_path)
        fps = str(meta_data['video']['@avg_frame_rate'])
        fps = int(fps.split('/')[0][:2])
        print('fps: %s' % fps)

        name_video = data_path.split('/')[-1].split('.mp4')[0]
        name_audio = None

        video_capture = np.array(video_capture, dtype=np.uint8)
        if audio:
            name_audio = os.path.join(save_location, '%s.wav' % name_video)
            command = "ffmpeg -loglevel panic -i %s -ab 160k -ac 2 -ar 44100 -vn -y %s" % (data_path, name_audio)
            subprocess.call(command, shell=True)

        if frames is None:
            frames = np.shape(video_capture)[0]

        channels = 3

        new_video_array = np.zeros((frames, side, side, channels), dtype='uint8')

        the_radius = 0

        for i in range(frames):
            if i % 20 == 0:
                print('%s: %s of %s' % (name_video, i, frames))
            frame = video_capture[i]
            new_frame, radius = align_face(frame, radius=the_radius, desired_face_width=side, mode='center')
            if i == 0:
                the_radius = radius

            # if no face detected, copy face from previous frame
            if new_frame is None:
                new_frame = new_video_array[i - 1]

            new_frame = np.array(new_frame, dtype='uint8')
            new_video_array[i] = new_frame

        print('END %s' % name_video)
        vid_name = os.path.join(save_location, '%s.mp4' % name_video)
        imageio.mimwrite(vid_name, new_video_array, fps=fps)
        if audio:
            # add audio to the video
            time.sleep(1)
            avi_vid_name = os.path.join(save_location, '%s.avi' % name_video)
            util.add_audio(vid_name, name_audio, avi_vid_name)
            command = "ffmpeg -loglevel panic -i %s -i %s -codec copy -shortest -y %s" % (vid_name, name_audio,
                                                                                          avi_vid_name)
            subprocess.call(command, shell=True)
            # remove first mp4
            util.remove_file(vid_name)
            # convert avi to mp4
            util.avi_to_mp4(avi_vid_name, vid_name)
            # remove the wav file
            util.remove_file(name_audio)
            # remove the avi file
            util.remove_file(avi_vid_name)
    else:
        print('Error: data_path does not exist')
