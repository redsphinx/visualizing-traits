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


def align_face(image):
    pred_path = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pred_path)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # resize it, and convert it to grayscale
    image = imutils.resize(image, width=800)
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


def align_faces_in_video(data_path, save_location, frames=None):
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    if os.path.exists(data_path):
        video_capture = skvideo.io.vread(data_path)
        video_capture = np.array(video_capture, dtype=np.uint8)

        name_video = data_path.split('/')[-1].split('mp4')[0]
        save_location = os.path.join(save_location, name_video)
        if not os.path.exists(save_location):
            os.mkdir(save_location)

        if frames is None:
            frames = np.shape(video_capture)[0]

        new_height, new_width, channels = 256, 256, 3

        new_video_array = np.zeros((frames, new_height, new_width, channels))

        no_face_counter = 0

        for i in tqdm.tqdm(range(frames)):
            frame = video_capture[i]
            new_frame = align_face(frame)

            # if no face detected, copy face from previous frame
            if new_frame is None:
                new_frame = new_video_array[i - 1]
                no_face_counter += 1
                print('no face detected %d' % no_face_counter)

            new_video_array[i] = new_frame

            # for saving individual frames as jpg
            # new_image = Image.fromarray(new_frame, mode='RGB')
            # new_image_save_path = os.path.join(save_location, '%04d.jpg' % i)
            # new_image.save(new_image_save_path)

        vid_name = os.path.join(save_location, 'test_2.mp4')
        imageio.mimwrite(vid_name, new_video_array, fps=30.)

    else:
        print('Error: data_path does not exist')


data_path_ = '/home/gabi/PycharmProjects/visualizing-traits/data/1Lv72Si4GnY.000.mp4'
save_location_ = '/home/gabi/PycharmProjects/visualizing-traits/data'
align_faces_in_video(data_path_, save_location_, frames=None)
