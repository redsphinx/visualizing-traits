# author:    redsphinx

import numpy as np
import time
from PIL import Image
import psutil
import skvideo.io
import os
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2


def destroy_frame():
    for proc in psutil.process_iter():
        if proc.name() == "display":
            proc.kill()


def show_frames(data_path, frames):
    data_path = '/home/gabi/PycharmProjects/visualizing-traits/data/1uC-2TZqplE.003.mp4'
    video_capture = skvideo.io.vread(data_path)
    video_capture = np.array(video_capture, dtype=np.uint8)

    for i in range(frames):
        frame = Image.fromarray(video_capture[0], mode='RGB')
        print('shape: ', np.shape(frame))
        frame.show()
        time.sleep(0.8)
        destroy_frame()


def align_face(image):
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", required=True,
    #                 help="path to facial landmark predictor")
    # ap.add_argument("-i", "--image", required=True,
    #                 help="path to input image")
    # args = vars(ap.parse_args())

    pred_path = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'

    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(args["shape_predictor"])
    predictor = dlib.shape_predictor(pred_path)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # resize it, and convert it to grayscale
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

        # print(type(faceAligned))

        # display the output images
        # cv2.imshow("Original", faceOrig)
        # cv2.imshow("Aligned", faceAligned)
        # cv2.waitKey(0)

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

        for i in range(frames):
            frame = video_capture[i]
            new_frame = align_face(frame)
            new_image = Image.fromarray(new_frame, mode='RGB')
            new_image_save_path = os.path.join(save_location, '%04d.jpg' % i)
            new_image.save(new_image_save_path)

    else:
        print('Error: data_path does not exist')


data_path = '/home/gabi/PycharmProjects/visualizing-traits/data/1uC-2TZqplE.003.mp4'
save_location = '/home/gabi/PycharmProjects/visualizing-traits/data'
align_faces_in_video(data_path, save_location)
