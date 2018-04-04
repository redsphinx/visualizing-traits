# author:    redsphinx

import numpy as np
import skvideo.io
import os
from face_utils.facealigner import FaceAligner
# import face_utils.helpers as h
import util2 as util
import tqdm
import dlib
import cv2
import imageio
# import librosa
import subprocess
import time
from scipy import ndimage
import project_paths as pp
from PIL import Image


def align_face(image, desired_face_width, radius=None, mode='similarity'):
    """
    Given an image, return processed image where face is aligned according to chosen mode.
    :param image:
    :param desired_face_width: should be 198 -- 224 for celeba dataset
    :param radius:
    :param mode:
    :return: image of aligned face, radius of face if radius is not None
    """
    # create the facial landmark predictor
    predictor = pp.PREDICTOR
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
    # print('len: ', len(face_rectangles))

    if len(face_rectangles) == 0:
        return None, 0

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

# p = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/face_utils/arya_250w.jpg'
# p = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/backup_face.jpg'
# align_face(p, desired_face_width=196)


def align_faces_in_video(data_path, frames=None, audio=True, side=196, mode='similarity'):
    """
    Align face in video.
    :param data_path:
    :param frames:
    :param audio:
    :param side: 198 originally, then 196
    :param mode:
    :return:
    """
    # uncomment when testing is over
    base_save_location = pp.BASE_SAVE_LOCATION
    # print('base_save_location = %s' % base_save_location)

    # use these for testing
    # base_save_location = '/home/gabi/PycharmProjects/visualizing-traits/data/testing'
    # base_save_location = '/home/gabi/PycharmProjects/visualizing-traits/data/luc'
    save_location = base_save_location

    # relevant when testing is over
    # which_test = data_path.strip().split('/')[-3]
    # # print('which_test = %s' % which_test)
    # which_video_folder = data_path.strip().split('/')[-2]
    # # print('which_video_folder = %s' % which_video_folder)
    # save_location = os.path.join(base_save_location, which_test, which_video_folder)

    if os.path.exists(data_path):
        video_capture = skvideo.io.vread(data_path)
        meta_data = skvideo.io.ffprobe(data_path)
        fps = str(meta_data['video']['@avg_frame_rate'])
        fps = int(fps.split('/')[0][:2])
        print('fps: %s' % fps)

        # name_video = data_path.split('/')[-1].split('.mp4')[0]
        name_video = data_path.split('/')[-1].split('.MTS')[0]
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
            # print i
            if i % 20 == 0:
                print('%s: %s of %s' % (name_video, i, frames))
            frame = video_capture[i]
            new_frame, radius = align_face(frame, radius=the_radius, desired_face_width=side, mode=mode)
            if i == 0:
                the_radius = radius

            # if no face detected, copy face from previous frame
            if new_frame is None:
                new_frame = new_video_array[i - 1]

            new_frame = np.array(new_frame, dtype='uint8')
            new_video_array[i] = new_frame

        print('END %s' % name_video)
        # vid_name = os.path.join(save_location, '%s.mp4' % name_video)
        # comment for testing
        vid_name = os.path.join(save_location, '%s_aligned.mp4' % name_video)
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


def align_celeba_faces_in_folder():
    list_names = os.listdir(pp.DATA_PATH)
    names_already_saved = os.listdir(pp.BASE_SAVE_LOCATION)
    list_names = list(set(list_names) - set(names_already_saved))

    # partition_size = len(list_names) / 4
    #
    # b = 3*partition_size
    # e = 4*partition_size

    for i in tqdm.tqdm(range(len(list_names))):
    # for i in tqdm.tqdm(range(b, e)):
        # print('%d / %d' % (i, len(list_names)))
        name = os.path.join(pp.DATA_PATH, list_names[i])
        frame = ndimage.imread(name).astype(np.uint8)
        new_frame, radius = align_face(frame, radius=0, desired_face_width=198, mode='similarity')
        if new_frame is not None:
            img = Image.fromarray(new_frame, mode='RGB')
            img = img.resize((224, 224), Image.ANTIALIAS)
            name = os.path.join(pp.BASE_SAVE_LOCATION, list_names[i])
            img.save(name)
        else:
            print('%s, no face' % name)


def align_anouk_data():
    list_names = os.listdir(pp.DATA_PATH)
    names_already_saved = os.listdir(pp.BASE_SAVE_LOCATION)
    list_names = list(set(list_names) - set(names_already_saved))
    for f in list_names:
        name = os.path.join(pp.DATA_PATH, f)
        align_faces_in_video(name)


def main():
    list_files_ = os.listdir(pp.DATA_PATH)
    list_files = []
    # seen_list = ['#35.MTS', '#56.MTS', '#77.MTS']
    not_seen_number = [1, 11, 12, 16, 24, 29, 41, 43, 53, 60, 70, 87, 88, 90, 94, 101]
    # print(len(not_seen_number))
    not_seen_list = [('#%d.mp4' % i) for i in not_seen_number]

    for f in list_files_:
        if f[0] is '#':
            # if f not in seen_list:
            if f in not_seen_list:
                file_size = os.path.getsize(os.path.join(pp.DATA_PATH, f)) / 1000000
                # if file_size < 100:
                    # print(f)
                list_files.append(f)

    del list_files_

    for f in list_files:
        print('file: ', f)
        file_name = os.path.join(pp.DATA_PATH, f)
        align_faces_in_video(file_name)


# dp = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_train/train-1/training80_01/1DCnIad1Y0w.002.mp4'
# align_faces_in_video(dp, frames=30)

# p2 = "/home/gabi/PycharmProjects/visualizing-traits/data/face_2.jpg"
# align_face(p2, 196)

# util.parallel_align('test-1', [0, 100], align_faces_in_video, number_processes=10)
# util.parallel_align('test-1', [400, 700], align_faces_in_video, number_processes=10)

# vid = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/test-1/test80_01/IGjI8aP14gg.000.mp4'
# align_faces_in_video(vid)
# main()
# align_celeba_faces_in_folder()
align_anouk_data()