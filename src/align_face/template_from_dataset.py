import numpy as np
import os
from src.deepimpression import training_util as tu
from PIL import Image
import tqdm
import dlib
import cv2
from src.align_face import util
from src.align_face.face_utils.helpers import shape_to_np
from scipy import ndimage

# total of 10000 videos
# get the face landmarks from a random frame in each video to calculate the mean position of each
# face landmark in the training dataset

# set seed for rng
seed = 6

# get all the video paths
all_paths = []
top_path = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_train'
folders_1 = os.listdir(top_path)
for i in folders_1:
    p1 = os.path.join(top_path, i)
    if os.path.isdir(p1):
        folders_2 = os.listdir(p1)
        for j in folders_2:
            p2 = os.path.join(p1, j)
            if os.path.isdir(p2):
                videos = os.listdir(p2)
                for v in videos:
                    p3 = os.path.join(p2, v)
                    all_paths.append(p3)

del folders_1

# for all videos, get random frame, get the landmark and store in array
height = 360
width = 640
channels = 3
number_landmarks = 68
num_videos = len(all_paths)
random_frame_landmarks_from_all_videos = np.zeros((num_videos, number_landmarks * 2))
predictor = '/home/gabi/PycharmProjects/visualizing-traits/data/predictor/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor)
detector = dlib.get_frontal_face_detector()

intervals = range(1100, num_videos + 1, 100)
for i1 in range(len(intervals)):
    if i1 < len(intervals):
        b = intervals[i1]
        e = intervals[i1 + 1]

    for i in tqdm.tqdm(range(b, e)):
        retries = 0
        tmp_seed = seed
        p = all_paths[i]
        print(p)
        largest_face_rectangle = None
        gray = None
        while largest_face_rectangle is None and retries < 5:
            arr_frame = tu.get_random_frame(p, tmp_seed)
            image_frame = Image.fromarray(arr_frame, mode='RGB')
            image_frame = image_frame.resize((width, height), Image.ANTIALIAS)
            arr_frame = list(image_frame.getdata())
            arr_frame = np.reshape(arr_frame, (height, width, channels)).astype(np.uint8)
            gray = cv2.cvtColor(arr_frame, cv2.COLOR_BGR2GRAY)
            face_rectangles = detector(gray, 2)
            largest_face_rectangle = util.find_largest_face(face_rectangles)
            if largest_face_rectangle is None:
                tmp_seed += 1
                retries += 1
                print(retries)

        try:
            landmarks = shape_to_np(predictor(gray, largest_face_rectangle))
        except TypeError:
            arr_frame = ndimage.imread('backup_face.jpg')
            face_rectangles = detector(arr_frame, 2)
            largest_face_rectangle = util.find_largest_face(face_rectangles)
            landmarks = shape_to_np(predictor(arr_frame, largest_face_rectangle))

        landmarks = np.ndarray.reshape(landmarks, (number_landmarks * 2))
        # print(landmarks)
        random_frame_landmarks_from_all_videos[i] = landmarks

    # save landmark to file
    action = 'a'
    file_path = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/landmark_template.txt'
    if not os.path.exists(file_path):
        action = 'w'
    with open(file_path, action) as my_file:
        for k in range(b, e):
            landmark_arr = random_frame_landmarks_from_all_videos[k]
            for lm in range(number_landmarks):
                my_file.write('%d' % landmark_arr[lm])
                if not lm == number_landmarks - 1:
                    my_file.write(',')
            my_file.write('\n')

    # compute mean landmark so far
    mean_landmark = np.mean(random_frame_landmarks_from_all_videos[1100:e], axis=0).astype(int)
    mean_landmark = np.reshape(mean_landmark, (68, 2))

    # draw the landmarks
    canvas = np.ones((height, width, 3)).astype(np.uint8)
    canvas *= 255
    for p in mean_landmark:
        x, y = p
        canvas[y, x] = [0, 0, 0]

    img = Image.fromarray(canvas, mode='RGB')
    template_folder = '/home/gabi/PycharmProjects/visualizing-traits/src/align_face/template_folder'
    template_name = 'TEMPLATE_%d_%d.jpg' % (0, e)
    if not os.path.exists(template_folder):
        os.mkdir(template_folder)
    img.save(os.path.join(template_folder, template_name))
    img.show()
