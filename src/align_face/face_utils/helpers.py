# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
from PIL import Image
from src.align_face import project_paths as pp
# import src.align_face.project_paths as pp


# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        # check if are supposed to draw the jawline
        if name == "jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    return output


def get_bbox(landmark_points):
    # take landmark and return the outermost pixels
    left_pixel = [0, 0]
    right_pixel = [0, 0]
    top_pixel = [0, 0]
    bot_pixel = [0, 0]
    min_x, max_x, min_y, max_y = 1000000, 0, 1000000, 0

    for i in landmark_points:
        if i[0] > max_x:
            right_pixel = i
            max_x = i[0]
        if i[0] < min_x:
            left_pixel = i
            min_x = i[0]
        if i[1] > max_y:
            bot_pixel = i
            max_y = i[1]
        if i[1] < min_y:
            top_pixel = i
            min_y = i[1]

    return left_pixel, right_pixel, top_pixel, bot_pixel


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_template_landmark():
    file_path = pp.TEMPLATE
    template = list(np.genfromtxt(file_path, dtype=str))
    num_landmarks = len(template)
    template_arr = np.zeros((num_landmarks, 2), dtype='int')
    for i in range(num_landmarks):
        x, y = template[i].strip().split(',')
        template_arr[i] = [int(x), int(y)]

    return template_arr


def get_bbox_template():
    template_arr = get_template_landmark()
    left = np.min([i[0] for i in template_arr])
    right = np.max([i[0] for i in template_arr])
    top = np.min([i[1] for i in template_arr])
    bottom = np.max([i[1] for i in template_arr])
    print(left, right, top, bottom)
    offset_horizontal = left
    offset_vertical = top
    side1 = right + offset_horizontal
    side2 = bottom + offset_vertical
    print(side1, side2, side1 == side2)

    canvas = np.ones((side1, side2, 3)).astype(np.uint8)
    canvas *= 255

    for p in template_arr:
        x, y = int(p[0]), int(p[1])
        canvas[y, x] = [0, 0, 0]
        canvas[y - 1, x] = [0, 0, 0]
        canvas[y - 1, x + 1] = [0, 0, 0]
        canvas[y, x + 1] = [0, 0, 0]
        canvas[y + 1, x + 1] = [0, 0, 0]
        canvas[y + 1, x] = [0, 0, 0]
        canvas[y + 1, x - 1] = [0, 0, 0]
        canvas[y, x - 1] = [0, 0, 0]
        canvas[y - 1, x - 1] = [0, 0, 0]

    canvas = Image.fromarray(canvas, mode='RGB')
    canvas.save('cropped_template_2.jpg')
