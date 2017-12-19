# import the necessary packages
from .helpers import FACIAL_LANDMARKS_IDXS
from .helpers import shape_to_np
from .helpers import get_bbox
import numpy as np
import cv2
from PIL import Image
# from skimage.transform import SimilarityTransform, ProjectiveTransform
# from skimage import transform
from scipy.misc import imshow, imsave


class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35), desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        # horizontal mouth position
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align_to_template_similarity(self, image, gray, rect):
        # example template. Just something I came up with
        template = {'mouth': [48, 66],
                    'left_eye': [32, 33],
                    'right_eye': [64, 33],
                    'nose': [48, 54]}

        shape = shape_to_np(self.predictor(gray, rect))

        (l_start, l_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (r_start, r_end) = FACIAL_LANDMARKS_IDXS["right_eye"]
        (m_start, m_end) = FACIAL_LANDMARKS_IDXS['mouth']
        (n_start, n_end) = FACIAL_LANDMARKS_IDXS['nose']

        left_eye_points = shape[l_start:l_end]
        right_eye_points = shape[r_start:r_end]
        mouth_points = shape[m_start:m_end]
        nose_points = shape[n_start: n_end]

        left_eye_bbox = get_bbox(left_eye_points)
        left_eye_center = [(left_eye_bbox[0][0] + left_eye_bbox[1][0]) / 2,
                           (left_eye_bbox[2][1] + left_eye_bbox[3][1]) / 2]
        right_eye_bbox = get_bbox(right_eye_points)
        right_eye_center = [(right_eye_bbox[0][0] + right_eye_bbox[1][0]) / 2,
                            (right_eye_bbox[2][1] + right_eye_bbox[3][1]) / 2]
        mouth_bbox = get_bbox(mouth_points)
        mouth_center = [(mouth_bbox[0][0] + mouth_bbox[1][0]) / 2, (mouth_bbox[2][1] + mouth_bbox[3][1]) / 2]
        
        nose_bbox = get_bbox(nose_points)
        nose_center = [(nose_bbox[0][0] + nose_bbox[1][0]) / 2, (nose_bbox[2][1] + nose_bbox[3][1]) / 2]

        pts1 = np.float32([left_eye_center, right_eye_center, mouth_center, nose_center])
        pts2 = np.float32([template['left_eye'], template['right_eye'], template['mouth'], template['nose']])

        # pts1 = np.float32([left_eye_center, right_eye_center, mouth_center, nose_center])
        # pts2 = np.float32([template['left_eye'], template['right_eye'], template['mouth'], template['nose']])

        # sort of works
        tf = SimilarityTransform()
        tf.estimate(pts1, pts2)
        print(tf.params)
        print('scale: %s\ntranslate: %s\nrotation: %s' % (str(tf.scale), str(tf.translation), str(tf.rotation)))

        tf2 = SimilarityTransform(rotation=tf.rotation, translation=tf.translation, scale=tf.scale)
        result = transform.warp(image, inverse_map=tf2.inverse, output_shape=(96, 96),  preserve_range=False)
        imshow(result)

        tf2 = SimilarityTransform(rotation=tf.rotation, translation=[60, 60], scale=0.8)
        result = transform.warp(image, inverse_map=tf2.inverse, output_shape=(96, 96), preserve_range=False)
        imshow(result)

        tf2 = SimilarityTransform(rotation=tf.rotation, translation=[-100, 0], scale=0.8)
        result = transform.warp(image, inverse_map=tf2.inverse, output_shape=(96, 96), preserve_range=False)
        imshow(result)

        tf2 = SimilarityTransform(rotation=tf.rotation, translation=[1, 10], scale=0.8)
        result = transform.warp(image, inverse_map=tf2.inverse, output_shape=(96, 96), preserve_range=False)
        imshow(result)

        # tf = cv2.estimateRigidTransform(pts1, pts2, fullAffine=False)
        # result = cv2.warpAffine(image, tf, (self.desiredFaceWidth, self.desiredFaceHeight))
        # img = Image.fromarray(result, mode='RGB')
        # img.show()

        return result

    def align_to_template_affine(self, image, gray, rect):
        # align by affine warping transform image to hard set landmark locations on a template

        # example template. Just something I came up with
        template = {'mouth': [48, 66],
                    'left_eye': [32, 33],
                    'right_eye': [64, 33]}

        shape = shape_to_np(self.predictor(gray, rect))

        (l_start, l_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (r_start, r_end) = FACIAL_LANDMARKS_IDXS["right_eye"]
        (m_start, m_end) = FACIAL_LANDMARKS_IDXS['mouth']

        left_eye_points = shape[l_start:l_end]
        right_eye_points = shape[r_start:r_end]
        mouth_points = shape[m_start:m_end]

        left_eye_bbox = get_bbox(left_eye_points)
        left_eye_center = [(left_eye_bbox[0][0] + left_eye_bbox[1][0]) / 2,
                                    (left_eye_bbox[2][1] + left_eye_bbox[3][1]) / 2]
        right_eye_bbox = get_bbox(right_eye_points)
        right_eye_center = [(right_eye_bbox[0][0] + right_eye_bbox[1][0]) / 2,
                                     (right_eye_bbox[2][1] + right_eye_bbox[3][1]) / 2]
        mouth_bbox = get_bbox(mouth_points)
        mouth_center = [(mouth_bbox[0][0] + mouth_bbox[1][0]) / 2, (mouth_bbox[2][1] + mouth_bbox[3][1]) / 2]
            
        pts1 = np.float32([left_eye_center, right_eye_center, mouth_center])
        pts2 = np.float32([template['left_eye'], template['right_eye'], template['mouth']])
        affine_transform_matrix = cv2.getAffineTransform(pts1, pts2)

        result = cv2.warpAffine(image, affine_transform_matrix, (self.desiredFaceWidth, self.desiredFaceHeight))

        return result

    def align_center(self, image, gray, rect, radius):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # extract the left and right eye (x, y)-coordinates
        (l_start, l_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (r_start, r_end) = FACIAL_LANDMARKS_IDXS["right_eye"]
        (m_start, m_end) = FACIAL_LANDMARKS_IDXS['mouth']
        (j_start, j_end) = FACIAL_LANDMARKS_IDXS['jaw']

        left_eye_points = shape[l_start:l_end]
        right_eye_points = shape[r_start:r_end]
        mouth_points = shape[m_start:m_end]
        jaw_points = shape[j_start:j_end]

        # compute the geometrical center of each eye
        left_eye_bbox = get_bbox(left_eye_points)
        left_eye_center = np.array([(left_eye_bbox[0][0] + left_eye_bbox[1][0]) / 2,
                                    (left_eye_bbox[2][1] + left_eye_bbox[3][1]) / 2])
        right_eye_bbox = get_bbox(right_eye_points)
        right_eye_center = np.array([(right_eye_bbox[0][0] + right_eye_bbox[1][0]) / 2,
                                     (right_eye_bbox[2][1] + right_eye_bbox[3][1]) / 2])
        mouth_bbox = get_bbox(mouth_points)
        mouth_center = np.array([(mouth_bbox[0][0] + mouth_bbox[1][0]) / 2, (mouth_bbox[2][1] + mouth_bbox[3][1]) / 2])

        center_pixel = np.mean([left_eye_center, right_eye_center, mouth_center], axis=0, dtype=int)
        jaw_bbox = get_bbox(jaw_points)
        lowest_face_y = jaw_bbox[3][1]

        if radius == 0:
            radius = lowest_face_y - center_pixel[1]
        top_left = [center_pixel[0] - radius, center_pixel[1] - radius]
        bottom_right = [center_pixel[0] + radius, center_pixel[1] + radius]
        img = Image.fromarray(image)
        cropped_image = img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1]))

        resized_image = cropped_image.resize((self.desiredFaceWidth, self.desiredFaceHeight))
        return np.array(resized_image, dtype='uint8'), radius

    def align_geometric_eyes(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        the_real_shape = np.shape(image)

        # extract the left and right eye (x, y)-coordinates
        (l_start, l_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (r_start, r_end) = FACIAL_LANDMARKS_IDXS["right_eye"]
        (m_start, m_end) = FACIAL_LANDMARKS_IDXS['mouth']

        left_eye_points = shape[l_start:l_end]
        right_eye_points = shape[r_start:r_end]
        mouth_points = shape[m_start:m_end]
        
        # print('lefT_eye_point: %s' % str(left_eye_points))
        # print('right_eye_point: %s' % str(right_eye_points))
        # print('mouth_point: %s' % str(mouth_points))

        # compute the center of mass for each eye
        # left_eye_center = left_eye_points.mean(axis=0).astype("int")
        # right_eye_center = right_eye_points.mean(axis=0).astype("int")

        # compute the geometrical center of each eye
        left_eye_bbox = get_bbox(left_eye_points)
        left_eye_center = np.array([(left_eye_bbox[0][0] + left_eye_bbox[1][0]) / 2, 
                                    (left_eye_bbox[2][1] + left_eye_bbox[3][1]) / 2])
        right_eye_bbox = get_bbox(right_eye_points)
        right_eye_center = np.array([(right_eye_bbox[0][0] + right_eye_bbox[1][0]) / 2,
                                    (right_eye_bbox[2][1] + right_eye_bbox[3][1]) / 2])
        
        # compute the geometrical center of the mouth
        mouth_bbox = get_bbox(mouth_points)
        mouth_center = np.array([(mouth_bbox[0][0] + mouth_bbox[1][0]) / 2, (mouth_bbox[2][1] + mouth_bbox[3][1]) / 2])

        # compute the angle between the eye centroids
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((left_eye_center[0] + right_eye_center[0]) // 2,
                      (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        def pretend_image(ship, h, w):
            new_array = np.zeros((h, w, 3))
            for pixel in ship:
                new_array[pixel[1]][pixel[0]] = 255
            return new_array

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        shape = pretend_image(shape, the_real_shape[0], the_real_shape[1])

        # shape = np.array(shape, dtype='uint8')
        # img = Image.fromarray(shape, mode='RGB')
        # img.save('/home/gabi/PycharmProjects/imutils/testing/just_landmarks.jpg')

        warped_landmarks = cv2.warpAffine(shape, M, (w, h), flags=cv2.INTER_CUBIC)
        # new_shape = get_shape_from_pretend()
        original_shape = np.shape(warped_landmarks)
        warped_landmarks = np.ndarray.flatten(warped_landmarks)
        length = len(warped_landmarks)
        warped_landmarks = np.array([255 if warped_landmarks[i] > 0 else 0 for i in range(length)], dtype='uint8')
        warped_landmarks = np.reshape(warped_landmarks, original_shape)
        # warped_landmarks = np.array(warped_landmarks, dtype='uint8')
        # img = Image.fromarray(warped_landmarks, mode='RGB')
        # img.save('/home/gabi/PycharmProjects/imutils/testing/warped_landmarks.jpg')

        new_list = []
        warped_landmarks = warped_landmarks[:, :, 0]
        for x in range(np.shape(warped_landmarks)[0]):
            for y in range(np.shape(warped_landmarks)[1]):
                if warped_landmarks[x][y] > 0:
                    new_list.append([x, y])

        warped_landmarks = np.array(new_list)

        # return the aligned face
        return output, [left_eye_points, right_eye_points, mouth_points], warped_landmarks

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)

        # extract the left and right eye (x, y)-coordinates
        (l_start, l_end) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (r_start, r_end) = FACIAL_LANDMARKS_IDXS["right_eye"]
        left_eye_points = shape[l_start:l_end]
        right_eye_points = shape[r_start:r_end]

        # compute the center of mass for each eye
        left_eye_center = left_eye_points.mean(axis=0).astype("int")
        right_eye_center = right_eye_points.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((left_eye_center[0] + right_eye_center[0]) // 2,
                      (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output