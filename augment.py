import cv2
import numpy as np
import random


class CenterResizeCrop(object):
    def __init__(self, scale_begin=23, windowsize=27):
        self.scale_begin = scale_begin
        self.windowsize = windowsize

    def __call__(self, image):
        length = np.array(range(self.scale_begin, self.windowsize + 1, 2))
        row_center = int((self.windowsize - 1) / 2)
        col_center = int((self.windowsize - 1) / 2)
        row = image.shape[1]
        col = image.shape[2]

        # ✅ FIX
        s = int(np.random.choice(length))
        halfsize_row = int((s - 1) / 2)
        halfsize_col = int((s - 1) / 2)

        r_image = image[:, row_center-halfsize_row:row_center+halfsize_row+1,
                        col_center-halfsize_col:col_center+halfsize_col+1]
        r_image_transpose = cv2.resize(np.transpose(r_image, [1, 2, 0]), (row, col))
        if r_image_transpose.ndim != 3:
            r_image_transpose = np.expand_dims(r_image_transpose, 2)
        r_image = np.transpose(r_image_transpose, [2, 0, 1])
        return r_image


class RandomResizeCrop(object):
    def __init__(self, scale=[0.5, 1], probability=0.5):
        self.scale = scale
        self.probability = probability

    def __call__(self, image):
        if random.uniform(0, 1) > self.probability:
            return image
        row = image.shape[1]
        col = image.shape[2]
        s = np.random.uniform(self.scale[0], self.scale[1])
        r_row = round(row * s)
        r_col = round(col * s)
        halfsize_row = int((r_row - 1) / 2)
        halfsize_col = int((r_col - 1) / 2)
        row_center = random.randint(halfsize_row, r_row - halfsize_row - 1)
        col_center = random.randint(halfsize_col, r_col - halfsize_col - 1)
        r_image = image[:, row_center-halfsize_row:row_center+halfsize_row+1,
                        col_center-halfsize_col:col_center+halfsize_col+1]
        r_image = np.transpose(cv2.resize(np.transpose(r_image, [1, 2, 0]), (row, col)), [2, 0, 1])
        return r_image


class CenterCrop(object):
    def __init__(self, scale_begin=23, windowsize=27, probability=0.5):
        self.scale_begin = scale_begin
        self.windowsize = windowsize
        self.probability = probability

    def __call__(self, image):
        if random.uniform(0, 1) > self.probability:
            return image

        length = np.array(range(self.scale_begin, self.windowsize, 2))
        row_center = int((self.windowsize - 1) / 2)
        col_center = int((self.windowsize - 1) / 2)

        # ✅ FIX
        s = int(np.random.choice(length))
        halfsize_row = int((s - 1) / 2)
        halfsize_col = int((s - 1) / 2)

        r_image = image[:, row_center-halfsize_row:row_center+halfsize_row+1,
                        col_center-halfsize_col:col_center+halfsize_col+1]
        r_image = np.pad(
            r_image,
            ((0, 0),
             (row_center - halfsize_row, row_center - halfsize_row),
             (col_center - halfsize_col, col_center - halfsize_col)),
            'constant',
            constant_values=0
        )
        return r_image
