import cv2
import os
import sys
import numpy as np


EXPERIMENT_DIRECTORY = sys.argv[1]

OTSU_MIN = 30
OTSU_MAX = 255

CANNY_MIN = 30
CANNY_MAX = 200


def demo_segment(path):
    filenames = os.listdir(path)
    for f in filenames:
        # Open as grayscale
        gray = cv2.imread(os.path.join(path, f), 0)
        base = '.'.join(f.split('.')[:-1])

        cv2.imwrite('results/{0}_gray.jpg'.format(base), gray)

        ret, thresh = cv2.threshold(
            gray,
            OTSU_MIN,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        cv2.imwrite('results/{0}_otsu.jpg'.format(base), thresh)

        edges = cv2.Canny(gray, 30, 200, apertureSize=5, L2gradient=True)

#        kernel = np.array([
#            [1, 1, 1, 1],
#            [1, 1, 1, 1],
#            [1, 1, 1, 1],
#            [1, 1, 1, 1]
#        ])
        closed = edges # cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('results/{0}_canny.jpg'.format(base), closed)

        orig_color = cv2.imread(os.path.join(path, f), 1)
        color = orig_color.reshape((-1, 3))

        color = np.float32(color)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0
        )
        K = 8
        ret, label, center = cv2.kmeans(
            color, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((orig_color.shape))

        cv2.imwrite('results/{0}_kmeans.jpg'.format(base), res2)


demo_segment(EXPERIMENT_DIRECTORY)
