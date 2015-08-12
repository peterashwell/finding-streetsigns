import cv2
import os
import sys
import numpy as np
from skimage import color


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

        #cv2.imwrite('results/{0}_gray.jpg'.format(base), gray)

        #ret, thresh = cv2.threshold(
        #    gray,
        #    OTSU_MIN,
        #    255,
        #    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        #)
        #cv2.imwrite('results/{0}_otsu.jpg'.format(base), thresh)

        edges = cv2.Canny(gray, 1000, 2500, apertureSize=5, L2gradient=True)
        kernel = np.ones((3, 3), np.uint8)
        binary = np.copy(edges)
        binary[binary == 0] = 0
        binary[binary != 0] = 1
        binary = cv2.dilate(binary, kernel, iterations=2)
        binary = binary * 255


#        kernel = np.array([
#            [1, 1, 1],
#            [1, 1, 1],
#            [1, 1, 1],
#        ])
        cv2.imwrite('results/{0}_before.jpg'.format(base), gray)
        #cv2.imwrite('results/{0}_canny.jpg'.format(base), edges)
        #cv2.imwrite('results/{0}_fat.jpg'.format(base), binary)

        orig_color = cv2.imread(os.path.join(path, f), 1)
        lab = color.rgb2lab(orig_color)

        lightness = lab[:, :, 0]
        #print('lightness max:', np.max(lightness))
        #print('lightness min:', np.min(lightness))
        lightness[lightness < 60] = 0
        lightness[lightness >= 60] = 255
        cv2.imwrite('results/{0}_lightness.jpg'.format(base), lightness)

        a_chan = lab[:, :, 1]
        #print('a min:', np.max(a_chan))
        #print('a max:', np.min(a_chan))
        a_chan[a_chan < 15] = 0
        a_chan[a_chan > 15] = 255
        cv2.imwrite('results/{0}_laba.jpg'.format(base), a_chan)

        combined = np.zeros(a_chan.shape)
        combined[a_chan != 0] = 1
        combined[lightness != 0] = 1
        combined = cv2.dilate(combined, kernel, iterations=5)
        cv2.imwrite('results/{0}_zkombined1.jpg'.format(base), combined * 255)
        combined = np.multiply(combined, edges)
        cv2.imwrite('results/{0}_zkombined2.jpg'.format(base), combined * 255)


demo_segment(EXPERIMENT_DIRECTORY)
