import cv2
from skimage import color
import numpy as np

from collections import namedtuple

LabImage = namedtuple('LabImage', ['l', 'a', 'b'])


def open_lab_image(path):
    # The 1 tells openCV imread to get rgb channels
    LOAD_RGB_IMAGE = 1
    rgb = cv2.imread(path, LOAD_RGB_IMAGE)
    lab_image = color.rgb2lab(rgb)

    # NOTE the maximum values of l, a, b are 100, +/- 128, +/- 128
    lab_image[:, :, 0] *= (2.55)
    lab_image[:, :, 1] += 128.0
    lab_image[:, :, 2] += 128.0

    # Convert to 8bit so algorithms can work with it easily
    lab_8bit = lab_image.astype(np.uint8)
    return LabImage(
        l=lab_8bit[:, :, 0],
        a=lab_8bit[:, :, 1],
        b=lab_8bit[:, :, 2]
    )

if __name__ == '__main__':
    lab = open_lab_image('/home/peter/streetview/tmp/hsb_test.jpg')
    cv2.imwrite('test_l.jpg', lab.l)
    cv2.imwrite('test_a.jpg', lab.a)
    cv2.imwrite('test_b.jpg', lab.b)
