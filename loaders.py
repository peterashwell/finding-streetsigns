from skimage import color
import numpy as np
import imread
import cv2

from collections import namedtuple

LabImage = namedtuple('LabImage', ['l', 'a', 'b'])
HsvImage = namedtuple('HsvImage', ['h', 's', 'v'])


def open_grayscale_image(path):
    CV2_AS_GRAYSCALE = 0
    image = cv2.imread(path, CV2_AS_GRAYSCALE)
    return image


def open_lab_image(path):
    # The 1 tells openCV imread to get rgb channels
    rgb = imread.imread(path)
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


def open_hsv_image(path):
    rgb = imread.imread(path)
    hsv_image = color.rgb2hsv(rgb)

    hsv_image *= 255
    hsv_image = hsv_image.astype(np.uint8)

    h = hsv_image[:, :, 0]
    s = hsv_image[:, :, 1]
    v = hsv_image[:, :, 2]

    return HsvImage(h=h, s=s, v=v)

if __name__ == '__main__':
    hsv = open_hsv_image('/home/peter/streetview/tmp/hsb_test.jpg')
    imread.imsave('test_h.jpg', hsv.h)
    imread.imsave('test_s.jpg', hsv.s)
    imread.imsave('test_v.jpg', hsv.v)

    # lab = open_lab_image('/home/peter/streetview/tmp/hsb_test.jpg')
    # imread.imsave('test_l.jpg', lab.l)
    # imread.imsave('test_a.jpg', lab.a)
    # imread.imsave('test_b.jpg', lab.b)
