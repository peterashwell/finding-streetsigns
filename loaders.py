import cv2
from skimage import color
import numpy as np


def open_lab_image(path):
    # The 1 tells openCV we want an RGB image
    rgb = cv2.imread(path, 1)
    lab_image = color.rgb2lab(rgb)
    return (lab_image[:, :, 0] / 100 * 255).astype(np.uint8)

if __name__ == '__main__':
    luminosity = open_lab_image('/home/peter/streetview/tmp/hsb_test.jpg')
    cv2.imwrite('test_luminosity.png', luminosity)
