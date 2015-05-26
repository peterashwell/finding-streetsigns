import cv2
from skimage import color


def open_lab_image(path):
    # The 1 tells openCV we want an RGB image
    rgb = cv2.imread(path, 1)
    lab_image = color.rgb2lab(rgb)
    cv2.imwrite('l.jpg', lab_image[:, :, 0])
    cv2.imwrite('a.jpg', lab_image[:, :, 1])
    cv2.imwrite('b.jpg', lab_image[:, :, 2])

if __name__ == '__main__':
    open_lab_image('/home/peter/streetview/tmp/hsb_test.jpg')
