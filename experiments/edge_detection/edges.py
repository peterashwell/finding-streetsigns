import cv2
import os
import sys
import numpy as np
from skimage import color as skcolor

from bisect import bisect_left


EXPERIMENT_DIRECTORY = sys.argv[1]
TRAINING_PATH = sys.argv[2]

OTSU_MIN = 30
OTSU_MAX = 255

CANNY_MIN = 30
CANNY_MAX = 200

MINIMUM_WIDTH = 30
MAXIMUM_WIDTH = 80
RATIO_HEIGHT_TO_WIDTH = 2.25

SIGN_EDGE_RATIO = 0.1


class Convolution:
    def __init__(self, row, col, score, region, convolution):
        self.row = row
        self.col = col
        self.score = score
        self.region = region
        self.convolution = convolution


def open_grayscale_image(path):
    CV2_AS_GRAYSCALE = 0
    image = cv2.imread(path, CV2_AS_GRAYSCALE)
    return image


def align_image_template(image, template):
    image = np.copy(image)
    template = np.copy(template)
    template_width = template.shape[1]
    template_height = template.shape[0]
    template_count = np.float32(np.sum(template))
    col_slots = image.shape[1] - template.shape[1]
    row_slots = image.shape[0] - template.shape[0]

    bests = []
    find_matrix = np.zeros_like(image)
    found_threshold = template_width * template_height * 0.8
    for row in range(row_slots):
        for col in range(col_slots):
            img_slice = image[row: row + template_height, col: col + template_width]

            convol = np.multiply(img_slice, template)
            convol_sum = np.sum(convol) / template_count

            scores = [c.score for c in bests]
            bisect_point = bisect_left(scores, convol_sum)
            bests.insert(bisect_point, Convolution(row, col, convol_sum, img_slice, convol))
            if len(bests) > 10:
                bests.pop(0)
                if (bisect_point != 0):
                    find_slice = find_matrix[row: row + template_height, col: col + template_width]
                    find_slice += 1
                    if len(np.where(find_slice >= 3)[0]) > found_threshold:
                        img_slice[find_slice >= 3] = 0
    return bests

def get_segmented_images(allimagespath):
    signs = get_signshaped_rects_fixed()
    filenames = os.listdir(allimagespath)
    for f in filenames[0:10]:
        base = '.'.join(f.split('.')[:-1])
        fullimgpath = os.path.join(allimagespath, f)
        segmented, lightness, achan = segment_one_at_path(fullimgpath)
        color_query_image = cv2.imread(fullimgpath)


        #print('template shape:', scaled_template.shape[::-1])
        for matchnum, sign in enumerate(signs):
            print('aligning:', sign, matchnum)
            #print('matching:', sign.depth(), 'to', segmented.depth())
            if sign.shape[0] >= segmented.shape[0] or sign.shape[1] >= segmented.shape[1]:
                continue
            #matches = cv2.matchTemplate(
            #    np.float32(segmented), np.float32(sign), cv2.TM_CCORR_NORMED
            #)
            best_convols = align_image_template(segmented, sign)
            for bcn, best_convol in enumerate(best_convols):
                #cv2.imwrite('bestconvols/{1}{0}_template.jpg'.format(matchnum, bcn), sign * 255)
                #cv2.imwrite('bestconvols/{1}{0}_bestslice.jpg'.format(matchnum, bcn), best_convol.region * 255)
                #cv2.imwrite('bestconvols/{1}{0}_bestconvol.jpg'.format(matchnum, bcn), best_convol.convolution * 255)
                #if matches.max() < 0.70:
                #    continue
                tw, th = sign.shape
                #pts = np.where(matches > 0.5)
                #print('pts:', pts)
                #print('value:', matches.max())
                #for row in range(pts[0].shape[0]):
                #    ptx = int(pts[0][row])
                #    pty = int(pts[1][row])
                #    tl = ptx + th
                #    tr = pty + tw

                # Adjust to center the edge
                ptx = best_convol.col
                pty = best_convol.row
                tl = ptx + sign.shape[1]
                tr = pty + sign.shape[0]
                lightness_or_achan = np.logical_or(lightness, achan)
                candidate_slot = lightness_or_achan[pty: tr, ptx: tl]
                #cv2.imwrite('bestconvols/{1}{0}_bestlora.jpg'.format(matchnum, bcn), candidate_slot * 255)
                candidate_score = np.sum(candidate_slot) / (1.0 * sign.shape[0] * sign.shape[1])
                #print('candidate score:', candidate_score)
                if candidate_score > 0.5:
                    cv2.rectangle(color_query_image, (ptx, pty), (tl, tr), (255, 0, 0), 2)
        cv2.imwrite('results/{0}_lightnessorachan.jpg'.format(base), np.logical_or(lightness, achan) *255)
        cv2.imwrite('results/{0}_edges.jpg'.format(base), segmented*255)
        cv2.imwrite('results/{0}_template.jpg'.format(base), color_query_image)


def get_segmented_signs():
    segmented_signs = []
    for train_fname in os.listdir(TRAINING_PATH):
        image_path = os.path.join(TRAINING_PATH, train_fname)
        color = cv2.imread(image_path, 1)
        gray = cv2.imread(image_path, 0)
        ratio = color.shape[0] / (1.0 * color.shape[1])
        print('loading training image:', image_path)
        segmented = segment_one(gray, color)[0]
        for desired_width in range(MINIMUM_WIDTH, 160, 10):
            desired_width = int(desired_width)
            desired_height = int(desired_width * ratio)
            print('scaling to:', desired_width, desired_height)
            scaled_segmented = cv2.resize(segmented, (desired_width, desired_height))
            scaled_segmented[scaled_segmented < 0.3] = 0
            scaled_segmented[scaled_segmented >= 0.3] = 1
            segmented_signs.append(scaled_segmented)
            cv2.imwrite('segsigns/{0}'.format(train_fname), scaled_segmented*255)
    return segmented_signs


def get_signshaped_rects():
    segmented_signs = []
    for train_fname in os.listdir(TRAINING_PATH):
        image_path = os.path.join(TRAINING_PATH, train_fname)
        #color = cv2.imread(image_path, 1)
        gray = cv2.imread(image_path, 0)
        ratio = gray.shape[0] / (1.0 * gray.shape[1])
        print('loading training image:', image_path)
        for sn, desired_width in enumerate(range(MINIMUM_WIDTH, MAXIMUM_WIDTH, 5)):
            desired_width = int(desired_width)
            desired_height = int(desired_width * ratio)
            template = np.ones((desired_height, desired_width))
            bs = int(desired_width * SIGN_EDGE_RATIO)
            template[bs: -bs, bs: -bs] = 0
            segmented_signs.append(template)
            base = ''.join(train_fname.split('.')[:-1])
            cv2.imwrite('segsigns/{0}{1}.jpg'.format(base, sn), template*255)
    return segmented_signs


def get_signshaped_rects_fixed():
    segmented_signs = []
    ratio = 2.25
    for sn, desired_width in enumerate(range(MINIMUM_WIDTH, MAXIMUM_WIDTH, 5)):
        desired_width = int(desired_width)
        desired_height = int(desired_width * ratio)
        template = np.ones((desired_height, desired_width))
        bs = int(desired_width * 0.1)
        template[bs: -bs, bs: -bs] = 0
        segmented_signs.append(template)
        cv2.imwrite('segsigns/{0}.jpg'.format(sn), template*255)
    return segmented_signs


def segment_one_at_path(imgpath):
    return segment_one(cv2.imread(imgpath, 0), cv2.imread(imgpath, 1))


def segment_one(gray, color):
    edges = cv2.Canny(gray, 1000, 2500, apertureSize=5, L2gradient=True)
    edges[edges != 0] = 1
    kernel_22 = np.ones((2, 2), np.uint8)
    kernel_33 = np.ones((3, 3), np.uint8)

    lab = skcolor.rgb2lab(color)

    lightness = lab[:, :, 0]
    lightness[lightness < 60] = 0
    lightness[lightness >= 60] = 1

    a_chan = lab[:, :, 1]
    a_chan[a_chan < 15] = 0
    a_chan[a_chan >= 15] = 1

    combined = np.zeros(a_chan.shape)
    combined[a_chan != 0] = 1
    combined[lightness != 0] = 1
    combined = cv2.dilate(combined, kernel_33, iterations=3)
    combined = np.multiply(combined, edges)
    combined = cv2.dilate(combined, kernel_22, iterations=4)

    return combined, lightness, a_chan


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
        kernel_22 = np.ones((2, 2), np.uint8)
        kernel_33 = np.ones((3, 3), np.uint8)
        binary = np.copy(edges)
        binary[binary == 0] = 0
        binary[binary != 0] = 1
        binary = cv2.dilate(binary, kernel_33, iterations=2)
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
        combined = cv2.dilate(combined, kernel_33, iterations=3)
        cv2.imwrite('results/{0}_zkombined1.jpg'.format(base), combined * 255)
        combined = np.multiply(combined, edges)
        cv2.imwrite('results/{0}_zkombined2.jpg'.format(base), combined * 255)
        combined = cv2.dilate(combined, kernel_22, iterations=3)
        cv2.imwrite('results/{0}_zkombined3.jpg'.format(base), combined * 255)


#get_segmented_signs()
#demo_segment(EXPERIMENT_DIRECTORY)
get_segmented_images(EXPERIMENT_DIRECTORY)
