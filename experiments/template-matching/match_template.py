from __future__ import print_function

import os
import sys
import cv2

import numpy as np

THRESHOLDED_IMAGES_PATH = sys.argv[1]

TEMPLATE_IMAGE_PATH = 'template.jpg'
RESULTS_PATH = 'results'

AS_GRAYSCALE = 0
MATCH_THRESHOLD = 0.7
MATCHING_METHOD = cv2.TM_SQDIFF_NORMED
LIMIT_AMOUNT = 10

template_image = cv2.imread(TEMPLATE_IMAGE_PATH, AS_GRAYSCALE)

query_images = os.listdir(THRESHOLDED_IMAGES_PATH)
limited_query_images = query_images[:LIMIT_AMOUNT]

SCALES = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5]
for query_fname in limited_query_images:
    query_path = os.path.join(THRESHOLDED_IMAGES_PATH, query_fname)
    query_image = cv2.imread(query_path, AS_GRAYSCALE)
    color_query_image = cv2.imread(query_path)

    #print('query shape:', query_image.shape[::-1])

    best_score = None
    for scale in SCALES:
        scaled_template = cv2.resize(template_image, (0, 0), fx=scale, fy=scale)

        #print('template shape:', scaled_template.shape[::-1])
        matches = cv2.matchTemplate(query_image, scaled_template, cv2.TM_CCOEFF_NORMED)

        #print('matches shape:', matches.shape[::-1])

        # print('qfname:', query_fname.split('.'))
        qf = ''.join(query_fname.split('.')[:-1])
        #print("qf:", qf)
        cv2.imwrite(os.path.join(RESULTS_PATH, '{0}_proc_{1}.jpg'.format(qf, scale)), matches * 255)

        #loc = np.where(np.abs(matches) >= MATCH_THRESHOLD)

        print('matches:', matches)
        if best_score is None or best_score < matches.max():
            print("new best:", best_score)
            best_score = matches.max()
            best = np.where(matches==matches.max())
            best_template = scaled_template

    for pt in zip(*best[::-1]):
        print("match at pt:", pt)
        tw, th = best_template.shape[::-1]
        tl = int(pt[0] + (tw / 2.0))
        tr = int(pt[1] + (th / 2.0))
        cv2.rectangle(color_query_image, pt, (tl, tr), (255, 0, 0), 2)

        cv2.imwrite(os.path.join(RESULTS_PATH, query_fname), color_query_image)
