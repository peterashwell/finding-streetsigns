# We are using python2 because OpenCV compatibility for python3 sucks
# Continue using a python3 linter however, so fix print statements
from __future__ import print_function

import cv2
import numpy as np
import os
import sys

from sift_wrapper import SiftWrapper
from loaders import open_lab_image

MIN_MATCH_COUNT = 3
# Lower - more specifity for matches
THRESHOLD = 0.6

TRAINING_PATH = sys.argv[1]
QUERY_PATH = sys.argv[2]
RESULT_PATH = sys.argv[3]

training_images = os.listdir(TRAINING_PATH)
query_images = os.listdir(QUERY_PATH)

one_train_image = training_images[2]
one_query_image = query_images[13]

sw = SiftWrapper()

training_map = {}
for train_fname in training_images:
    image_path = os.path.join(TRAINING_PATH, train_fname)
    image = open_lab_image(image_path).l
    training_map[train_fname] = sw.do_sift(image)

for qnum, query_fname in enumerate(query_images):
    print("reading {0}".format(query_fname))
    training_hits = 0
    query_image_path = os.path.join(QUERY_PATH, query_fname)
    # Get 'a' component of lab image
    query_image = open_lab_image(query_image_path).l
    output_image = np.copy(query_image)

    sift_query = sw.do_sift(query_image)
    for train_fname in training_map.keys():
        sift_train = training_map[train_fname]

        knn_result = sw.do_knn_sift(sift_train, sift_query)

        if knn_result.num_found > MIN_MATCH_COUNT:
            training_hits += 1
            print("matched: {0}".format(train_fname))
            src_pts = np.float32(knn_result.source_points).reshape(-1, 1, 2)
            dst_pts = np.float32(knn_result.destination_points).reshape(
                -1, 1, 2
            )

            for dst_pt in dst_pts:
                cv2.circle(
                    output_image, tuple(dst_pt[0]), 5, (255, 255, 255), -1
                )
    if training_hits > 0:
        # Write each query image out with markers from training images
        # NOTE query_fname includes .jpg extension
        output_path = os.path.join(RESULT_PATH, query_fname)
        cv2.imwrite(output_path, output_image)
