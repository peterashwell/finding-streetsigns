# We are using python2 because OpenCV compatibility for python3 sucks
# Continue using a python3 linter however, so fix print statements
from __future__ import print_function

import cv2
import numpy as np
import os
import sys

from sift_wrapper import SiftWrapper
from loaders import open_grayscale_image

MIN_MATCH_COUNT = 5

# Lower - more specifity for matches
THRESHOLD = 0.3

TRAINING_PATH = sys.argv[1]
QUERY_PATH = sys.argv[2]
RESULT_PATH = sys.argv[3]

training_images = os.listdir(TRAINING_PATH)
query_images = os.listdir(QUERY_PATH)

one_train_image = training_images[2]
one_query_image = query_images[13]

sw = SiftWrapper()

training_feature_map = {}
training_image_map = {}
for train_fname in training_images:
    image_path = os.path.join(TRAINING_PATH, train_fname)
    image = open_grayscale_image(image_path)
    training_feature_map[train_fname] = sw.do_sift(image)
    training_image_map[train_fname] = image

for qnum, query_fname in enumerate(query_images):
    print("reading {0}".format(query_fname))
    training_hits = 0
    query_image_path = os.path.join(QUERY_PATH, query_fname)
    # Get 'a' component of lab image
    query_image = open_grayscale_image(query_image_path)
    output_image = np.copy(query_image)

    sift_query = sw.do_sift(query_image)

    all_src_pts = []
    all_dst_pts = []
    for train_fname in training_feature_map.keys():
        sift_train = training_feature_map[train_fname]

        knn_result = sw.do_knn_sift(sift_train, sift_query)

        if knn_result.num_found > MIN_MATCH_COUNT:
            print("matched: {0}".format(train_fname))
            training_hits += 1
            src_pts = np.float32(knn_result.source_points).reshape(-1, 1, 2)
            train_height, train_width = training_image_map[train_fname].shape
            train_height *= 1.0
            train_width *= 1.0
            print("train ratio:", train_height / train_width)
            all_src_pts += [pt / train_width for pt in src_pts]
            all_dst_pts += knn_result.destination_points
            dst_pts = np.float32(knn_result.destination_points).reshape(
                -1, 1, 2
            )

            for dst_pt in dst_pts:
                cv2.circle(
                    output_image, tuple(dst_pt[0]), 5, (255, 255, 255), -1
                )

    print('src:', all_src_pts)
    print('dst:', all_dst_pts)

    # We shaped all the streetsigns in a 2:1 height:width size
    # So this is the basis template for our homography
    corners = np.array([
        [0, 0],
        [0, 2.25],
        [1.0, 2.25],
        [1.0, 0]
    ])
    template_corners = np.float32(corners).reshape(-1, 1, 2)

    still_finding_signs = True
    while still_finding_signs and len(all_dst_pts):
        from_template = np.float32(all_src_pts).reshape(-1, 1, 2)
        to_query_image = np.float32(all_dst_pts).reshape(-1, 1, 2)

        print(from_template, to_query_image)
        M = cv2.estimateRigidTransform(from_template, to_query_image, False)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('homography:\n', M)

        # We have found an affine transform
        if M is not None:
            # Fill out the perspective matrix with a dummy row
            M = np.vstack((M, np.array([0, 0, 1])))
            sign_corners = cv2.perspectiveTransform(template_corners, M)
            cv2.polylines(output_image, [np.int32(sign_corners)], True, 255, 3)

            # Remove points we have already found as a sign and repeat
            new_all_dst_pts = []
            new_all_src_pts = []
            for index, point in enumerate(all_dst_pts):
                # Check point is on or inside region of found sign (0 or +1)
                if cv2.pointPolygonTest(sign_corners, point, True) < 0:
                    new_all_dst_pts.append(point)
                    new_all_src_pts.append(all_src_pts[index])
            all_dst_pts = new_all_dst_pts
            all_src_pts = new_all_src_pts

        else:
            still_finding_signs = False

    if training_hits:
        # Write each query image out with markers from training images
        # NOTE query_fname includes .jpg extension
        output_path = os.path.join(RESULT_PATH, query_fname)
        cv2.imwrite(output_path, output_image)
