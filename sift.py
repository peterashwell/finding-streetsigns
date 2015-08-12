# We are using python2 because OpenCV compatibility for python3 sucks
# Continue using a python3 linter however, so fix print statements
from __future__ import print_function

import cv2
import numpy as np
import os
import sys
import copy

from sift_wrapper import SiftWrapper
from loaders import open_grayscale_image

MIN_MATCH_COUNT = 5

# Lower - more specifity for matches
TRAINING_PATH = sys.argv[1]
QUERY_PATH = sys.argv[2]
RESULT_PATH = sys.argv[3]

training_images = os.listdir(TRAINING_PATH)
query_images = os.listdir(QUERY_PATH)

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
            dst_pts = np.float32(knn_result.destination_points).reshape(-1, 1, 2)

    #print('src:', all_src_pts)
    #print('dst:', all_dst_pts)

    # We shaped all the streetsigns in a 2:1 height:width size
    # So this is the basis template for our homography
    # Also, stretch it out a little to potentially include more matches
    #
    # (-0.1,-0.5) (1.1,-0.5)
    #     +-----------+
    #     |           |
    #     |           |
    #     |           |
    #     |           |
    #     |           |
    #     |           |
    #     |           |
    #     |           |
    #     |           |
    #     +-----------+
    # (-0.1,2.75) (1.1,2.75)

    attempts = 3
    train_matches = np.copy(all_src_pts)
    query_matches = np.copy(all_dst_pts)

    while attempts and len(query_matches):
        if len(train_matches) < 4 and len(query_matches) < 4:
            break
        from_template = np.float32(train_matches).reshape(-1, 1, 2)
        to_query_image = np.float32(query_matches).reshape(-1, 1, 2)

        # Affine transform
        M = cv2.estimateRigidTransform(from_template, to_query_image, False)
        if M is None:
            M = cv2.findHomography(from_template, to_query_image, cv2.RANSAC, 5.0)[0]
        else:
            M = np.vstack((M, np.array([0, 0, 1])))
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('homography:\n', M)

        normal_corners = np.array([
            [0, 0],
            [0, 2.25],
            [1.0, 2.25],
            [1.0, 0]
        ])
        stretched_corners = np.array([
            [-0.05, -0.5],
            [-0.05, 2.75],
            [1.05, 2.75],
            [1.05, -0.5]
        ])
        # Turn into vectors for matrix multiplication
        normal_corners = np.float32(normal_corners).reshape(-1, 1, 2)
        stretched_corners = np.float32(stretched_corners).reshape(-1, 1, 2)

        # We have found an affine transform
        if M is not None:
            attempts -= 1
            # Fill out the perspective matrix with a dummy row
            normal_corners = cv2.perspectiveTransform(normal_corners, M)
            stretched_corners = cv2.perspectiveTransform(stretched_corners, M)
            cv2.polylines(output_image, [np.int32(normal_corners)], True, 255, 3)

            # Remove points we have already found as a sign and repeat
            new_query_matches = []
            new_train_matches = []
            for index, point in enumerate(query_matches):
                # Check point is on or inside region of found sign (0 or +1)
                tuplepoint = (point[0], point[1])
                if cv2.pointPolygonTest(stretched_corners, tuplepoint, True) < 0:
                    new_query_matches.append(point)
                    new_train_matches.append(train_matches[index])
            query_matches = new_query_matches
            train_matches = new_train_matches
        else:
            attempts = False

    if training_hits:
        # Add points not matched to a homography
        for dst_pt in query_matches:
            int_pt = tuple([int(x) for x in dst_pt])
            cv2.circle(output_image, int_pt, 5, (255, 255, 255), -1)

        # Plot match lines
        # Pad output image to place where sign would be if it were there
        #fill = np.zeros(output_image.shape)
        #output_image = np.concatenate((output_image, fill), axis=1)

        # Scale and shift matches so that they fit in the box
        scale_factor = output_image.shape[1] / 2.25
        shift_factor = output_image.shape[0]
        for train, query in zip(all_src_pts, all_dst_pts):
            train = np.int32(train[0] * scale_factor)
            train[0] += shift_factor
            train = (train[0], train[1])
            query = (int(query[0]), int(query[1]))
            #cv2.line( output_image, train, query, [255, 0, 0], 5)


        # Write each query image out with markers from training images
        # NOTE query_fname includes .jpg extension
        output_path = os.path.join(RESULT_PATH, query_fname)
        cv2.imwrite(output_path, output_image)
