import cv2
import numpy as np
import os
import sys

from sift_wrapper import SiftWrapper

MIN_MATCH_COUNT = 3
# Lower - more specifity for matches
THRESHOLD = 0.6

training_path = sys.argv[1]
query_path = sys.argv[2]

training_images = os.listdir(training_path)
query_images = os.listdir(query_path)

one_train_image = training_images[2]
one_query_image = query_images[13]

sw = SiftWrapper()

training_map = {}
for train_fname in training_images:
    image = cv2.imread(os.path.join(training_path, train_fname), 0)
    training_map[train_fname] = sw.do_sift(image)

for qnum, query_fname in enumerate(query_images):
    print "reading {0}".format(query_fname)
    training_hits = 0
    query_image = cv2.imread(
        os.path.join(query_path, query_fname), 0
    )
    output_image = np.copy(query_image)

    sift_query = sw.do_sift(query_image)
    for train_fname in training_map.keys():
        sift_train = training_map[train_fname]

        knn_result = sw.do_knn_sift(sift_train, sift_query)

        if knn_result.num_found > MIN_MATCH_COUNT:
            training_hits += 1
            print "matched: {0}".format(train_fname)
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
        output_path = os.path.join('results', 'riley', '{0}'.format(query_fname))
        cv2.imwrite(output_path, output_image)
