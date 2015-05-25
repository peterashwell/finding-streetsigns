from collections import namedtuple

import cv2


class SiftWrapper:
    # Wrapper for OpenCV SIFT algorithm results
    SiftResult = namedtuple('SiftResult', ['keypoints', 'descriptors'])
    KnnResult = namedtuple(
        'KnnResult',
        ['num_found', 'source_points', 'destination_points']
    )

    # Enum value for flann kdtree is 0
    FLANN_INDEX_KDTREE = 0

    # Distance threshold in knn to be considered a hit
    KNN_DISTANCE_THRESHOLD = 0.5

    def __init__(self):
        self.sift = cv2.SIFT()
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5), dict(checks=50)
        )

    def do_sift(self, img):
        sift_result = self.sift.detectAndCompute(img, None)
        return self.SiftResult(
            keypoints=sift_result[0], descriptors=sift_result[1]
        )

    def do_knn_sift(self, train, query):
        matched_points = []

        # Find matches along knn chain within threshold
        matches = self.flann.knnMatch(
            train.descriptors, query.descriptors, k=2
        )
        for a, b in matches:
            if a.distance < b.distance * self.KNN_DISTANCE_THRESHOLD:
                matched_points.append(a)

        # Extract and reshape keypoint locations
        source = []
        dest = []
        for point in matched_points:
            source.append(train.keypoints[point.queryIdx].pt)
            dest.append(query.keypoints[point.trainIdx].pt)

        return self.KnnResult(
            num_found=len(matched_points),
            source_points=source,
            destination_points=dest
        )
