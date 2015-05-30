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

    # K used in KNN matching
    KNN_MATCH_AMOUNT = 2

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

        # NOTE This fixes issue http://bit.ly/1AzNOvF
        # Lousy coding in OpenCV segfaults when you try to match
        # K nearest neighbours against a smaller number of points
        if (len(train.keypoints) < self.KNN_MATCH_AMOUNT or
                len(query.keypoints) < self.KNN_MATCH_AMOUNT):
            return self.KnnResult(
                num_found=0, source_points=[], destination_points=[]
            )

        # Find matches along knn chain within threshold
        matches = self.flann.knnMatch(
            train.descriptors, query.descriptors, k=self.KNN_MATCH_AMOUNT
        )
        for pair in matches:
            a, b = pair
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
