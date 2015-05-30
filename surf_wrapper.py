from collections import namedtuple
import cv2


class SurfWrapper:
    SurfResult = namedtuple('SurfResult', ['keypoints', 'descriptors'])
    KnnResult = namedtuple(
        'KnnResult',
        ['num_found', 'source_points', 'destination_points']
    )

    FLANN_INDEX_KDTREE = 0

    # Number of points to find in matching
    HESSIAN = 400

    # K to use in Knn
    KNN_MATCH_AMOUNT = 2
    KNN_DISTANCE_THRESHOLD = 0.5

    def __init__(self):
        self.surf = cv2.SURF(self.HESSIAN)
        # Set descriptor size to 128-dim
        self.surf.extended = True
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5), dict(checks=50)
        )

    def do_surf(self, img):
        result = self.surf.detectAndCompute(img, None)
        return self.SurfResult(
            keypoints=result[0], descriptors=result[1]
        )

    def do_knn_surf(self, train, query):
        matched_points = []
        if (len(train.keypoints) < self.KNN_MATCH_AMOUNT or
                len(query.keypoints) < self.KNN_MATCH_AMOUNT):
            return self.KnnResult(
                num_found=0, source_points=[], destination_points=[]
            )

        matches = self.flann.knnMatch(
            train.descriptors, query.descriptors, k=self.KNN_MATCH_AMOUNT
        )

        for pair in matches:
            a, b = pair
            if a.distance < b.distance * self.KNN_DISTANCE_THRESHOLD:
                matched_points.append(a)

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
