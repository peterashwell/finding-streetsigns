import collections
import copy
import cv2
import glob
import itertools
import json
from lmfit import minimize, Parameter
import numpy as np
import os
from pygco import cut_from_graph
import random
from scipy.spatial import Delaunay

POINTS_DIR = 'points/*'
HOMOGRAPHY_SEEDS_AMT = 3000
OUTLIER_DISTANCE = 50
LARGE_DISTANCE = 10000
DELAUNAY_WEIGHTING = 1
REPROJECTION_WEIGHTING = 1
PAIRWISE_WEIGHTING = -1


def random_combination(iterable, how_many, in_each):
    combinations = []
    while len(combinations) < how_many:
        this_combination = []
        while len(this_combination) < in_each:
            choice = random.choice(iterable)
            if choice not in this_combination:
                this_combination.append(choice)
        already = False
        for existing_combination in combinations:
            if all([item in this_combination for item in existing_combination]):
                already = True
        if not already:
            combinations.append(this_combination)
    return combinations


def split_src_dst(pairs):
    src, dst = [], []
    for pair in pairs:
        src.append(pair[0])
        dst.append(pair[1])
    return src, dst


# Triangulate a point set and return an edge graph of adjacent points
def triangulate_query(all_pairs):
    src, dst = split_src_dst(all_pairs)
    src = np.array(src)
    dst = np.array(dst)
    tri = Delaunay(dst)

    edges = []

    for index in xrange(tri.vertex_neighbor_vertices[0].shape[0] - 1):
        next_index = index + 1

        vertex = tri.vertex_neighbor_vertices[1][index]
        next_vertex = tri.vertex_neighbor_vertices[1][next_index]

        point = dst[vertex]
        next_point = dst[next_vertex]

        distance = np.linalg.norm(point - next_point)
        edges.append((vertex, next_vertex, DELAUNAY_WEIGHTING * int(distance)))
        #edges.append((vertex, next_vertex))

    return np.array(edges).astype(np.int32)


# Lol - actually a-symmetric transfer error - forwards only
# Can't be bothered scaling the error in both directions
# Sign sizes are different so to make both distances meaningful need that
def asymmetric_transfer_error(homography, pair):
    first, second = pair

    # Make coords homogeneous
    first = np.array(first + [1])
    second = np.array(second + [1])

    first_t = np.dot(homography, first)

    if first_t[2] < np.finfo(np.float32).eps:
        return LARGE_DISTANCE

    # Set z=1 on homogeneous coords after transform
    first_t /= first_t[2]

    # Compute distance between transformed points
    first_d = np.linalg.norm(first_t - second)

    if np.isnan(first_d):
        return LARGE_DISTANCE

    return first_d



def reestimate_transform(transform, points):
    # Define objective as function of all points above and current transform (params)
    def objective(params):
        # Build array of homography parameters and reshape
        params_dict = params.valuesdict()
        unpacked = np.array([
            params_dict['val' + str(i)]
            for i in xrange(0, 9)
        ])
        unoptimized_transform = unpacked.reshape(3, 3)
        # Compute reprojection error across all points
        reprojection_distances = [
            asymmetric_transfer_error(unoptimized_transform, p) for p in points
        ]
        #return reprojection_distances
        return sum(reprojection_distances)

    # Create parameters from initial transform by flattening and building dict
    parameter_list = transform.flatten().tolist()

    parameters = [
        Parameter('val' + str(i), value=val)
        for i, val in enumerate(parameter_list)
    ]

    # Run and return optimized parameters for reprojection error
    minimize(objective, parameters, method='nelder')

    unpacked = np.array([ p.value for p in parameters ])
    return unpacked.reshape(3, 3)


def find_homography(four_pairs):
    src, dst = split_src_dst(four_pairs)
    src = np.array(src)
    dst = np.array(dst)
    transform = cv2.findHomography(src, dst, False)

    try:
        np.linalg.inv(transform[0])
    except:
        return None
    return transform[0]


def compile_transfer_errors(transforms, pts):
    per_model_errors = []
    for pair in pts:
        row = []
        for transform in transforms:
            row.append(REPROJECTION_WEIGHTING * int(round(asymmetric_transfer_error(transform, pair))))
        per_model_errors.append(row)
    return np.array(per_model_errors).astype(np.int32)


def apply_pygco(transforms, pts):
    edges = triangulate_query(pts)
    unaries = compile_transfer_errors(transforms, pts)
    # Outlier label
    unaries = np.insert(unaries, 0, OUTLIER_DISTANCE, axis=1)
    pairwise = PAIRWISE_WEIGHTING * np.eye(len(transforms) + 1).astype(np.int32)

    print 'edges', edges.shape
    print 'unaries', unaries.shape
    print 'transforms', pairwise.shape

    print 'edges', edges
    print 'unaries', unaries
    print 'transforms', pairwise

    result = cut_from_graph(edges, unaries, pairwise)

    print 'result', result

    cnt = collections.Counter()
    for label in result:
        cnt[label] += 1
    best_label = cnt.most_common(2)[0][0]
    second_best_label = cnt.most_common(2)[1][0]

    print 'best label', best_label, second_best_label
    best_points = []
    second_best_points = []
    for index, label in enumerate(result):
        if label == best_label:
            best_points.append(pts[index])

        elif label == second_best_label:
            second_best_points.append(pts[index])

    return (transforms[best_label - 1], transforms[second_best_label - 1], best_points, second_best_points)


# Read file line by line, converting each to json
# Lines look like "[[x1, y1], [x2, y2]]"
def handle_points(pts):
    random.shuffle(pts)
    print ('num points:', len(pts))
    sets_of_four_pairs = random_combination(pts, HOMOGRAPHY_SEEDS_AMT, 4)
    transforms = []
    for four_pairs in sets_of_four_pairs:
        # unpack the combination
        transform = find_homography(four_pairs)
        if transform is not None:
            transforms.append(transform)

    print 'done'
    return apply_pygco(transforms, pts)


def get_sign_transform(transform):
    normal_corners = np.array([
        [0, 0],
        [0, 2.25],
        [1.0, 2.25],
        [1.0, 0]
    ])

    normal_corners = np.float32(normal_corners).reshape(-1, 1, 2)
    normal_corners = cv2.perspectiveTransform(normal_corners, transform)
    return np.int32([normal_corners])

# Open each points file and read in matches
for pts_file in glob.glob(POINTS_DIR):
    pts_lines = open(pts_file).read().strip().split('\n')
    pts = list(map(json.loads, pts_lines))
    best_transform, second_best_transform, best_points, second_best_points = handle_points(pts)

    basename = os.path.splitext(os.path.basename(pts_file))[0]

    imagepath = 'images/' + basename + '.jpg'
    image = cv2.imread(imagepath)

    print 'best transform', get_sign_transform(best_transform)
    print 'second best transform', get_sign_transform(second_best_transform)

    cv2.polylines(image, get_sign_transform(best_transform), True, 255, 3)
    cv2.polylines(image, get_sign_transform(second_best_transform), True, 255, 3)

    improved_best = reestimate_transform(np.copy(best_transform), best_points)
    improved_second_best = reestimate_transform(np.copy(second_best_transform), second_best_points)
    cv2.polylines(image, get_sign_transform(improved_best), True, (0, 255, 255), 1)
    cv2.polylines(image, get_sign_transform(improved_second_best), True, (0, 255, 255), 1)

    for pairs in best_points:
        dst = pairs[1]
        print 'dst', dst
        cv2.circle(image, tuple([int(o) for o in dst]), 5, (0, 255, 0), -1)

    for pairs in second_best_points:
        dst = pairs[1]
        print 'dst', dst
        cv2.circle(image, tuple([int(o) for o in dst]), 5, (0, 0, 255), -1)

    cv2.imwrite('results/' + basename + '.jpg', image)
