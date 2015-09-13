import collections
import copy
import cv2
import glob
import itertools
import json
from pygco import cut_from_graph
import numpy as np
import random
from scipy.spatial import Delaunay

POINTS_DIR = 'points/*'
HOMOGRAPHY_SEEDS_AMT = 1000
OUTLIER_DISTANCE = 500


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
        edges.append((vertex, next_vertex, int(distance)))
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
        return OUTLIER_DISTANCE

    # Set z=1 on homogeneous coords after transform
    first_t /= first_t[2]

    # Compute distance between transformed points
    first_d = np.linalg.norm(first_t - second)

    if np.isnan(first_d):
        return OUTLIER_DISTANCE

    return first_d


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
            row.append(int(round(asymmetric_transfer_error(transform, pair))))
        per_model_errors.append(row)
    return np.array(per_model_errors).astype(np.int32)


def apply_pygco(transforms, pts):
    edges = triangulate_query(pts)
    unaries = compile_transfer_errors(transforms, pts)
    # Outlier label
    unaries = np.insert(unaries, 0, OUTLIER_DISTANCE, axis=1)
    pairwise = -1 * np.eye(len(transforms) + 1).astype(np.int32)

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
    best_label = cnt.most_common(1)[0]


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

    apply_pygco(transforms, pts)
    print 'done'

# Open each points file and read in matches
for pts_file in glob.glob(POINTS_DIR):
    pts_lines = open(pts_file).read().strip().split('\n')
    pts = list(map(json.loads, pts_lines))
    handle_points(pts)