import cv2
import numpy as np
import os
import sys

MIN_MATCH_COUNT = 3
# Lower - more specifity for matches
THRESHOLD = 0.5

training_path = sys.argv[1]
query_path = sys.argv[2]

training_images = os.listdir(training_path)
query_images = os.listdir(query_path)

one_train_image = training_images[2]
one_query_image = query_images[13]

img1 = cv2.imread(
    os.path.join(query_path, one_query_image), 0
)
img2 = cv2.imread(
    os.path.join(training_path, one_train_image), 0
)

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < THRESHOLD *n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3)

    for src_pt in src_pts:
        print("src_pt:", src_pt)
        cv2.circle(img1, tuple(src_pt[0]), 5, (255, 255, 255), -1)

    for dst_pt in dst_pts:
        print("dst_pt:", dst_pt)
        cv2.circle(img2, tuple(dst_pt[0]), 5, (255, 255, 255), -1);

else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

#img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
print("writing 1")
cv2.imwrite('queryout.png', img1)
print("writing 2")
cv2.imwrite('trainout.png', img2)

#plt.imshow(img3, 'gray'),plt.show()
