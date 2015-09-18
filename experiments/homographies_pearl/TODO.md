Part One: Does it work at all
-----------------------------
 - [x] Create training set of points used in matches and corresponding images
 - [x] Compute initial homographies from points using SVD
 - [x] Compute STE for all points to all homographies
 - [x] Do not use spatial smoothing and use levenberg-merquadt for one iteration using gco
 - [x] If it works, use spatial smoothing and see if there is improvement

 Part Two: Why are there weird errors, iterations, and improvement
 - [ ] Try using affine transforms only from seeds
 - [ ] Discard outliers from models once found
 - [ ] Instead of just reprojection error, use homography healthyness
 - [ ] Experiment with the distance features to be stronger
 - [ ] Introduce iterations and examine lower homographies to see if good ones could emerge later
 - [o] Try and figure out why it crashes randomly
