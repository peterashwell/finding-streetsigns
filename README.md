- [X] get output from SIFT object descriptors
- [X] fiddle with the knn scores to try and get intelligent results
- [X] try on other images
- [ ] improve training set to have types of signs + sources
- [ ] write code to bound box the parking signs
- [ ] run SIFT on training set and see how it goes
- [ ] try to bootstrap training set / get first plotted result

Improve training set to have types of signs + sources
=====================================================
- Collect 5 samples of each sign type I know of, keep track of sources
- 'Each sign type' includes paired signs as distinct types for now

Write code to bound box the parking signs
=========================================
- Use a homography of points and the border of the training image
- If multiple training images match, draw all homographies

Run SIFT on training set and see how it goes
============================================
- Train on entire set of sample signs
- Start applying to riley st set with bounding boxes
- Review results

Try to bootstrap training set / get first plotted result
========================================================
- Modify code running on query set to extract bounded results
- Use bounded results to feed into training set with modifications
- Modify code running on query set to extract headings, locations, etc
  of found sign. Plot these onto a map
