- [X] get output from SIFT object descriptors
- [X] fiddle with the knn scores to try and get intelligent results
- [X] try on other images
- [x] improve training set to have types of signs + sources
- [x] run SIFT on training set and see how it goes
- [ ] figure out what's wrong with the no stopping signs
- [ ] write code to bound box the parking signs
- [ ] speed up classification
- [ ] try to bootstrap training set / get first plotted result

Improve training set to have types of signs + sources
=====================================================
- Collect 5 samples of each sign type I know of, keep track of sources
- 'Each sign type' includes paired signs as distinct types for now


Run SIFT on training set and see how it goes
============================================
- Train on entire set of sample signs
- Start applying to riley st set with bounding boxes
- Review results

Figure out what's wrong with the no stopping signs
==================================================
- The bright, red backgrounded no stopping signs don't get many hits
- Perhaps running in HSB will get better results, in saturation space

Write code to bound box the parking signs
=========================================
- Choose amongst matches which has most hits and produce homography
- Map homography region onto query image
- Extract approximate location using angular size
- Extract facing using skew and aspect of image

Try to bootstrap training set / get first plotted result
========================================================
- Modify code running on query set to extract bounded results
- Use bounded results to feed into training set with modifications
- Modify code running on query set to extract headings, locations, etc
  of found sign. Plot these onto a map
