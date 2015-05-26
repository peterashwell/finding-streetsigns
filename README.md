- [X] get output from SIFT object descriptors
- [X] fiddle with the knn scores to try and get intelligent results
- [X] try on other images
- [x] improve training set to have types of signs + sources
- [x] run SIFT on training set and see how it goes
- [ ] try Lab a channel rather than rgb grayscale for no stopping
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

Try Lab 'a' channel rather than rgb grayscale for no stopping signs
==================================================
- Create training set of no stopping signs from riley set
- The bright, red backgrounded no stopping signs don't get many hits
- Try running with Lab conversion that had good results in ImageJ
- Convert to Lab, extract a, and inspect results on riley-nostopping
- Modify code to report negatives and get false negative rate

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
