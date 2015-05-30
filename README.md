- [x] get output from SIFT object descriptors
- [x] fiddle with the knn scores to try and get intelligent results
- [x] try on other images
- [x] improve training set to have types of signs + sources
- [x] run SIFT on training set and see how it goes
- [x] build a new training set of 'has signs' and 'not signs'
- [x] deal with openCV headaches
- [ ] try Lab a channel rather than rgb grayscale for no stopping
- [ ] speed up classification with multiprocessing
- [ ] write code to bound box the parking signs
- [ ] extract sign location and orientation using angular size
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

Build a new training set of 'has signs' and 'not signs' from riley
==================================================================
- find all images that have signs from riley
- construct complement set of images without signs 
- re-run object recognition on that set and get precision, recall 
- summarise error cases

Deal with OpenCV headaches
==========================
- In OpenCV 3 SIFT was moved out of the core repos. Tried to install and failed
- Falling back to OpenCV 2.4, following
  http://stackoverflow.com/questions/18561910/opencv-python-cant-use-surf-sift
  for how to use in python

Try Lab 'a' channel rather than rgb grayscale for no stopping signs
==================================================
- Try running with Lab conversion that had good results in ImageJ
- Convert to Lab, extract a, and inspect results on riley-nostopping
- Modify code to report negatives and get false negative rate

Speed up the classification process
===================================
- Use subprocess to spin off processes, each one lodaing the training set
- See if this has any problems with thrashing the disk. Shouldn't do

Write code to bound box the parking signs
=========================================
- Choose amongst matches which has most hits and produce homography
- Map homography region onto query image
- Extract approximate location using angular size
- Extract facing using skew and aspect of image

Extract sign location and orientation using angular size
========================================================
- Using a the width of the sign, the heading of the image, the fov size can
  find the exact distance to the sign
- Should be able to use the homography to find the facing of the sign

Try to bootstrap training set / get first plotted result
========================================================
- Modify code running on query set to extract bounded results
- Use bounded results to feed into training set with modifications
- Modify code running on query set to extract headings, locations, etc
  of found sign. Plot these onto a map
