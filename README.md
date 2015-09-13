- [x] get output from SIFT object descriptors
- [x] fiddle with the knn scores to try and get intelligent results
- [x] try on other images
- [x] improve training set to have types of signs + sources
- [x] run SIFT on training set and see how it goes
- [x] build a new training set of 'has signs' and 'not signs'
- [x] deal with openCV headaches
- [x] try Lab a channel rather than rgb grayscale for no stopping
- [x] speed up classification with multiprocessing
- [x] Try SURF features to improve recall
- [x] Try edge features to improve recall
- [ ] Tier results set to be less optimistic e.g. remove distant / unreadable
- [x] Write code to segment the image preferring the parking signs
- [x] write code to bound box the parking signs
- [x] try template matching to find sign boundaries
- [x] improve homographies by using estimateRigidTransform
- [x] do complete finding of homography, remove region, and repeat
- [x] try to get more matches for no stopping and no parking by including error cases in training set
- [ ] filter out no-homography matches on riley and do an evaluation
- [ ] splitting up the set of SIFT matches into multiple signs
- [ ] combine matches across alpha channel segmentation to get better results for no stopping
- [ ] make training set for new style small signs, loading zones, bus zones
- [ ] stretch top and bottom of match to include more points if it does
- [ ] back up affine transform with homography
- [ ] create a new training set using blobs on images rather than whole files
- [ ] create a kippax dataset
- [ ] extract sign location and orientation using angular size and transform
- [ ] try to bootstrap training set / get first plotted result

Improve homographies by using estimateRigidTransform
----------------------------------------------------
 - This restricts to affine transformations only
 - It works! sort of... The wild homographies are cut out of computation
 - Will use instead of findHomography in the future

Improve training set to have types of signs + sources
-----------------------------------------------------
- Collect 5 samples of each sign type I know of, keep track of sources
- 'Each sign type' includes paired signs as distinct types for now

Run SIFT on training set and see how it goes
--------------------------------------------
- Train on entire set of sample signs
- Start applying to riley st set with bounding boxes
- Review results

Build a new training set of 'has signs' and 'not signs' from riley
------------------------------------------------------------------
- find all images that have signs from riley
- construct complement set of images without signs 
- re-run object recognition on that set and get precision, recall 
- summarise error cases

Deal with OpenCV headaches
--------------------------
- In OpenCV 3 SIFT was moved out of the core repos. Tried to install and failed
- Falling back to OpenCV 2.4, following
  http://stackoverflow.com/questions/18561910/opencv-python-cant-use-surf-sift
  for how to use in python

Try Lab 'a' channel rather than rgb grayscale for no stopping signs
--------------------------------------------------
- Try running with Lab conversion that had good results in ImageJ
- Convert to Lab, extract a, and inspect results on riley-nostopping
- Modify code to report negatives and get false negative rate
- RESULT: Not that great. Sticking with grayscale for now 
- HSV saturation space may have improved no stopping results

Speed up the classification process
-----------------------------------
- Use subprocess to spin off processes, each one lodaing the training set
- See if this has any problems with thrashing the disk. Shouldn't do
- RESULT: Not bothering with this for now. It's fast enough

Try SURF features to improve recall
-----------------------------------
- Use SURF extractor instead of SIFT
- Evaluate results with new script
- RESULT: While it wasn't bad, wasn't impressive either. More errors to get same
  recall value.
- Need to compare with SIFT results to see if SURF is a good complement

Tier results set to be less result e.g. remove distant / unreadable
-------------------------------------------------------------------
- Right now the riley set is pretty fucking hard
- Remove or tier the difficult images so that more realistic and straightforward
  examinations of error cases are possible

Try edge features to improve recall
-----------------------------------
- Do sobel edge extraction then apply SIFT 
- Perhaps by writing edge\_loader
- RESULT: This was shit. I don't think SIFT features like the extremeness
  of edges. Probably because they are based on gaussians
- Try again with canny edge detection as a feature in the future, combined with
  template matching

Write code to segment the image preferring parking signs
--------------------------------------------------------
- current best idea I have to get sign boundaries
- signs should be easy to segment out - vote on the segments using the keypoints
  to get sets of pixels covering (each) sign
- Approach is first to use Canny -> Watershed -> basic voting scheme

Write code to bound box the parking signs
-----------------------------------------
- Choose amongst matches which has most hits and produce homography
- Map homography region onto query image
- Extract approximate location using angular size
- Extract facing using skew and aspect of image
- RESULT: Poor. Manual extraction of sign boundaries may be something I need to
  consider. Could also consider attempting template matching from the start
- Try using Canny edge detection + segmentation. Assign to each segment the
  matching sign from keypoint, or using a voting measure

Extract sign location and orientation using angular size
--------------------------------------------------------
- Using a the width of the sign, the heading of the image, the fov size can
  find the exact distance to the sign
- Should be able to use the homography to find the facing of the sign

Try to bootstrap training set / get first plotted result
--------------------------------------------------------
- Modify code running on query set to extract bounded results
- Use bounded results to feed into training set with modifications
- Modify code running on query set to extract headings, locations, etc
  of found sign. Plot these onto a map


Improve homographies by using estimateRigidTransform
----------------------------------------------------
 - Use estimateRigidTransform to restrict mapping to affine only
 - Works well, some crazy results have gone
 - Will use instead of findHomography from now on

Improve homographies by filtering SIFT matches
----------------------------------------------
 - Try to filter by using singular sign types
 - Try to filter by simple clustering methods
 - Try to filter by iterative improvement
 - Try to filter by using segmentation
