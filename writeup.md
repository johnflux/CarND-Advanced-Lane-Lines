## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_chessboard.png "Undistorted"
[image2]: ./output_images/undistort_straight_lines1.png "Road Transformed"
[image3]: ./output_images/sobelx.png "Sobelx operator applied"
[image4]: ./output_images/thresholded.png "Threshold on gradient and color"
[imageWarp1]: ./output_images/warper1.png "Warp on original"
[imageWarp2]: ./output_images/warper2.png "Warp on sobel'ed"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in `./calibrate.ipynb`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to one of the checkboard images using the `cv2.undistort()` function and obtained the following result:

On the left is the original calibration image, with the determined chessboard corners overlayed, and on the right is the undistorted version.

![alt text][image1]

I pickled the transformation details in `camera_undistort_matrix.pkl` in the variables `camera_undistort_matrix` and `camera_undistort_dist`.

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

In `lanes.py`, the function `loadUndistortedImageAsYUV()` applies the undistortion matrix found above (by loading in the above pickle file).

This function applies the distortion correction using the `cv2.undistort` function.  Here it is applied to one of the test images: (Left is original, right is distortion corrected)
![alt text][image2]

#### 2. Using color and gradient to create a thresholded binary image.

In `lanes.py`, the function `sobelx(img)` applies the sobel_x operator to the gray scale part of the YUV image.  Note that I have left the two color channels untouched for convenience.

The result of applying this to the same distortion-corrected image above is:

![alt text][image3]

I used a combination of color and gradient thresholds to generate a image (thresholding steps in `applyThresholds(img)` function in `lanes.py`).  I filter for the gradient being greater the 50, and the color being either yellow or white.  Here's the output for the same image above:

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears from line 44 in the file `lanes.py`.  The `warper()` function takes as inputs an image (`img`) and returns a perspective transformed version of the image.  I chose to hardcod the source and destination points as so:

```python
scale = np.divide(np.array([1280, 720]), [img.shape[1],img.shape[0]])
src_points = np.float32([[253,678],[592,450],[687,450],[1054,678]] * scale)
dst_points = np.float32([[320,700],[320,0],[960,0],[960,700]] * scale)
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 253, 678      | 320, 0        |
| 592, 450      | 320, 720      |
| 1054,678      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][imageWarp1]
![alt text][imageWarp2]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
