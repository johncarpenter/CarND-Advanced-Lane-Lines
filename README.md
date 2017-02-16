## Advanced Lane Finding

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Building off the [first lane finding project](https://github.com/johncarpenter/CarND-LaneLines-P1), this project attempts to build a more robust system for detecting lane markings with a front-facing camera.

### Summary

We were able to detect lane markings for two simple scenarios. The steps to determine the line markings were; first, we preprocess the image using a combination of Sobel filters and different color spaces. This will isolate the line markings within the image. Second, We transform those image into a top-down view and use a sliding histogram window to arrive at a point cloud of potential line matches. Then we then fit a curve to those lines and reproject the lines back into the driving space. Finally, we apply a simple Kalman filter to reduce the impact of errors and smooth out the curves


### Results

[![Video of Performance](http://img.youtube.com/vi/XbCRGD4AG5c/0.jpg)](http://www.youtube.com/watch?v=XbCRGD4AG5c)

The above video shows the detection algorithm in progress (in green). In the top right corner we can see the intermediary images used to locate the lane markings. Road curvature and offsets are derived as a part of the detection algorithm.

### Detailed Steps

#### 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images, and Apply a distortion correction to raw images.

![Calibration Image](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/calibration_test.png)

Using the provided images within the `camera_cal` directory, a correction file was created to adjust images for camera distortion. One image is excluded from the calculation for a verification.

```python
python calibrate.py [image_directory]
```
This generates a `camera.p` pickle file which will be used in the remaining goals.

### Test Playground

The next 4 steps involved


#### 2. Use color transforms, gradients, etc., to create a thresholded binary image.






Apply a perspective transform to rectify binary image ("birds-eye view").
Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.

Output visual

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :).

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!
