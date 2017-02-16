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

Using the provided images within the `camera_cal` directory, a correction file was created to adjust images for camera distortion. The program is run with.

```python
python calibrate.py [image_directory]
```
This generates a `camera.p` pickle file which will be used in the remaining goals.

The calibration is done with the following steps;
1. All the images in the directory and loaded and converted to gray scale. One image is removed from the set to be used for verification testing.
2. Since we are using a regularly spaced grid we can calculate the destination grid coordinates.
3. OpenCV provides a method to calculate the source coordinates with `cv2.findChessboardCorners`
4. Mapping those source to the destination provides a transform matrix and that matrix is stored in the pickle file for retrieval
5. The isolated image is then reprojected using the transform matrix to visually verify the results.(See image above)

### Test Playground

The next 4 steps involved transforming the images into a format that will help to determine the line markings. To this end the program `image_processing_test.py` was created to demonstrate a processing pipeline with still images used to optimize the algorithms. The output of that program is a image (below) file, containing all the intermediary images generated in creating the image processing pipeline.

![Testing Image](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/test1.jpg)

Here are the description of the images included;

| Image Name | Details                                      |
|------------|----------------------------------------------|
| Raw        | Raw Image, with Perspective area highlighted |
| Verify     | Verification image to ensure the video       |
 processing is the same as the test                         |
| Calibrated | Raw Image after camera calibration           |
| S Channel  | Image is converted to HSL and the Saturation channel is extracted.|
| X          | Sobel Filter only X direction |
| Y          | Sobel Filter only Y direction |
| Magnitude  | Magnitude of the Sobel Filter |
| Dir        | Direction of the Sobel Filter |
| Combined   | The merged image from the pipeline (see below) |
| Warp       | The combined image with the perspective warp applied |
| Clipped    | We remove 100 pixels from the left and right side of the image |
| Histogram  | Visualization tools for the histogram search |
| Final      | Post histogram determination and mapped into driving space |


#### 2. Use color transforms, gradients, etc., to create a thresholded binary image.

Combining the images above we looked for a method to isolated the line markings as clearly as possible for later processing. After a number of iterations (hence the playground) we ended up with the following process;

The combined image was those pixels that passed either;

1. Saturation Filter between pixel values of 110 and 255
2. Sobel Direction between 0.1 and 1.1 (pi) and Sobel Magnitude greater than 30

Trial-and-error was used to determine the filter combination and I suspect a more rigorous approach should be used. The types of filters had a very large impact on the results and would vary between different driving conditions.

Full Size Images
[Test 1](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/test1.png)
[Test 2](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/test2.png)
[Test 3](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/test3.png)
[Test 4](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/test4.png)
[Test 5](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/test5.png)
[Test 6](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/test6.png)
[Test 7](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/test7.png)
[Test 8](http://static.2linessoftware.com.s3-website-us-east-1.amazonaws.com/images/test8.png)

#### 3. Apply a perspective transform to rectify binary image ("birds-eye view").
The Warp images in the above examples highlight the transformation between the driving perspective and the birds eye view perspective. Using some knowledge of the location of the front facing camera we can estimate the road in driving perspective and map that unto the top-down view. Examples above


#### 4. Detect lane pixels and fit to find the lane boundary.

This step involved determining which pixels from the Warped image should be included within a curve fit calculation. To do this we approached it in two steps.



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
