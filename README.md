# Computer Vision

## Computer vision exercise with Python and OpenCV.

This repo contains three differents Jupyter Notebooks, divided on different sections and problems, from applying filters to an image, to the estimation of fundamental matrix ![equation](http://mathurl.com/jnymkmc.png) of the cameras. The code and the images are also available on the repo.

----

### Filtering and subsetting

* __Gaussian filters__: one of the exercises consists on create a gaussian filter to create a mask and convolves the images with this maks.
* __Hybrid images__: the hybrid images are two differente images mixed on a new image that contain the low frecuencies of one of the images, and the high frecuencies of the other image, to create the "ilussion" that if you look the image closer, it looks like the image where we take the high freccuencies, and if we look at smaller scale, the image of the low frecuencies appear.
* __Gaussian Pyramid__: to make it easier to appreciate the effect of the hybrid images, you can create a gaussian pyramid where it appears the same image at different scales.

### Keypoints, descriptors, homographies and panoramas

* __Harris points detection__: these is my own implementation of Harris Points detector, that detect and compute this points at three different scales, and show 100 points (of the 1500 points in total) of every scale on an new image. The green points are on the original scale, the blue ones are belongs to the mid scale and the red ones to the last scale.
* __KAZE/AKAZE detectors__: I use one of this detectors to detect and compute de keypoints of two images, and calculate the matches between two images with a brute force matcher and cross validation, using OpenCV funtions ```AKAZE_create``` or ```KAZE_create```, ```detectAndCompute```, ```BF_Matcher``` and ```match```.
* __Panorama construction__: to create a panorama, I use all of the previous point to find the *homography* between two images with ```findHomography```. With this I can create a linear panorama using a white canvas to insert the images transformed by the homography.

### Camera estimation and epipolar geometry

* __Camera estimation__: camera estimation from points correspondences using the ___DLT algorithm___ and the Frobenius norm to calculate the error.
* __Camera calibration__: camera calibration using chessboard images and the OpenCV functions ```findChessboardCorners```, ```drawChessboardCorners``` to visualize the pattern and ```calibrateCamera``` to calibrate the camera. To correct the lense distortion I use ```getOptimalNewCameraMatrix``` and ```undistort```.
* __Fundamental matrix estimation__: ![equation](http://mathurl.com/jnymkmc.png) estimation using BRISK/ORB detector to get points corresponences and 8-point algorithm with RANSAC. Also we can see the epilines on the images.
* __Essential matrix estimation, translation and rotation between two images__: essential matrix estimation using points correspondences, and the 4 possible solutions to [R|T] matrix problem.
