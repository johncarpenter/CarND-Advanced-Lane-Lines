# Calibrates Camera and Stores calibration constants
#
import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import random


def calibrate(directory,nx=9,ny=6):
    print("Reading {}".format(directory))

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(directory+'/*.jpg')
    test_image_name = images.pop(random.randint(0,len(images)-1))
    print("Removing {} for validation".format(test_image_name))

    test_img = cv2.imread(test_image_name)
    img_size = (test_img.shape[1], test_img.shape[0])

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        print("Processing {}".format(fname))
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("No corners found for {} ".format(fname))


    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    output_results(mtx,dist)

    visual_test(test_img,mtx,dist,savefig=True)

def visual_test(img, mtx, dist,savefig=False):
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Calibrated Image', fontsize=30)
    if(savefig):
        plt.savefig('calibration_test.png')
    else:
        plt.show()

def output_results(mtx,dist,outfile='camera_pickle.p'):
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( outfile, "wb" ) )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera Calibration')
    parser.add_argument('input', help='Calibration directory',required=True,dest="input")
# TODO
    #parser.add_argument('-s', help='Chessboard size x,y ',required=False,dest="n")
    #parser.add_argument('-t', help='Test Calibration Image',required=False,dest="test")



    args = parser.parse_args()

    calibrate(args.dir)
