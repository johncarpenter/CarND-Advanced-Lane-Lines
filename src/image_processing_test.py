#
# for evaluating the image processing pipeline
#
import argparse,os,glob,sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2, math
import numpy as np

import image_processor as Image

import matplotlib.gridspec as gridspec


def perspective(img):

    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])


    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def process_image(inputfile, camera):
    print("Processing {}".format(inputfile))

    inter_images = []

    raw_image = mpimg.imread(inputfile)
    inter_images.append((raw_image,'Raw','brg'))

    image = raw_image.copy()

    if(camera):
        print("Importing Camera Calibration from {}".format(camera))
        mtx,dist = Image.read_camera_data(camera)
        image = cv2.undistort(image, mtx, dist, None, mtx)

        inter_images.append((image.copy(),'Calibrated','brg'))
    else:
        print("Skipping Camera Calibration")

    image2 = Image.hls_select(image,thresh=(100, 220))
    inter_images.append((image2,'S Channel (100,220)','gray'))

    ksize=15

    # Apply each of the thresholding functions
    gradx = Image.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20,120))
    inter_images.append((gradx,'X','gray'))

    grady = Image.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20,100))
    inter_images.append((grady,'Y','gray'))

    mag_binary = Image.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    inter_images.append((mag_binary,'Mag','gray'))

    dir_binary = Image.dir_threshold(image, sobel_kernel=ksize, thresh=(0.7,1.2))
    inter_images.append((dir_binary,'Dir','gray'))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1)&(image2 == 1))|((dir_binary == 1)&(image2 == 1))] = 1

    inter_images.append((combined,'Combined','gray'))

    warp = perspective(combined)
    inter_images.append((warp,'Warp','gray'))

    render_results(inter_images)

def render_results(images,output_file=False):

    nrow = math.ceil(len(images) / 4)

    gs = gridspec.GridSpec(nrow,4)

    fig = plt.figure(figsize=(20,10))
    for ndx,pair in enumerate(images):
        ax = fig.add_subplot(gs[ndx])
        ax.set_title("{}".format(pair[1]))
        ax.imshow(pair[0],cmap=pair[2])

    if(output_file):
        plt.savefig("image_test_results.png")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Preprocessing Testing Tool')
    parser.add_argument('input', type=argparse.FileType('r'),help='test images file')
    parser.add_argument('-c', dest="camera",help='Calibration File from calibrate.py')


    args = parser.parse_args()

    process_image(args.input.name, args.camera)
