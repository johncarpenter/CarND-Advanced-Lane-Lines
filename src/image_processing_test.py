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

inter_images = []




def process_image(inputfile, camera):
    print("Processing {}".format(inputfile))

    raw_image = mpimg.imread(inputfile)

    inter_images.append((draw_overlay(raw_image.copy()),'Raw','brg'))

    image = raw_image.copy()

    #verify,l,r = Image.process_image(image, camera)
    #inter_images.append((verify.copy(),'Verify','brg'))

    if(camera):
        print("Importing Camera Calibration from {}".format(camera))
        mtx,dist = Image.read_camera_data(camera)
        image = cv2.undistort(image, mtx, dist, None, mtx)

        inter_images.append((image.copy(),'Calibrated','brg'))
    else:
        print("Skipping Camera Calibration")

    s = Image.hls_select(image,thresh=(0, 255))
    inter_images.append((s,'S Channel (100,220)','gray'))
    #image2 = cv2.GaussianBlur(image2, (5, 5), 0)

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
    combined[((gradx == 1)&(s == 1))|((dir_binary == 1)&(mag_binary == 1))] = 1
    inter_images.append((combined,'Combined','gray'))

    warp = Image.warp_image(combined)
    inter_images.append((warp,'Warp','gray'))


    left,right,out_img = Image.histogram_find_lane(warp)

    out_img=draw_lane_lines(out_img,left)
    out_img=draw_lane_lines(out_img,right)


    inter_images.append((out_img,'Lane Finding','brg'))

    #warp=draw_lane_lines(warp,left,right)
    #inter_images.append((warp,'Lane Finding','brg'))

    overlay = Image.draw_lane_poly(warp,left,right)
    overlay = Image.unwarp_image(overlay)

    final = cv2.addWeighted(raw_image, 1, overlay, 0.3, 0)

    inter_images.append((final,'Final','brg'))

    print("Left {} : Right {}".format(left,right))

    render_results(inter_images)


def draw_lane_lines(img, line_fit,color=[255, 255, 0],thickness=3):

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    line_fitx = line_fit[0]*ploty**2 + line_fit[1]*ploty + line_fit[2]

    for i in range(0,len(line_fitx)-1):
        cv2.line(img, (int(line_fitx[i]), int(ploty[i])), (int(line_fitx[i+1]), int(ploty[i+1])), color, thickness)

    return img;



def render_results(images,output_file=False):

    nrow = math.ceil(len(images) / 2)

    gs = gridspec.GridSpec(nrow,2)

    fig = plt.figure(figsize=(20,10))
    for ndx,pair in enumerate(images):
        ax = fig.add_subplot(gs[ndx])
        ax.set_title("{}".format(pair[1]))
        ax.imshow(pair[0],cmap=pair[2])

    if(output_file):
        plt.savefig("image_test_results.png")
    else:
        plt.show()

def draw_overlay(img, color=[255, 0, 0], thickness=3):
    src,dst = Image.get_transform_parameters(img)

    lines = src.astype(int)

    cv2.line(img, tuple(lines[0]), tuple(lines[1]), color,thickness)
    cv2.line(img, tuple(lines[1]), tuple(lines[2]), color,thickness)
    cv2.line(img, tuple(lines[2]), tuple(lines[3]), color,thickness)
    cv2.line(img, tuple(lines[3]), tuple(lines[0]), color,thickness)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Preprocessing Testing Tool')
    parser.add_argument('input', type=argparse.FileType('r'),help='test images file')
    parser.add_argument('-c', dest="camera",help='Calibration File from calibrate.py')


    args = parser.parse_args()

    process_image(args.input.name, args.camera)
