#
# for evaluating the image processing pipeline
#
import argparse,os,glob,sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2, math
import numpy as np

from ImageProcessor import ImageProcessor
from Line import Line

import matplotlib.gridspec as gridspec

class ImageProcessingPlayground:
    def __init__(self,camera = None, output = False):
        self.Image = ImageProcessor(camera, use_smoothing = False)

        self.output = output
        self.inter_images = []
        self.camera = camera

    def process_image(self,inputfile):
        '''
        Playground to evaluate various image processing techniques on test images
        '''
        print("Processing {}".format(inputfile))

        raw_image = mpimg.imread(inputfile)

        self.inter_images.append((self.draw_overlay(raw_image.copy()),'Raw','brg'))

        image = raw_image.copy()

        left_line = Line()
        right_line = Line()

        verify,l,r = self.Image.process_image(image, left_line, right_line)
        self.inter_images.append((verify.copy(),'Verify','brg'))

        if(self.camera):
            print("Importing Camera Calibration from {}".format(self.camera))
            mtx,dist = self.Image.read_camera_data(self.camera)
            image = cv2.undistort(image, mtx, dist, None, mtx)

            self.inter_images.append((image.copy(),'Calibrated','brg'))
        else:
            print("Skipping Camera Calibration")

        s = self.Image.hls_select(image,thresh=(110, 255))
        self.inter_images.append((s,'S Channel (110,255)','gray'))

        ksize=15

        # Apply each of the thresholding functions
        gradx = self.Image.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(25,255))
        self.inter_images.append((gradx,'X','gray'))

        grady = self.Image.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(25,255))
        self.inter_images.append((grady,'Y','gray'))

        mag_binary = self.Image.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 255))
        self.inter_images.append((mag_binary,'Mag','gray'))

        dir_binary = self.Image.dir_threshold(image, sobel_kernel=ksize, thresh=(0.2,1.0))
        self.inter_images.append((dir_binary,'Dir','gray'))

        combined = np.zeros_like(dir_binary)
        combined[((s == 1))|((gradx == 1)&(grady == 1))|((dir_binary == 1)&(mag_binary == 1))] = 1

        self.inter_images.append((combined,'Combined','gray'))

        warp = self.Image.warp_image(combined)
        self.inter_images.append((warp,'Warp','gray'))

        left_line,right_line,out_img = self.Image.histogram_find_lane(warp,left_line,right_line,render_out=True)

        overlay = self.Image.draw_lane_poly(warp,left_line.current_fit,right_line.current_fit)

        # Remap to drive perspective
        overlay = self.Image.unwarp_image(overlay)

        # Add the overlay to the original image
        final = cv2.addWeighted(raw_image, 1, overlay, 0.3, 0)

        # draw road info
        final = self.Image.draw_road_info(final, left_line, right_line)

        # Add processing overlay
        final = self.Image.draw_processing_inlay(final, out_img, left_line, right_line)

        self.inter_images.append((final,'Final','brg'))

        self.render_results()


    def render_results(self, images_per_row = 4):

        images = self.inter_images
        nrow = math.ceil(len(images) / images_per_row)

        gs = gridspec.GridSpec(nrow,images_per_row)

        fig = plt.figure(figsize=(20,10))
        for ndx,pair in enumerate(images):
            ax = fig.add_subplot(gs[ndx])
            ax.set_title("{}".format(pair[1]))
            ax.imshow(pair[0],cmap=pair[2])

        if(self.output):
            plt.savefig(self.output)
        else:
            plt.show()
        self.inter_images = []

    def draw_overlay(self,img, color=[255, 0, 0], thickness=3):
        src,dst = self.Image.get_transform_parameters(img)

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
    parser.add_argument('-o','--output',help='Output test file', dest="output")

    args = parser.parse_args()

    ip = ImageProcessingPlayground(camera = args.camera,output = args.output)
    ip.process_image(args.input.name)
