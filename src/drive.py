import sys,argparse
import os
import cv2
import image_processor as ImageProcessor
from moviepy.editor import VideoFileClip
import numpy as np

camera = False
left_line = Line()
right_line = Line()

def process_image(rawimage):

    image,left_line,right_line = ImageProcessor.process_image_smoothing(rawimage, camera)

    return image


def process_video(infile,outfile):
    print("Reading {}".format(os.path.basename(infile)))



    clip = VideoFileClip(infile)
    adj_clip = clip.fl_image(process_image)
    adj_clip.write_videofile(outfile, audio=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adv Lane Finding')
    parser.add_argument('-i','--input', help='Input video file',required=True,dest="input")
    parser.add_argument('-o','--output',help='Output video file', required=True,dest="output")
    parser.add_argument('-c', dest="camera",help='Calibration File from calibrate.py')
    args = parser.parse_args()

    camera = args.camera

    process_video(args.input, args.output)
