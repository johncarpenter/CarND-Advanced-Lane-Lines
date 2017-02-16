import sys,argparse
import os
import cv2
from moviepy.editor import VideoFileClip
import numpy as np

from Detector import Detector

def process_video(infile,outfile, camera):
    print("Reading {}".format(os.path.basename(infile)))

    ld = Detector(use_smoothing = True, camera = camera)

    clip = VideoFileClip(infile)
    adj_clip = clip.fl_image(ld.process_image)
    adj_clip.write_videofile(outfile, audio=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adv Lane Finding')
    parser.add_argument('-i','--input', help='Input video file',required=True,dest="input")
    parser.add_argument('-o','--output',help='Output video file', required=True,dest="output")
    parser.add_argument('-c', dest="camera",help='Calibration File from calibrate.py')
    args = parser.parse_args()

    process_video(args.input, args.output, args.camera)
