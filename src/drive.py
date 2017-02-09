import sys,argparse
import os
import cv2
from moviepy.editor import VideoFileClip


import utils


def process_image(rawimage):

    # Calibrate the image
    mtx, dist = utils.read_camera_data()
    image = cv2.undistort(rawimage, mtx, dist, None, mtx)

    # Apply corrections
    process_pipeline(image)

    # Perspective Transform Image



    return rawimage

def process_video(infile,outfile):
    print("Reading {}".format(os.path.basename(infile)))
    clip = VideoFileClip(infile)
    adj_clip = clip.fl_image(process_image)
    adj_clip.write_videofile(outfile, audio=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adv Lane Finding')
    parser.add_argument('-i','--input', help='Input video file',required=True,dest="input")
    parser.add_argument('-o','--output',help='Output video file', required=True,dest="output")
    args = parser.parse_args()

    process_video(args.input, args.output)
