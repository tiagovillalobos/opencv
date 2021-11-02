#<!---------------------------------------------------------------------------->
#<!--                  IFSP - Instituto Federal de São Paulo                 -->
#<!--                       Tópicos Avançados (TPA A6)                       -->
#<!-- File       : Ex102_video_stream.py                                     -->
#<!-- Description: Script to convert the video stream in two different       -->
#<!--            : color spaces (RGB and HSV)                                -->
#<!-- Author     : Fabricio Batista Narcizo (narcizo[at]itu[dot]dk)          -->
#<!-- Information: No additional information                                 -->
#<!-- Date       : 10/10/2021                                                -->
#<!-- Change     : 10/10/2021 - Creation of this script                      -->
#<!-- Review     : 10/10/2021 - Finalized                                    -->
#<!---------------------------------------------------------------------------->

__version__ = "$Revision: 2021101001 $"

################################################################################
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

################################################################################
# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default=0, type=int,
                required=False, help="Video capture device ID or file path.")
args = vars(ap.parse_args())

# Hint: You can find more information about opening, converting and showing
#       images using OpenCV on official OpenCV docs (http://docs.opencv.org)

# TODO: Implement your solution here.

camera = cv2.VideoCapture(0)

def frame_single_channel(frame, channel):

    if(channel == 'blue'):
        frame[:, :, 1] = 0
        frame[:, :, 2] = 0

    if(channel == 'green'):
        frame[:, :, 0] = 0
        frame[:, :, 2] = 0

    if(channel == 'red'):
        frame[:, :, 0] = 0
        frame[:, :, 1] = 0

    return frame

while True:
    # Read frame-by-frame.
    ret, frame = camera.read()
    if not ret:
        break

    #(a) Conversão de espaço de cores
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #(b) Separação de canais de cores:
    image_rgb_channels = cv2.split(image_rgb)
    image_rgb_channel_r = image_rgb_channels[0]
    image_rgb_channel_g = image_rgb_channels[1]
    image_rgb_channel_b = image_rgb_channels[2]

    image_hsv_channels = cv2.split(image_hsv)
    image_hsv_channel_r = image_hsv_channels[0]
    image_hsv_channel_g = image_hsv_channels[1]
    image_hsv_channel_b = image_hsv_channels[2]

    #(c) Mude a representação do canal de cor:
    # function frame_single_channel
    
    # Show the current frame.
    cv2.imshow("frame", frame_single_channel(frame, 'green'))
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()