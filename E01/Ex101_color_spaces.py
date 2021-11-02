#<!---------------------------------------------------------------------------->
#<!--                  IFSP - Instituto Federal de São Paulo                 -->
#<!--                       Tópicos Avançados (TPA A6)                       -->
#<!-- File       : Ex101_color_spaces.py                                     -->
#<!-- Description: Script to convert the input images in two different       -->
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
import matplotlib.image as mpimg

################################################################################
# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="/inputs/lena.jpg", type=str,
                required=False, help="Path to the image.")
args = vars(ap.parse_args())

# Create the Matplotlib window.
fig = plt.figure()

# Hint: You can find more information about opening, converting and showing
#       images using OpenCV on official OpenCV docs (http://docs.opencv.org)

# TODO: Implement your solution here.

# Read the input image.
image = cv2.imread("/home/paula/Documentos/Tiago/TPA6/E01/inputs/lena.jpg")

#(a) Conversão de espaço de cores
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
def create_single_array(image):
    height, width, channels = image.shape
    return np.zeros((width, height), dtype=np.uint8)

 
single_rgb = create_single_array(image_rgb)
single_hsv = create_single_array(image_hsv)

image_rgb_red = cv2.merge([image_rgb_channel_r, single_rgb, single_rgb])
image_rgb_green = cv2.merge([single_rgb, image_rgb_channel_g, single_rgb])
image_rgb_blue = cv2.merge([single_rgb, single_rgb, image_rgb_channel_b])

image_hsv_red = cv2.merge([image_hsv_channel_r, single_hsv, single_hsv])
image_hsv_green = cv2.merge([single_hsv, image_hsv_channel_g, single_hsv])
image_hsv_blue = cv2.merge([single_hsv, single_hsv, image_hsv_channel_b])


# Show the final image.
plt.imshow(image_hsv_red)
plt.show()
