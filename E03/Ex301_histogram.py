#<!---------------------------------------------------------------------------->
#<!--                  IFSP - Instituto Federal de São Paulo                 -->
#<!--                       Tópicos Avançados (TPA A6)                       -->
#<!-- File       : Ex301_histogram.py                                        -->
#<!-- Description: Script to represent the pixel distribution function based -->
#<!--            : on each grayscale level of an input image                 -->
#<!-- Author     : Fabricio Batista Narcizo (narcizo[at]itu[dot]dk)          -->
#<!-- Information: No additional information                                 -->
#<!-- Date       : 16/11/2021                                                -->
#<!-- Change     : 16/11/2021 - Creation of this script                      -->
#<!-- Review     : 16/11/2021 - Finalized                                    -->
#<!---------------------------------------------------------------------------->

__version__ = "$Revision: 2021111601 $"

################################################################################
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import path

################################################################################
# Construct the argument parser and parse the arguments.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="Path to the image")
args = vars(ap.parse_args())

# Root directory
root = Path(path.dirname(path.realpath(__file__)))

# Get the input filename
filename = str(root / "./inputs/lena.jpg") if args["image"] is None else args["image"]

# Loads a grayscale image from a file passed as argument.
image = cv2.imread(filename, cv2.IMREAD_COLOR)

# Create the Matplotlib figures.
fig_imgs = plt.figure("Images")
fig_hist = plt.figure("Histograms")

# This function creates a Matplotlib window and shows four images.
def showImage(image, pos, title="Image", isGray=False):
    sub = fig_imgs.add_subplot(2, 2, pos)
    sub.set_title(title)
    if isGray:
        sub.imshow(image, cmap="gray")
    else:
        sub.imshow(image)
    sub.axis("off")

# This function creates a Matplotlib window and shows four histograms.
def showHistogram(histogram, pos, title="Histogram"):
    sub = fig_hist.add_subplot(2, 2, pos)
    sub.set_title(title)
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    plt.xlim([0, 256])
    plt.plot(histogram)

# TODO: You code here

def convertToPLT(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 
grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
showImage(convertToPLT(grayscale), 1, isGray=True, title="Grayscale Image")

grayscaleHistogram = cv2.calcHist([grayscale], [0], None, [256], [0, 256])
showHistogram(grayscaleHistogram, 1, title="Grayscale Histogram")

# 

# 
shuffle = grayscale.copy()
np.random.shuffle(shuffle)
shuffle = cv2.transpose(shuffle)
np.random.shuffle(shuffle)
showImage(convertToPLT(shuffle), 2, title="Shuffled Image")

shuflleHistogram = cv2.calcHist([shuffle], [0], None, [256], [0, 255])
showHistogram(shuflleHistogram, 2, title="Shuffled Histogram")
# 

# 
red = image.copy()
red[:, :, 0] = 0
red[:, :, 1] = 0
redHistogram = cv2.calcHist([red], [0], None, [256], [0, 256])

green = image.copy()
green[:, :, 0] = 0
green[:, :, 2] = 0
greenHistogram = cv2.calcHist([green], [0], None, [256], [0, 256])

blue = image.copy()
blue[:, :, 1] = 0
blue[:, :, 2] = 0
blueHistogram = cv2.calcHist([blue], [0], None, [256], [0, 256])

rgbHistogram = np.zeros([256, 3])
rgbHistogram[:,0] = redHistogram.T
rgbHistogram[:,1] = greenHistogram.T
rgbHistogram[:,2] = blueHistogram.T
showHistogram(rgbHistogram, 3, title="RGB Histogram")
showImage(convertToPLT(image), 3, title="RGB Image")
# 

#
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
showImage(hsv, 4, title="HSV Image")

hChannel, sChannel, vChannel = cv2.split(hsv)
histogramChannelH = cv2.calcHist([hChannel], [0], None, [256], [0, 255])
histogramChannelS = cv2.calcHist([sChannel], [0], None, [256], [0, 255])
histogramChannelV = cv2.calcHist([vChannel], [0], None, [256], [0, 255])

hsvHistogram = np.zeros([256, 3])
hsvHistogram[:,0] = histogramChannelH.T
hsvHistogram[:,1] = histogramChannelS.T
hsvHistogram[:,2] = histogramChannelV.T

showHistogram(hsvHistogram, 4, title="HSV Histogram")
#

# Show the Matplotlib windows.
plt.show()