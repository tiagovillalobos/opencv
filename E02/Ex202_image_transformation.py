#<!---------------------------------------------------------------------------->
#<!--                  IFSP - Instituto Federal de São Paulo                 -->
#<!--                       Tópicos Avançados (TPA A6)                       -->
#<!-- File       : Ex202_image_transformation.py                             -->
#<!-- Description: Script to apply geometric transformations (rotation,      -->
#<!--            : scale and translation) in image sequences                 -->
#<!-- Author     : Fabricio Batista Narcizo (narcizo[at]itu[dot]dk)          -->
#<!-- Information: No additional information                                 -->
#<!-- Date       : 27/10/2021                                                -->
#<!-- Change     : 27/10/2021 - Creation of this script                      -->
#<!-- Review     : 27/10/2021 - Finalized                                    -->
#<!---------------------------------------------------------------------------->

__version__ = "$Revision: 2021102701 $"

################################################################################
import cv2 as cv
import math
import numpy as np

########################################################################
def affineTransformation(image, theta=0, t=(0, 0), s=1.):
    
    height, width = image.shape[:2]
    center = (width/2, height/2)

    #Translation
    T = np.float32([
        [1, 0, t[0]], 
        [0, 1, t[1]],
        [0, 0, 1]
    ])
    if s > 1: s = s / 10
    #Rotation
    R = cv.getRotationMatrix2D(center, theta, s)

    #Scale
    
    S = np.float32([
        [s, 0, 0],
        [0, s, 0]
    ])

    print(s)

    TM = np.matmul(R, T, S)

    return cv.warpAffine(src=image, M=TM, dsize=(width, height))


def nothing(x):
    pass

cv.namedWindow('camera')
cv.namedWindow('transformations')

cv.createTrackbar('rotation', 'transformations', 0, 360, nothing)
cv.createTrackbar('scale', 'transformations', 1, 10, nothing)
cv.createTrackbar('translation(x)', 'transformations', 0, 512, nothing)
cv.createTrackbar('translation(y)', 'transformations', 0, 512, nothing)

cv.setTrackbarMin('scale', 'transformations', 1)
cv.setTrackbarMin('translation(x)', 'transformations', -512)
cv.setTrackbarMin('translation(y)', 'transformations', -512)

camera = cv.VideoCapture(0)

while True:

    ret, frame = camera.read()
    
    if not ret: break

    theta = int(cv.getTrackbarPos('rotation','transformations'))
    s = float(cv.getTrackbarPos('scale','transformations'))
    tx = int(cv.getTrackbarPos('translation(x)','transformations'))
    ty = int(cv.getTrackbarPos('translation(y)','transformations'))

    result = affineTransformation(frame, theta, (tx, ty), s)

    images = np.hstack((frame, result))



    cv.imshow('transformations', images)

    key = cv.waitKey(1) & 0xFF

    if key == 27: break

camera.release()
cv.destroyAllWindows()