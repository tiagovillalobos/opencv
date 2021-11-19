import cv2 as cv
import numpy as np

base_width = 384
base_height = 192

def null(x):
    pass

def rectangle_area(width, height):
    return width * height

def circle_area(radius):
    return 3.14 * radius * radius

cv.namedWindow('image')

cv.createTrackbar('width', 'image', base_width, 512, null)
cv.createTrackbar('height', 'image', base_height, 512, null)

cv.createTrackbar('radius','image', 15, 255, null)

while(1):

    #Criando uma imagem com fundo preto
    image = np.zeros((512, 512, 3), np.uint8)
    
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    width = int(cv.getTrackbarPos('width','image'))
    height = int(cv.getTrackbarPos('height','image'))
    radius = int(cv.getTrackbarPos('radius','image'))

    real_width = base_width + 64
    real_height = base_height + 160

    new_width = (width - base_width) // 2
    new_heigth = (height - base_height) // 2
    
    #Desenhando um retangulo:
    cv.rectangle(image, (64 - new_width, 160 - new_heigth), ((real_width + new_width), (real_height + new_heigth)), (0, 255, 0), -1)

    #Desenhando um circulo
    cv.circle(image,((image.shape[0]//2), (image.shape[1]//2)), radius, (0,0,255), -1)

    #Desenhando linhas
    cv.line(image, (64 - new_width, 160 - new_heigth), ((real_width + new_width), (real_height + new_heigth)), (255, 0, 0))

    #texto
    text = 'Area do Retangulo: ' + str(rectangle_area(width, height))
    cv.putText(image, text, (0, 20), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0))

    text = 'Area do Circulo: ' + str(circle_area(radius))
    cv.putText(image, text, (0, 40), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0))

    cv.imshow('image', image)
    
    key = cv.waitKey(1) & 0xFF

    if key == 27: 
        break

cv.destroyAllWindows() 