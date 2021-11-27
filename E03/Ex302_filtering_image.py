#<!---------------------------------------------------------------------------->
#<!--                  IFSP - Instituto Federal de São Paulo                 -->
#<!--                       Tópicos Avançados (TPA A6)                       -->
#<!-- File       : Ex302_filtering_image.py                                  -->
#<!-- Description: Script to generate and remove noise using filters         -->
#<!-- Author     : Fabricio Batista Narcizo (narcizo[at]itu[dot]dk)          -->
#<!-- Information: No additional information                                 -->
#<!-- Date       : 16/11/2021                                                -->
#<!-- Change     : 16/11/2021 - Creation of this script                      -->
#<!-- Review     : 16/11/2021 - Finalized                                    -->
#<!---------------------------------------------------------------------------->

__version__ = "$Revision: 2021111601 $"

################################################################################
import cv2
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from functools import partial
from matplotlib.widgets import Slider

from pathlib import Path
from os import path

################################################################################

def showImages(**images):
    """
    Show multiple images using matplotlib.
    """
    # When a double-starred parameter is declared such as $**images$, then all
    # the keyword arguments from that point till the end are collected as a
    # dictionary called $'images'$.

    # Create a new matplotlib window.
    plt.figure()

    # Set the default colormap to gray and apply to current image if any.
    plt.gray()

    # Enumarate the ID, window name and images passed as parameter.
    for (pos, (name, image)) in enumerate(images.items()):
        # Show the image in a new subplot.
        
        plt.subplot(2, len(images), pos+1)
        plt.title(name)
        plt.imshow(image)

    # Show the images.
    plt.show()

def showSignals(row, **images):
    """
    Show multiple signals using matplotlib.
    """
    # When a double-starred parameter is declared such as $**images$, then all
    # the keyword arguments from that point till the end are collected as a
    # dictionary called $'images'$.

    # Create a list of colors.
    colors = [ "C0", "C1", "C2", "C3" ]

    # Enumarate the ID, window name and images passed as parameter.
    for (pos, (name, image)) in enumerate(images.items()):
        # Get the image width and the current color.
        w = image.shape[1]
        color = colors[pos % round(len(images) / 2)]

        # This vector contains a list of all valid grayscale level.
        bins = np.array(range(w))

        # Plot the signal in a new subplot.
        ax[pos].cla()
        ax[pos].set_title(name)
        ax[pos].grid(None, "major", "both")
        ax[pos].axis([0, w, 0, 255])
        ax[pos].plot(bins, image[row, :], color=color, linewidth=1)

    # Show the input image.
    color = cv2.cvtColor(images["Original"], cv2.COLOR_GRAY2RGB)
    cv2.line(color, (0, row), (color.shape[1], row), (255, 255, 255), 2)
    ax[pos+1].imshow(color)

def updateRow(slider, val):
    """
    This function will be performed when the user changes the row slider.
    """
    # Create a mask for the slider to set only integer number.
    slider.val = int(round(val))
    slider.poly.xy[2] = slider.val, 1
    slider.poly.xy[3] = slider.val, 0
    slider.valtext.set_text(slider.valfmt % slider.val)

    # Draw the filtered signal.
    showSignals(slider.val, Uniform=uniform, Gaussian=gaussian, Salt_and_Pepper=saltAndPepper,
                Original=image, Uniform_Filtered=uniformFiltered,
                Gaussian_Filtered=gaussianFiltered,
                Salt_and_Pepper_Filtered=saltAndPepperFiltered)

def to_uint8(data):
    """
    This function convert the vector data to unsigned integer 8-bits.
    """
    # Maximum pixel.
    latch = np.zeros_like(data)
    latch[:] = 255

    # Minimum pixel.
    zeros = np.zeros_like(data)

    # Unrolled to illustrate steps.
    d = np.maximum(zeros, data)
    d = np.minimum(latch, d)

    # Cast to uint8.
    return np.asarray(d, np.uint8)


#<!--------------------------------------------------------------------------->
#<!--                  GENERATE RANDOM NOISE DISTRIBUTION                   -->
#<!--------------------------------------------------------------------------->

def saltAndPepperNoise(image, density):
    """
    This function generates and adds salt-and-pepper noise to the input image.
    """
    # You have to add noise in this variable.
    noised = image.copy()
    h, w = noised.shape

    # TODO: Your code here
    noised[(np.random.rand(h, w) < density)] = [0]
    noised[(np.random.rand(h, w) < density)] = [255]

    return noised

def gaussianNoise(image, mu, sigma):
    """
    This function generates and adds Gaussian noise to the input image.
    """
    # You have to add noise in this variable.
    noised = image.copy()
    h, w = noised.shape

    # TODO: Your code here
    norm = to_uint8(np.random.normal(mu, sigma, (h,w)))
    noised = cv2.add(noised, norm)

    return noised

def uniformNoise(image, low, high):
    """
    This function generates and adds uniform noise to the input image.
    """
    # You have to add noise in this variable.
    noised = image.copy()
    h, w = noised.shape

    # TODO: Your code here
    norm = to_uint8(np.random.uniform(low, high, (h,w)))
    noised = cv2.add(noised, norm)

    return noised


#<!--------------------------------------------------------------------------->
#<!--                                FILTERS                                -->
#<!--------------------------------------------------------------------------->

def saltAndPepperFilter(image, n=3):
    """
    This function removes salt-and-pepper noise from the input image.
    """
    # You have to filter this variable.
    filtered = image.copy()
    h, w = filtered.shape

    # TODO: Your code here
    kernel = np.ones((n,n))*(1.0/n**2)
    filtered = cv2.filter2D(filtered, -1, kernel)

    return filtered

def gaussianFilter(image, n=5):
    """
    This function removes Gaussian noise from the input image.
    """
    # You have to filter this variable.
    filtered = image.copy()
    h, w = filtered.shape

    # TODO: Your code here
    filtered = cv2.GaussianBlur(filtered, (n,n), 3)

    return filtered

def uniformFilter(image, n=5):
    """
    This function removes uniform noise from the input image.
    """
    # You have to filter this variable.
    filtered = image.copy()
    h, w = filtered.shape

    # TODO: Your code here
    filtered = cv2.GaussianBlur(filtered, (n,n), 3)

    return filtered


#<!--------------------------------------------------------------------------->
#<!--                              INPUT IMAGE                              -->
#<!--------------------------------------------------------------------------->

root = Path(path.dirname(path.realpath(__file__)))

# Input image filename.
filename = str(root / "./inputs/lena.jpg")

# Loads an image from a file passed as argument.
image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)


#<!--------------------------------------------------------------------------->
#<!--                      ADD NOISE TO THE INPUT IMAGE                     -->
#<!--------------------------------------------------------------------------->

# Create an image with salt and pepper noise.
saltAndPepper = saltAndPepperNoise(image, 0.05)

# Create an image with Gaussian noise.
gaussian = gaussianNoise(image, 0, 50)

# Create an image with uniform noise.
uniform = uniformNoise(image, 0, 50)

# Show the original image and the noised images.
showImages(Uniform=uniform, Gaussian=gaussian, Salt_and_Pepper=saltAndPepper, Original=image)


#<!--------------------------------------------------------------------------->
#<!--                       FILTERING THE NOISED IMAGE                      -->
#<!--------------------------------------------------------------------------->

# Filter the salt and pepper noise.
saltAndPepperFiltered = saltAndPepperFilter(saltAndPepper)

# Filter the Gaussian noise.
gaussianFiltered = gaussianFilter(gaussian)

# Filter the uniform noise.
uniformFiltered = uniformFilter(uniform)

# Show the original image and the noised images.
showImages(Uniform=uniformFiltered, Gaussian=gaussianFiltered,
           Salt_and_Pepper=saltAndPepperFiltered, Original=image)



#<!--------------------------------------------------------------------------->
#<!--                  SHOW NOISE AS A 1D IMPULSE FUNCTION                  -->
#<!--------------------------------------------------------------------------->

# Image resolution
h, w = image.shape

# Create a new matplotlib window.
fig = plt.figure()
ax = [plt.subplot(2, 4, x+1) for x in range(8)]

# Define the row slider.
axcolor = "lightgoldenrodyellow"
slider_ax = plt.axes([0.1225, 0.02, 0.78, 0.03], facecolor=axcolor)
slider_row = Slider(slider_ax, "Row", 1.0, h, valinit=1, valfmt="%i")
slider_row.on_changed(partial(updateRow, slider_row))

# Show the matplotlib window.
showSignals(1, Uniform=uniform, Gaussian=gaussian, Salt_and_Pepper=saltAndPepper,
            Original=image, Uniform_Filtered=uniformFiltered,
            Gaussian_Filtered=gaussianFiltered,
            Salt_and_Pepper_Filtered=saltAndPepperFiltered)

# Show the images.
plt.show()