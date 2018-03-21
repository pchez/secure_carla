'''Camera Attack'''

import numpy as np
import matplotlib.pyplot as plt

from . import sensor

from skimage import data
from skimage.transform import swirl
from skimage import transform, data, io

def fisheye(xy):
    center = np.mean(xy, axis=0)
    center = [400.0,200.0]
    xc, yc = (xy - center).T

    # Polar coordinates
    r = np.sqrt(xc**2 + yc**2)
    theta = np.arctan2(yc, xc)

    #r = 0.8 * np.exp(r**(1/2.1) / 1.8)
    #r = 4 * np.exp(r**(1/2.1) / 1.8) #reduces the effect 
    #r = 2.1 * np.exp(r**(1/2.1) / 3.5) #level 1 
    r = 2.1 * np.exp(r**(1/1.5)/11.3) #level 2
    #r = 2.1 * np.exp(r**(1/3.0)/1.45) #level 3
    #r = 2.1 * np.exp(r**(1/3.0)/1.2) #level 3

    return np.column_stack((
        r * np.cos(theta), r * np.sin(theta)
        )) + center

def noise_attack(image):
	noise = np.random.randint(0,130,image.shape,dtype=np.dtype("uint8"))
	#print image.shape
	#print type(image)
	return image + noise

def block_attack(image):
	image[115:355,325:450,:] = 0
	return image

def flip_attack(image):
	return np.fliplr(image)

def warp_attack(image):
	image = transform.warp(image, fisheye, mode='wrap')
	image = image*255
	image = image.astype("uint8")
	#print image.shape
	#print type(image)
	return image

def colour_attack(image):
	tmp_array = np.zeros(image.shape, dtype="uint8")
	tmp_array[:,:,0] = image[:,:,0]
	image = tmp_array
	return image


def perform_attack(image):
	perform_noise_attack = False 
	perform_block_attack = False
	perform_flip_attack = False
	perform_warp_attack =  True
	perform_colour_attack = False

	width = image.width
	height = image.height

	image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	image = np.reshape(image, (height, width, 4))
	image.setflags(write=1)

	if perform_noise_attack:
		image = noise_attack(image)

	if perform_block_attack: 
		image = block_attack(image)

	if perform_flip_attack:
		image = flip_attack(image)
	
	if perform_warp_attack:
		image = warp_attack(image)

	if perform_colour_attack:
		image = colour_attack(image)

	#Serialize the data
	image = np.reshape(image, (height*width*4))
	image = image.tostring()
	return image
