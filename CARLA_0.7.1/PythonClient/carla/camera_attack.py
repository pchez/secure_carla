'''Camera Attack'''

import numpy as np
import matplotlib.pyplot as plt

from . import sensor

from skimage import data
from skimage.transform import swirl
from skimage import transform, data, io

def noise_attack(image):
	noise = np.random.randint(0,130,image.shape,dtype=np.dtype("uint8"))
	return image + noise

def block_attack(image):
	image[115:355,325:450,:] = 0
	return image

def flip_attack(image):
	return np.fliplr(image)

def warp_attack(image):

	return image

def colour_attack(image):
	tmp_array = np.zeros(image.shape, dtype="uint8")
	tmp_array[:,:,0] = image[:,:,0]
	image = tmp_array
	return image

def perform_attack(image):
	perform_noise_attack = False 
	perform_block_attack = False
	perform_flip_attack = True
	perform_warp_attack = False
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
