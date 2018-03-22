'''Camera Attack'''

import numpy as np
import matplotlib.pyplot as plt

from . import sensor
from warp_functions import *

from skimage import data
from skimage.transform import swirl
from skimage import transform, data, io, filters
import Image
import time


class CameraAttack(object):
	def __init__(self, config):
		self.config = config
		return

	def noise_attack(self,image):
		noise = np.random.randint(self.config['noise']['attack_mean'],
					  self.config['noise']['attack_variance'],
					  image.shape,dtype=np.dtype("uint8"))
		return image + noise

	def block_attack(self,image):
		position_x = int(self.config['block']['position_x'])
		position_y = int(self.config['block']['position_y'])
		width = int(self.config['block']['width'])
		height = int(self.config['block']['height'])
		image[position_y:position_y+height,position_x:position_x+width,:] = 0
		return image

	def flip_attack(self,image):
		if self.config['flip']['use_up_down_flip']:
			return np.flipud(image)
		elif self.config['flip']['use_left_right_flip']:
			return np.fliplr(image)
		return image

	def warp_attack(self,image):
		if self.config['warp']['use_fish_eye_attack']:
			image = transform.warp(image, fisheye, mode='wrap')
		elif self.config['warp']['use_swirl_attack']:
			image = swirl(image,(image.shape[1]/2, image.shape[0]/2), rotation=0, strength=3, radius=600)
		image = image*255
		image = image.astype("uint8")
		return image

	def colour_attack(self,image):
		tmp_array = np.zeros(image.shape, dtype="uint8")
		tmp_array[:,:,0] = (image[:,:,0]*self.config['colour']['blue_channel']).astype("uint8")
		tmp_array[:,:,1] = (image[:,:,1]*self.config['colour']['green_channel']).astype("uint8")
		tmp_array[:,:,2] = (image[:,:,2]*self.config['colour']['red_channel']).astype("uint8")
		image = tmp_array
		return image

	def laser_attack(self,image):
		foreground = Image.open("/home/carla/Downloads/laser_transparent.png")
		foreground = np.array(foreground)
		foreground[:,:,0] = 0
		foreground[:,:,2] = 0
		foreground = Image.fromarray(foreground)
	
		basewidth = 800
		wpercent = (basewidth/float(foreground.size[0]))
		hsize = int((float(foreground.size[1])*float(wpercent)))
		foreground = foreground.resize((basewidth,hsize), Image.ANTIALIAS)
		foreground = foreground.crop((0,200,800,800))

		background = Image.fromarray(image)
		background = background.convert('RGBA')
		print background.size
		combined = Image.new('RGBA',background.size)
		combined = Image.alpha_composite(combined,background)
		combined = Image.alpha_composite(combined,foreground)
		combined = np.array(combined)	

		return combined

	def sticker_attack(self,image):
		foreground = Image.open(self.config['sticker']['image_path'])
		foreground = np.array(foreground)
		foreground = Image.fromarray(foreground)

		background = Image.fromarray(image)
		background = background.convert('RGBA')
		combined = Image.new('RGBA',background.size)
		combined = Image.alpha_composite(combined,background)
		combined = Image.alpha_composite(combined,foreground)
		combined = np.array(combined)	

		return combined	
	def blur_attack(self,image):
		image = filters.gaussian(image, sigma =self.config['blur']['sigma'], multichannel=True)
		image = image*255
		image = image.astype("uint8")
		return image


	def perform_attack(self,image):

		width = image.width
		height = image.height

		image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
		image = np.reshape(image, (height, width, 4))
		image.setflags(write=1)

		if self.config['noise']['use_attack']:
			image = self.noise_attack(image)

		if self.config['block']['use_attack']: 
			image = self.block_attack(image)

		if self.config['flip']['use_attack']:
			image = self.flip_attack(image)
	
		if self.config['warp']['use_attack']:
			image = self.warp_attack(image)

		if self.config['colour']['use_attack']:
			image = self.colour_attack(image)

		if self.config['laser']['use_attack']:
			image = self.laser_attack(image)
	
		if self.config['sticker']['use_attack']:
			image = self.sticker_attack(image)

		if self.config['blur']['use_attack']:
			image = self.blur_attack(image)

		#Serialize the data
		image = np.reshape(image, (height*width*4))
		image = image.tostring()
		return image
