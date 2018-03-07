# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Secure CARLA."""

import struct

from contextlib import contextmanager

from . import sensor
from . import settings
from . import tcp
from . import util
from . import image_converter
import logging
import numpy as np
import time
import sys
import configparser
import os.path

try:
    from . import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError('cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')


class SecureCarla(object):
    def __init__(self, config_file=None):

	self.wait_counter = 0
	# Load variance, mean, and offset parameters here:
        # Parse config file if config file provided 
        
	print config_file
        if config_file is not None:
            self.config = self.parse_config(config_file)

        return
        

    def parse_config(self, config_file):

        config_params = {}
        
        # Initialize config parser object
        config = configparser.ConfigParser()

        if not os.path.isfile(config_file):
            print('Cannot find', config_file+'. ', 'Exiting now.')
            exit(1)
        else:
            # Read and parse config file
            config.read(config_file)
            
            for config_key in config.items():
                if config_key[0] == 'DEFAULT':
                    continue
                config_params[config_key[0]] = dict(config.items(config_key[0]))
                config_params[config_key[0]] = {key:float(val) for key, val in config_params[config_key[0]].items()}
            
            # Print configuration parameters loaded
            print('Finished loading configs:')
            for key in config_params.keys():
                print('---',key,'---')
                print(config_params[key])

        return config_params


    def add_location_noise(self, agent, mean, var):
        agent.transform.location.x = agent.transform.location.x + np.random.normal(mean,var)
	agent.transform.location.y = agent.transform.location.y + np.random.normal(mean,var)
	agent.transform.location.z = agent.transform.location.z + np.random.normal(mean,var)
    
    def add_rotation_noise(self, agent, mean, var):
	agent.transform.rotation.pitch = agent.transform.rotation.pitch + np.random.normal(mean,var)
	agent.transform.rotation.yaw = agent.transform.rotation.yaw + np.random.normal(mean,var)
	agent.transform.rotation.roll = agent.transform.rotation.roll + np.random.normal(mean,var)
    
    def add_transform_noise(self, agent, lmean, lvar, rmean, rvar):
	self.add_location_noise(agent, lmean, lvar)
	self.add_rotation_noise(agent, rmean, rvar)

    def add_acceleration_noise(self, agent):
        agent.acceleration.x = agent.acceleration.x + np.random.normal(mean,var)
	agent.acceleration.y = agent.acceleration.y + np.random.normal(mean,var)
	agent.acceleration.z = agent.acceleration.z + np.random.normal(mean,var)

    def add_box_extent_noise(self, agent, mean, var):
	agent.box_extent.x = agent.box_extent.x + np.random.normal(mean,var)
	agent.box_extent.y = agent.box_extent.y + np.random.normal(mean,var)
	agent.box_extent.z = agent.box_extent.z + np.random.normal(mean,var)

    def add_speed_noise(self, agent, mean, var):
	agent.forward_speed = agent.forward_speed + np.random.normal(mean,var)

    def traffic_light_inject(self, agent):
        this_config = self.config['trafficlight']
	self.add_transform_noise(agent.traffic_light, this_config['location_mean'], this_config['location_var'], this_config['rotation_mean'], this_config['rotation_var'])

    def speed_limit_sign_inject(self, agent):
        this_config = self.config['speedlimit']
	self.add_transform_noise(agent.speed_limit_sign, this_config['location_mean'], this_config['location_var'], this_config['rotation_mean'], this_config['rotation_var'])

    def vehicle_inject(self, agent):
        this_config = self.config['vehicle']
	self.add_transform_noise(agent.vehicle, this_config['location_mean'], this_config['location_var'], this_config['rotation_mean'], this_config['rotation_var'])
	self.add_box_extent_noise(agent.vehicle,0,10)
	self.add_speed_noise(agent.vehicle,this_config['speed_mean'],this_config['speed_var'])

    def pedestrian_inject(self, agent):
        this_config = self.config['pedestrian']
	self.add_transform_noise(agent.pedestrian, this_config['location_mean'], this_config['location_var'], this_config['rotation_mean'], this_config['rotation_var'])
	self.add_box_extent_noise(agent.pedestrian,0,10)
	self.add_speed_noise(agent.pedestrian,this_config['speed_mean'],this_config['speed_var'])

    def agent_inject(self, agent_type, agent):
	if(agent_type == 'traffic_light'):
		self.traffic_light_inject(agent)
	elif(agent_type == 'speed_limit_sign'):
		self.speed_limit_sign_inject(agent)
	elif(agent_type == 'vehicle'):
		self.vehicle_inject(agent)
	elif(agent_type == 'pedestrian'):
		self.pedestrian_inject(agent)
	#agents ={
	#	'traffic_light' : self.traffic_light_inject(agent),
	#	'speed_limit_sign' : self.speed_limit_sign_inject(agent),
	#	'vehicle' : self.vehicle_inject(agent),
	#	'pedestrian' : self.pedestrian_inject(agent)}[agent_type](agent

	#print(agents[agent_type](agent))

    def player_inject(self, player):
	this_config = self.config['player']
        self.add_transform_noise(player, this_config['location_mean'], this_config['location_var'], this_config['rotation_mean'], this_config['rotation_var'])
	self.add_acceleration_noise(player,this_config['accel_mean'],this_config['accel_var'])
	self.add_speed_noise(player,this_config['speed_mean'],this_config['speed_var'])

    def inject_adversarial(self,measurements, sensor_data):
	'''
	logging.info("Measurement Values:")
        logging.info('Speed = %f ',measurements.player_measurements.forward_speed)
 	logging.info('Accel = %f ',measurements.player_measurements.acceleration.x)
	logging.info('x = %f ',measurements.player_measurements.transform.location.x)
 	logging.info('y = %f ',measurements.player_measurements.transform.location.y)
	logging.info('z = %f ',measurements.player_measurements.transform.location.z)
	logging.info('pitch = %f ',measurements.player_measurements.transform.rotation.pitch)
 	logging.info('yaw = %f ',measurements.player_measurements.transform.rotation.yaw)
	logging.info('roll = %f ',measurements.player_measurements.transform.rotation.roll)
	
	#Inject noise into the player measurements
	self.player_inject(measurements.player_measurements)

	logging.info("Aversarial Measurement Values:")
        logging.info('Speed = %f ',measurements.player_measurements.forward_speed)
 	logging.info('Accel = %f ',measurements.player_measurements.acceleration.x)
	logging.info('x = %f ',measurements.player_measurements.transform.location.x)
 	logging.info('y = %f ',measurements.player_measurements.transform.location.y)
	logging.info('z = %f ',measurements.player_measurements.transform.location.z)
	logging.info('pitch = %f ',measurements.player_measurements.transform.rotation.pitch)
 	logging.info('yaw = %f ',measurements.player_measurements.transform.rotation.yaw)
	logging.info('roll = %f ',measurements.player_measurements.transform.rotation.roll)
	for a in measurements.non_player_agents:
		if a.WhichOneof('agent') == 'pedestrian':
			self.agent_inject(a.WhichOneof('agent'), a)
			logging.info('x = %f ',a.pedestrian.transform.location.x)
 			logging.info('y = %f ',a.pedestrian.transform.location.y)
			logging.info('z = %f ',a.pedestrian.transform.location.z)
			break
	'''
	#print(len(sensor_data['CameraRGB'].raw_data))
	#sensor_data['CameraRGB'].data = sensor_data['CameraRGB'].data + np.random.normal(0,10,size=sensor_data['CameraRGB'].data.shape)
	#sensor_data['CameraRGB'].save_to_disk("/home/carla/Documents/carla_images/camera_outputs/0000.png")
	#print("Done")
	#time.sleep(5)
	#print(sensor_data['CameraRGB'].data.shape)
	converted_data = image_converter.to_rgb_array(sensor_data['CameraRGB'])

	image = sensor_data['CameraRGB']
	array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image.height, image.width, 4))
	#The data is in ABGR format at the moment
	noise = np.random.randint(0,130,array.shape,dtype=np.dtype("uint8"))
	#print noise[0:10,0:10,:]
	#print type(noise[0,0,0])
	#print type(array[0,0,0])
	#array = array + noise
	array.setflags(write=1)
	array[150:250,350:450,:] = 0
	#for x in range(100,200):
	  #    array[x,100:200] = 0
	#print array.shape
	array = np.reshape(array, (image.height*image.width*4))
	
	#array = array[:, :, :3]
    	#array = array[:, :, ::-1]
	#print array.shape
	array = array.tostring()
	print len(array)
	print array == sensor_data['CameraRGB'].raw_data 
	sensor_data['CameraRGB'].raw_data = array
    def vehicle_inject(self, agent):
	self.add_transform_noise(agent.vehicle, 0, 10)
	self.add_box_extent_noise(agent.vehicle,0,10)
	self.add_speed_noise(agent.vehicle,0,5)

    def pedestrian_inject(self, agent):
	self.add_transform_noise(agent.pedestrian, 0, 10)
	self.add_box_extent_noise(agent.pedestrian,0,10)
	self.add_speed_noise(agent.pedestrian,0,5)

    def agent_inject(self, agent_type, agent):
	if(agent_type == 'traffic_light'):
		self.traffic_light_inject(agent)
	elif(agent_type == 'speed_limit_sign'):
		self.speed_limit_sign_inject(agent)
	elif(agent_type == 'vehicle'):
		self.vehicle_inject(agent)
	elif(agent_type == 'pedestrian'):
		self.pedestrian_inject(agent)
	#agents ={
	#	'traffic_light' : self.traffic_light_inject(agent),
	#	'speed_limit_sign' : self.speed_limit_sign_inject(agent),
	#	'vehicle' : self.vehicle_inject(agent),
	#	'pedestrian' : self.pedestrian_inject(agent)}[agent_type](agent

	#print(agents[agent_type](agent))

    def player_inject(self, player):
	self.add_transform_noise(player,0,10)
	self.add_acceleration_noise(player,0,1)
	self.add_speed_noise(player,0,5)

    def inject_adversarial(self,measurements, sensor_data):
	logging.info("Measurement Values:")
	'''
        logging.info('Speed = %f ',measurements.player_measurements.forward_speed)
 	logging.info('Accel = %f ',measurements.player_measurements.acceleration.x)
	logging.info('x = %f ',measurements.player_measurements.transform.location.x)
 	logging.info('y = %f ',measurements.player_measurements.transform.location.y)
	logging.info('z = %f ',measurements.player_measurements.transform.location.z)
	logging.info('pitch = %f ',measurements.player_measurements.transform.rotation.pitch)
 	logging.info('yaw = %f ',measurements.player_measurements.transform.rotation.yaw)
	logging.info('roll = %f ',measurements.player_measurements.transform.rotation.roll)
	
	#Inject noise into the player measurements
	self.player_inject(measurements.player_measurements)

	logging.info("Aversarial Measurement Values:")
        logging.info('Speed = %f ',measurements.player_measurements.forward_speed)
 	logging.info('Accel = %f ',measurements.player_measurements.acceleration.x)
	logging.info('x = %f ',measurements.player_measurements.transform.location.x)
 	logging.info('y = %f ',measurements.player_measurements.transform.location.y)
	logging.info('z = %f ',measurements.player_measurements.transform.location.z)
	logging.info('pitch = %f ',measurements.player_measurements.transform.rotation.pitch)
 	logging.info('yaw = %f ',measurements.player_measurements.transform.rotation.yaw)
	logging.info('roll = %f ',measurements.player_measurements.transform.rotation.roll)
	for a in measurements.non_player_agents:
		if a.WhichOneof('agent') == 'pedestrian':
			self.agent_inject(a.WhichOneof('agent'), a)
			logging.info('x = %f ',a.pedestrian.transform.location.x)
 			logging.info('y = %f ',a.pedestrian.transform.location.y)
			logging.info('z = %f ',a.pedestrian.transform.location.z)
			break
	'''
	image = sensor_data['CameraRGB']
	array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image.height, image.width, 4))

	#Noise injection
	noise = np.random.randint(0,130,array.shape,dtype=np.dtype("uint8"))
	#array = array + noise

	#Block injection
	array.setflags(write=1)
	#array[115:355,325:450,:] = 0

	#Flip the image
	#array = np.flipud(array)

	#One channel only
	tmp_array = np.zeros(array.shape, dtype="uint8")
	tmp_array[:,:,0] = array[:,:,0]
	array = tmp_array

	#Serialize the data
	array = np.reshape(array, (image.height*image.width*4))
	array = array.tostring()
	sensor_data['CameraRGB'].raw_data = array
	
	#self.wait_counter = self.wait_counter + 1
	#print self.wait_counter
	if self.wait_counter >= 50:
		sensor_data['CameraRGB'].save_to_disk("/home/carla/Documents/carla_images/camera_outputs/one_colour.png")
		print("Done")
		time.sleep(5)
	return measurements, sensor_data
	

