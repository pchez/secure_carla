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

try:
    from . import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError('cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')


class SecureCarla(object):
    def __init__(self):
	#Need to load the variance and mean parameters here 
	self.wait_counter = 0
	return

    def add_location_noise(self, agent, mean, var):
	agent.transform.location.x = agent.transform.location.x + np.random.normal(mean,var)
	agent.transform.location.y = agent.transform.location.y + np.random.normal(mean,var)
	agent.transform.location.z = agent.transform.location.z + np.random.normal(mean,var)
    
    def add_rotation_noise(self, agent, mean, var):
	agent.transform.rotation.pitch = agent.transform.rotation.pitch + np.random.normal(mean,var)
	agent.transform.rotation.yaw = agent.transform.rotation.yaw + np.random.normal(mean,var)
	agent.transform.rotation.roll = agent.transform.rotation.roll + np.random.normal(mean,var)
    
    def add_transform_noise(self, agent, mean, var):
	self.add_location_noise(agent, 0, 10)
	self.add_rotation_noise(agent, 0, 1)

    def add_acceleration_noise(self, agent, mean, var):
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
	self.add_transform_noise(agent.traffic_light, 0, 10)

    def speed_limit_sign_inject(self, agent):
	self.add_transform_noise(agent.speed_limit_sign, 0, 10)

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

	image = sensor_data['CameraRGB']
	array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image.height, image.width, 4))

	#Noise injection
	noise = np.random.randint(0,130,array.shape,dtype=np.dtype("uint8"))
	#array = array + noise

	#Block injection
	array.setflags(write=1)
	#array[125:375,325:475,:] = 0

	#Serialize the data
	array = np.reshape(array, (image.height*image.width*4))
	array = array.tostring()
	sensor_data['CameraRGB'].raw_data = array
	
	#self.wait_counter = self.wait_counter + 1
	#print self.wait_counter
	if self.wait_counter >= 50:
		sensor_data['CameraRGB'].save_to_disk("/home/carla/Documents/carla_images/camera_outputs/block.png")
		print("Done")
		time.sleep(5)
	return measurements, sensor_data
	

