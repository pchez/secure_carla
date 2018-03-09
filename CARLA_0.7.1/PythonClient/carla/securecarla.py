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

PLAYER_ALL = 0
PLAYER_MEASUREMENTS = 1

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

    def get_distance_to_agent(self, player, agent):
        distance = np.sqrt((player.transform.location.x - agent.transform.location.x)**2 + (player.transform.location.y - agent.transform.location.y)**2 + (player.transform.location.z - agent.transform.location.z)**2)

        return distance

    # Returns the distance value under noise and attack 
    def distance_attack(self, this_config, agent, player=None):
        use_gaussian = int(this_config['use_gaussian_noise'])
        if use_gaussian:
            noise = np.random.normal(this_config['dist_noise_mean'], this_config['dist_noise_var'])
        else:
            noise = np.random.uniform(this_config['dist_noise_low'], this_config['dist_noise_low']) 

        use_attack = int(this_config['use_attack'])
        if use_attack:
            attack = np.random.normal(this_config['dist_attack_mean'], this_config['dist_attack_var'])
        else:
            attack = 0

        distance = get_distance_to_agent(agent, player)
	distance = distance + noise + attack 

    # Modifies the accel value of the agent/player with noise and attack
    def accel_attack(self, this_config, agent):
        use_gaussian = int(this_config['use_gaussian_noise'])
        if use_gaussian:
            noise = np.random.normal(this_config['acceel_noise_mean'], this_config['accel_noise_var'])
        else:
            noise = np.random.uniform(this_config['accel_noise_low'], this_config['accel_noise_low']) 
        
        use_attack = int(this_config['use_attack'])
        if use_attack:
            attack = np.random.normal(this_config['accel_attack_mean'], this_config['accel_attack_var'])
        else:
            attack = 0
        
        agent.acceleration.x = agent.acceleration.x + noise + attack
	agent.acceleration.y = agent.acceleration.y + noise + attack
	agent.acceleration.z = agent.acceleration.z + noise + attack

    # Modifies the forward speed value of the agent/player with noise and attack
    def speed_attack(self, this_config, agent):
        use_gaussian = int(this_config['use_gaussian_noise'])
	if use_gaussian:
            noise = np.random.normal(this_config['speed_noise_mean'], this_config['speed_noise_var'])
        else:
            noise = np.random.uniform(this_config['speed_noise_low'], this_config['speed_noise_low']) 
        
        use_attack = int(this_config['use_attack'])
        if use_attack:
            attack = np.random.normal(this_config['speed_attack_mean'], this_config['speed_attack_var'])
        else:
            attack = 0

        agent.forward_speed = agent.forward_speed + noise + attack

    def traffic_light_inject(self, agent, player):
        this_config = self.config['trafficlight']
	self.distance_attack(this_config, agent.traffic_light, player)

    def speed_limit_sign_inject(self, agent, player):
        this_config = self.config['speedlimit']
	self.distance_attack(this_config, agent.speed_limit_sign, player)

    def vehicle_inject(self, agent, player):
        this_config = self.config['vehicle']
	self.distance_attack(this_config, agent.vehicle, player)
	self.speed_attack(this_config, agent.vehicle)

    def pedestrian_inject(self, agent, player):
        this_config = self.config['pedestrian']
	self.distance_attack(this_config, agent.pedestrian, player)
	self.speed_attack(this_config, agent.pedestrian)

    def agent_inject(self, player, agent_type, agent):
	if(agent_type == 'traffic_light'):
		self.traffic_light_inject(agent, player)
	elif(agent_type == 'speed_limit_sign'):
		self.speed_limit_sign_inject(agent, player)
	elif(agent_type == 'vehicle'):
		self.vehicle_inject(agent, player)
	elif(agent_type == 'pedestrian'):
		self.pedestrian_inject(agent, player)
	#agents ={
	#	'traffic_light' : self.traffic_light_inject(agent),
	#	'speed_limit_sign' : self.speed_limit_sign_inject(agent),
	#	'vehicle' : self.vehicle_inject(agent),
	#	'pedestrian' : self.pedestrian_inject(agent)}[agent_type](agent

	#print(agents[agent_type](agent))

    def player_inject(self, player):
	this_config = self.config['player']
	self.accel_attack(this_config, player)
	self.speed_attack(this_config, player)

    def log_measurements(self, mode):
        if mode == PLAYER_ALL:
            logging.info("Measurement Values:")
            logging.info('Speed = %f ',measurements.player_measurements.forward_speed)
            logging.info('Accel = %f ',measurements.player_measurements.acceleration.x)
            logging.info('x = %f ',measurements.player_measurements.transform.location.x)
            logging.info('y = %f ',measurements.player_measurements.transform.location.y)
            logging.info('z = %f ',measurements.player_measurements.transform.location.z)
            logging.info('pitch = %f ',measurements.player_measurements.transform.rotation.pitch)
            logging.info('yaw = %f ',measurements.player_measurements.transform.rotation.yaw)
            logging.info('roll = %f ',measurements.player_measurements.transform.rotation.roll)
            
            logging.info("Adversarial Measurement Values:")
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
        
	self.distance_attack(agent.vehicle, 0, 10)
	self.speed_attack(agent.vehicle,0,5)



    def inject_adversarial(self,measurements, sensor_data):
	
	#Inject noise into the player measurements
	self.player_inject(measurements.player_measurements)
        log_measurements(PLAYER_ALL)

	for a in measurements.non_player_agents:
            if a.WhichOneof('agent') == 'pedestrian':
                self.agent_inject(a.WhichOneof('agent'), a)
                logging.info('x = %f ',a.pedestrian.transform.location.x)
                logging.info('y = %f ',a.pedestrian.transform.location.y)
                logging.info('z = %f ',a.pedestrian.transform.location.z)
                break
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
	

