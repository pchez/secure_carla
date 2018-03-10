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
        
        self.true_distances = []            # True distances to each agent
        self.adversarial_distances = []     # False distances to each agent
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
        distance = self.get_distance_to_agent(agent, player)
	self.true_distances.append(distance)
        distance = distance + noise + attack 
        self.adversarial_distances.append(distance)

    # Modifies the accel value of the agent/player with noise and attack
    def accel_attack(self, this_config, agent):
        use_gaussian = int(this_config['use_gaussian_noise'])
        if use_gaussian:
            noise = np.random.normal(this_config['accel_noise_mean'], this_config['accel_noise_var'])
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

    def log_measurements(self, measurements):
        logging.info('Player speed = %f ',measurements.player_measurements.forward_speed)
        logging.info('Player accel = %f ',measurements.player_measurements.acceleration.x)
        logging.info('Player x = %f ',measurements.player_measurements.transform.location.x)
        logging.info('Player y = %f ',measurements.player_measurements.transform.location.y)
        logging.info('Player z = %f ',measurements.player_measurements.transform.location.z)
        
        for i,a in enumerate(measurements.non_player_agents):
            if a.WhichOneof('agent') == 'vehicle':
                logging.info('vehicle forward speed = %f ', a.vehicle.forward_speed)
                logging.info('vehicle x = %f ', a.vehicle.transform.location.x)
                logging.info('vehicle y = %f ', a.vehicle.transform.location.y)
                logging.info('vehicle z = %f ', a.vehicle.transform.location.z)
                # Only print distance values if attack already launched and new distances have
                # already been stored in true_distances and adversarial_distances
                if i < len(self.true_distances):
                    logging.info('true distance to agent: %f', self.true_distances[i])
                    logging.info('false distance to agent: %f', self.adversarial_distances[i])
                break
    


    def inject_adversarial(self, measurements, sensor_data):
	self.true_distances = []
        self.adversarial_distances = []

        logging.info("Measurement Values:")
        self.log_measurements(measurements)

        #Inject noise into the player measurements
	self.player_inject(measurements.player_measurements)
        
	for a in measurements.non_player_agents:
            self.agent_inject(measurements.player_measurements, a.WhichOneof('agent'), a)
	
        logging.info("Adversarial Measurement Values:")
        self.log_measurements(measurements)
        
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
	

