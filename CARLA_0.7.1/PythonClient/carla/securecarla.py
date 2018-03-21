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
from . import camera_attack
import csv
import logging
import numpy as np
import time
import sys
import configparser
import os.path
import datetime
import Queue

from collections import OrderedDict

try:
    from . import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError('cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')


class SecureCarla(object):
    def __init__(self, config_file=None):
	self.wait_counter = 0
	self.step = 0
	self.output_time = datetime.datetime.now()
	self.agent_num = 0
	self.first_reading = True
	self.past_player = None

	# Load variance, mean, and offset parameters here:
        # Parse config file if config file provided 
        if config_file is not None:
            self.config = self.parse_config(config_file)
	self.csv_file = '../../securecarla_details.csv'

	self.true_distances = {}            # True distances to each agent
	self.noise_distances = {}	    # Distances with added noise
        self.adversarial_distances = {}     # False distances to each agent
	self._dict_distances = OrderedDict([('step', -1),
				('src_node', -1),
				('dest_node', -1),
				('noise_distance', -1),
				('adversarial_distance', -1),
				('true_distance', -1),
				])
	#for s in range(0,int(self.config['all']['num_sensors'])):
	#	self._dict_distances['sensor{}'.format(s)] = -1

	self.meas_fifo = Queue.Queue() #to hold measurements during delay attack
        self.sensor_fifo = Queue.Queue() #to hold camera frames during delay attack
        self.meas_buf = []
        self.sensor_buf = []

	print self._dict_distances

	with open(self.csv_file, 'w') as rfd:

                rw = csv.DictWriter(rfd, self._dict_distances.keys())
                rw.writeheader()
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

    def gauss_var_from_dist(self, distance, fixed_variance):
        grad = fixed_variance/10
        variance = grad*distance
	return variance

    def uniform_parms_from_dist(self, distance, fixed_low, fixed_high):
        grad_low = fixed_low/10
        low = grad_low*distance

	grad_high = fixed_high/10
        high = grad_high*distance
	return low, high

    # Return which sensor can see the agent based on the player's yaw and agent's x,y location
    def is_in_sensors_fov(self, player, agent, distance_to_agent):
        sensor_fov = self.config['all']['sensor_fov']
        player_yaw = player.transform.orientation.yaw
        player_x = player.transform.location.x
        player_y = player.transform.location.y
        agent_x = agent.transform.location.x
        agent_y = agent.transform.location.y
        sensor_front_yaw = player_yaw
        sensor_back_yaw = (sensor_front_yaw - 180) if (sensor_front_yaw - 180) > -360 else (sensor_front_yaw + 180)
        sensor_right_yaw = (sensor_right_yaw - 90) if (sensor_right_yaw - 90) > -360 else (sensor_right_yaw + 270)
        sensor_left_yaw = (sensor_left_yaw + 90) if (sensor_left_yaw + 90) < 360 else (sensor_left_yaw - 270)
        
        # Calculate agent's angle from player, convert to be within -360 to 360
        agent_angle = player_yaw - np.arctan((agent_y - player_y)/(agent_x - player_x))
        if agent_angle > 360:
            agent_angle = agent_angle - 360
        elif agent_angle < -360:
            agent_angle = agent_angle + 360
        else: #360 or -360
            agent_angle = 0

        # Determine which sensor's FOV the agent falls in. For now, just select one of the sensors
        # (basically assuming that only one distance sensor picking up the agent)
        # 0: SENSOR_FRONT, 1: SENSOR_BACK, 2: SENSOR_RIGHT, 3: SENSOR_RIGHT
        sensors = [sensor_front_yaw, sensor_back_yaw, sensor_right_yaw, sensor_left_yaw]
        for i,sensor in enumerate(sensors):
            left_bound = sensor + sensor_fov
            right_bound = sensor - sensor_fov
            left_bound = left_bound if left_bound > 360 else left_bound - 360
            right_bound = right_bound if right_bound < 360 else right_bound + 360

            if sensor < left_bound and sensor > right_bound:
                which_sensor = i
                break

        return which_sensor
    
    # Returns the distance value under noise and attack 
    def distance_threshold_attack(self, this_config, agent, agent_id, player=None):

	distance = self.get_distance_to_agent(agent, player)
        	
	if distance < 1000:
            print("Distance: {}, Agent_ID: {}".format(distance,agent_id))
        """
        TODO: [pseudocode]
        get yaw, get FOV from config file, theta measured as angle of the sensor direction 
        loop through 4 sensors, if in the FOV of any of them, apply the attack
        is_in_sensors_fov function (player, agent, distance_to_agent):
            see if the agent's position relative to the player falls within any (choose 1) of the following:
            sensor_front: yaw = self.yaw
            sensor_left: yaw - 90
            sensor_right: yaw + 90
            sensor_back: yaw - 180
            remember to convert to within 0-360
        """
	if(distance < self.config['all']['distance_threshold']):
            self.true_distances[agent_id] = distance
            adversarial_distances = []
            noise_distances = []
            which_sensor = self.is_in_sensors_fov(player, agent, distance)
            print agent_id, 'being detected by', which_sensor

            for s in range(0,int(self.config['all']['num_sensors'])):
                use_gaussian = int(this_config['use_gaussian_noise'])
                if use_gaussian:
                    # The variance is distance dependent
                    variance = self.gauss_var_from_dist(distance, this_config['dist_noise_var'])
                    noise = np.random.normal(this_config['dist_noise_mean'], variance)
                else:
                    # The low and high parameters are distance dependent
                    low, high = self.uniform_parms_from_dist(distance, this_config['dist_noise_low'], this_config['dist_noise_high'])
                    noise = np.random.uniform(low, high) 
                noise_distances.append(distance + noise)

                use_attack = int(this_config['use_attack'])
                if use_attack:
                    attack = np.random.normal(this_config['dist_attack_mean'], this_config['dist_attack_var'])
                else:
                    attack = 0
                adversarial_distances.append(distance + noise + attack)
            self.noise_distances[agent_id] = noise_distances
            self.adversarial_distances[agent_id] = adversarial_distances

    # Modifies the accel value of the agent/player with noise and attack
    def accel_attack(self, this_config, agent):
        use_gaussian = int(this_config['use_gaussian_noise'])
        if use_gaussian:
            noise = np.random.normal(this_config['accel_noise_mean'], this_config['accel_noise_var'])
        else:
            noise = np.random.uniform(this_config['accel_noise_low'], this_config['accel_noise_high']) 
        
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
            noise = np.random.uniform(this_config['speed_noise_low'], this_config['speed_noise_high']) 
        
        use_attack = int(this_config['use_attack'])
        if use_attack:
            attack = np.random.normal(this_config['speed_attack_mean'], this_config['speed_attack_var'])
        else:
            attack = 0

        agent.forward_speed = agent.forward_speed + noise + attack

    def traffic_light_inject(self, agent, player):
        this_config = self.config['trafficlight']
	self.distance_threshold_attack(this_config, agent.traffic_light, agent.id, player)

    def speed_limit_sign_inject(self, agent, player):
        this_config = self.config['speedlimit']
	self.distance_threshold_attack(this_config, agent.speed_limit_sign, agent.id, player)

    def vehicle_inject(self, agent, player):
        this_config = self.config['vehicle']
	self.distance_threshold_attack(this_config, agent.vehicle, agent.id, player)
	self.speed_attack(this_config, agent.vehicle)

    def pedestrian_inject(self, agent, player):
        this_config = self.config['pedestrian']
	self.distance_threshold_attack(this_config, agent.pedestrian, agent.id, player)
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
        #logging.info('Player speed = %f ',measurements.player_measurements.forward_speed)
        #logging.info('Player accel = %f ',measurements.player_measurements.acceleration.x)
        logging.info('Player yaw = %f ',measurements.player_measurements.transform.rotation.yaw)
        logging.info('Player pitch = %f ',measurements.player_measurements.transform.rotation.pitch)
        logging.info('Player roll = %f ',measurements.player_measurements.transform.rotation.roll)
        
	self.agent_num = 0
        for i,a in enumerate(measurements.non_player_agents):
            if a.WhichOneof('agent') == 'vehicle':
		#self.agent_num = self.agent_num +1
		#logging.info('agent_ID: {}'.format(self.agent_num))
                #logging.info('vehicle forward speed = %f ', a.vehicle.forward_speed)
                #logging.info('vehicle x = %f ', a.vehicle.transform.location.x)
                #logging.info('vehicle y = %f ', a.vehicle.transform.location.y)
                #logging.info('vehicle z = %f ', a.vehicle.transform.location.z)
                # Only print distance values if attack already launched and new distances have
                # already been stored in true_distances and adversarial_distances
		if (self.agent_num == 15):
		        if i < len(self.true_distances):
			    self.log_measurement_results(self.noise_distances[a.id], 
							 self.adversarial_distances[a.id],
							 self.true_distances[a.id])
                	logging.info('true distance to agent: %f', self.true_distances[a.id])
                	logging.info('false distance to agent: %f', self.adversarial_distances[a.id][0])
                

    def log_measurement_results(self, noise_distances, adversarial_distances, true_distance):
	
	#now = datetime.datetime.now()
	#if(now.second != self.output_time.second):
	#self._dict_distances['datetime'] = now
	src_node = self.step%int(self.config['all']['num_sensors'])
	self._dict_distances['step'] = self.step
	self._dict_distances['src_node'] = src_node
	self._dict_distances['dest_node'] = 7 
	self._dict_distances['noise_distance'] = noise_distances[src_node]
	self._dict_distances['adversarial_distance'] = adversarial_distances[src_node]
	self._dict_distances['true_distance'] = true_distance

	#for s in range(0,int(self.config['all']['num_sensors'])):
	#	self._dict_distances['sensor{}'.format(s)] = adversarial_distance[s]
	
	with open(self.csv_file, 'a+') as rfd:
		w = csv.DictWriter(rfd, self._dict_distances.keys())
		w.writerow(self._dict_distances)
	#self.output_time = now
	self.step = self.step + 1
    
        return

    def delay_attack(self, curr_frame, nframes, measurements, sensor_data):
        # Don't launch attack until nframes (set in config file) reached
        if curr_frame < nframes:
            self.wait_counter = curr_frame + 1
            self.meas_fifo.put(measurements)
            self.sensor_fifo.put(sensor_data)
            delayed_measurements = measurements
            delayed_sensor_data = sensor_data
        else: # Pop a value and push on latest value
            delayed_measurements = self.meas_fifo.get()
            delayed_sensor_data = self.sensor_fifo.get()
            self.meas_fifo.put(measurements)
            self.sensor_fifo.put(sensor_data)
        return delayed_measurements, delayed_sensor_data

    def frame_swap_attack(self, curr_frame, nframes, measurements, sensor_data):
        # Don't execute frame swap until nframes (set in config file) reached
        if curr_frame < nframes:
            self.wait_counter = curr_frame + 1
            self.meas_buf.append(measurements)
            self.sensor_buf.append(sensor_data)
            swapped_measurements = measurements
            swapped_sensor_data = sensor_data
        else: 
            # Randomly select an index of the buffer to return
            # Populate that index with the current measurement
            index = np.random.random_integers(0,nframes-1)
	    print "index:{}".format(index)
            swapped_measurements = self.meas_buf[index]
            swapped_sensor_data = self.sensor_buf[index]
            self.meas_buf[index] = measurements
            self.sensor_buf[index] = sensor_data
        return swapped_measurements, swapped_sensor_data

    def inject_adversarial(self, measurements, sensor_data):

        #logging.info("Measurement Values:")
        #self.log_measurements(measurements)

        #Inject noise into the player measurements
	self.player_inject(measurements.player_measurements)
        
	for a in measurements.non_player_agents:
            self.agent_inject(measurements.player_measurements, a.WhichOneof('agent'), a)
	
        #Delay attack modifies measurements and sensor_data to values from a previous frame
        if self.config['time']['delay_attack'] == 1:
            measurements, sensor_data = self.delay_attack(self.wait_counter, self.config['time']['nframes'], measurements, sensor_data) 

        #Frame swap attack swaps measurements and sensor_data of adjacent frames
        elif self.config['time']['frame_swap_attack'] == 1:
            measurements, sensor_data = self.frame_swap_attack(self.wait_counter, self.config['time']['nframes'], measurements, sensor_data)

        logging.info("Adversarial Measurement Values:")
        self.log_measurements(measurements)

	#sensor_data['CameraRGB'].raw_data = camera_attack.perform_attack(sensor_data['CameraRGB'])
	
	#self.wait_counter = self.wait_counter + 1
	print self.wait_counter
	if self.wait_counter >= 50:
		sensor_data['CameraRGB'].save_to_disk("/home/carla/Documents/carla_outputs/camera_outputs/fisheye.png")
		print("Done")
		time.sleep(5)
	return measurements, sensor_data
	

