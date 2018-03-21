#Structure for SecureCarla class to log true and adversarial measurements

import numpy as np

class SCSensor(object):
    def __init__(self, config_file=None):
        self.detected_by_sensor = np.nan
        self.true_distance = np.nan
        self.noise_distances = np.nan #list with length num_sensors
        self.adversarial_distances = np.nan #list with length num_sensors
        self.true_speed = np.nan 
        self.noise_speed = np.nan #list with length num_sensors?
        self.adversarial_speed = np.nan #list with length num_sensors?
        self.accel_x = np.nan
        self.accel_y = np.nan
        self.accel_y = np.nan
        self.pitch = np.nan
        self.yaw = np.nan
        self.roll = np.nan
    
    def setDistanceSensor(self, which_sensor):
        self.detected_by_sensor = which_sensor

    def updateDistances(self, true_dist, noise_dist, adv_dist):
        self.true_distance = true_dist
        self.noise_distances = noise_dist
        self.adversarial_distances = adv_dist
        
    def updateSpeeds(self, true_speed, noise_speed, adv_speed):
        self.true_speed = true_speed
        self.noise_speed = noise_speed
        self.adverarial_speed = adv_speed

    def updateAccel(self, x, y, z):
        self.accel_x = x
        self.accel_y = y
        self.accel_y = z

    def updateOrientation(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        
