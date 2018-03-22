

"""Generic Attack"""

import numpy as np
import copy

class GenericAttack(object):
    def __init__(self):

        return
    
    def final_scene_attack(self, final_scene):

        return

    def depth_map_attack(self, depth_map):

        return

    def semantic_segmentation_attack(self, semantic_segmentation):

        return

    def lidar_attack(self, lidar):

        return

    def measurement_attack(self, measurements):

        return 
    
    def perform_attack(self, measurements, sensor_data):

        self.final_scene_attack(sensor_data['CameraRGB'])
        #self.depth_map_attack(sensor_data['CameraDepth'])
        #self.semantic_segmentation_attack(sensor_data['CameraSemSeg'])
        #self.lidar_attack(sensor_data['lidar'])
        self.measurement_attack(measurements)

        return measurements, sensor_data
