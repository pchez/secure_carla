

"""Generic Defense"""

import numpy as np
import copy

class GenericDefense(object):
    def __init__(self):

        return
    
    def final_scene_defense(self, final_scene):

        return

    def depth_map_defense(self, depth_map):

        return

    def semantic_segmentation_defense(self, semantic_segmentation):

        return

    def lidar_defense(self, lidar):

        return

    def measurement_defense(self, measurements):

        return 
    
    def perform_defense(self, measurements, sensor_data):

        self.final_scene_defense(sensor_data['CameraRGB'])
        #self.depth_map_defense(sensor_data['CameraDepth'])
        #self.semantic_segmentation_defense(sensor_data['CameraSemSeg'])
        #self.lidar_defense(sensor_data['lidar'])
        self.measurement_defense(measurements)

        return measurements, sensor_data
