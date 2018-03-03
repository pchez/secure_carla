from __future__ import print_function
import configparser
import sys
import os.path


def parse_config(config_file):

    config_params = {}
    
    # Initialize config parser object
    config = configparser.ConfigParser()

    if not os.path.isfile(config_file):
        print('Cannot find', config_file+'. ', 'Exiting now.')
        exit(1)
    else:
        # Read and parse config file
        config.read(config_file)
        
        # Read config params into dict
        config_params['sensors_default'] = dict(config.items('sensors_default'))
        config_params['sensors'] = dict(config.items('sensors'))
        config_params['camera_rgb'] = dict(config.items('camera_rgb'))

        # Convert string values to numeric
        config_params['sensors_default'] = {key:float(val) for key, val in config_params['sensors_default'].items()}
        config_params['sensors'] = {key:float(val) for key, val in config_params['sensors'].items()}
        config_params['camera_rgb'] = {key:float(val) for key, val in config_params['camera_rgb'].items()}
        
        # Print configuration parameters loaded
        print('Finished loading configs:')
        for key in config_params.keys():
            print('---',key,'---')
            print(config_params[key])

    return config_params

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print('No config file provided. Usage: python parse_config_attack.py <name_of_config_file>')
        exit(1)
    else:
        config_file = sys.argv[1]
        config_params = parse_config(config_file) 
