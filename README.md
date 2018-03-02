# secure_carla
Secure CARLA

1. Config:
    - Dependency: configparser module.
        pip install configparser
    - Modify config_attack.ini to add/delete parameters and groups as needed
    - python parse_config_attack.py config_attack.ini will load groups into config_params, a dict of dicts,      for example: config_params['sensors_default'] = {'accel_mean':0, 'accel_var':1, accel_offset', ...}
