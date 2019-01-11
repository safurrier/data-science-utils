import importlib
import pathlib

CONFIG_LOC = 'src/lab/load_data_config.py'

# Turn string locaiton into path
config_loc_path = pathlib.Path(CONFIG_LOC)

# Get path as positive slash divided string
# split it on '/' until the last part (the file)
# Add the stem of the file name (remove extension)
# Join on '.' so that it can be imported using importlib

import_string = '.'.join(config_loc_path.as_posix().split('/')[:-1]+[config_loc_path.stem])
config = importlib.import_module(import_string)

# To use as a dictionary
config = config.__dict__