import ruamel.yaml as yaml
import os
import sys

# Change target_fname to a file in the root dir
# If default target_fname='README.md' works, then simply:


import warnings
import os

def get_file_absolute_path(target_fname: str='README.md', levels_to_check: int=10, verbose=0):
    """Pass a filename that exists in a directory an unknown number of 
    levels higher
    """
    original_wd = os.getcwd()
    for x in range(0, levels_to_check):
        # If reached the max number of directory levels change to original wd and print message
        if x + 1 == levels_to_check:
            os.chdir(original_wd)
            if verbose:
                warnings.warn(f"""\n\nUnable to find directory with file {target_fname} within {levels_to_check} parent directories""")  
            return
        # Check if README exists
        #cwd_files = 
        if os.path.isfile(target_fname):
            target_dir = os.getcwd()
            if verbose:
                print(f'Found target file in {target_dir}')
            return target_dir
        # If not found move back one directory level        
        else:
            os.chdir('../')   
    
os.chdir(get_fpath_absolute_path())

# Add directory to PATH
path = os.getcwd()

if path not in sys.path:
    sys.path.append(path)
