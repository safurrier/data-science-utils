import os
import pathlib

# Convert file directory to path
file_dir = 'path/to/file/dir'
file_dir = pathlib.Path(file_dir)

# Get file paths for directory and file
file_list = [file_dir / file 
             for file 
             in os.listdir(file_dir)]

# Get list of file paths with most recent first
files = sorted(file_list, key=os.path.getctime, reverse=True)