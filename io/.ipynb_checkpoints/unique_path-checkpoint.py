import pathlib

def unique_path(directory, name_pattern):
    """Given a pattern (e.g.'test{:03d}.txt') and a directory return
    a pathlib.Path object with a unique file name"""
    # Turn directory into a pathlib.Path object
    # if not already one
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)    
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        # If the file doesn't exist, return this path
        if not path.exists():
            return path