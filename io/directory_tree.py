import pathlib

def display_directory_tree(directory: pathlib.Path):
    """Print a tree directory of the a given directory"""
    # Turn directory into a pathlib.Path object
    # if not already one
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')