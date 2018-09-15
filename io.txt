def file_search_and_replace(directory, search, replace, verbose=True):
    """Given a directory, search all filenames in it for the regex
    pattern provided. If found, replace with the provided string 
    by renaming.
    Set verbose=True to see which files are renamed"""
    from pathlib import Path
    import re
    # Make path out of provided directory
    directory_path = Path(directory)
    # Search directory fielnames
    for filename in os.listdir(directory_path):
        # If there's a pattern match
        if re.search(search, filename):
            # Create a new filename replacing the old pattern
            new_fname = re.sub(search, replace, filename)
            # Rename it
            os.rename(directory_path / filename, directory_path / new_fname)
            # If verbose print the renamed files
            if verbose:
                print(f'Rename:\n{directory_path / filename}\nTo:\n{directory_path / new_fname}\n\n')