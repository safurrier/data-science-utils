import os
def get_absolute_fpath(target_fname: str = 'README.md', levels_to_check: int = 10, verbose=0):
    """Pass a filename that exists in a directory an unknown number of
    levels higher. Return the string absolute path of the file
    """
    original_wd = os.getcwd()
    for x in range(0, levels_to_check):
        # If reached the max number of directory levels change to original wd and print message
        if x + 1 == levels_to_check:
            os.chdir(original_wd)
            if verbose:
                warnings.warn(
                    f"""\n\nUnable to find directory with file {target_fname} within {levels_to_check} parent directories""")
            return
        # Check if README exists
        # cwd_files =
        if os.path.isfile(target_fname):
            target_dir = os.getcwd()
            if verbose:
                print(f'Found target file in {target_dir}')
            return target_dir
        # If not found move back one directory level
        else:
            os.chdir('../')