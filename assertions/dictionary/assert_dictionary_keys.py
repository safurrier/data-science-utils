def assert_dictionary_keys(dictionary, required_keys, verbose=0):
    missing_keys = []
    for key in required_keys:
        if key not in dictionary.keys():
            missing_keys.append(key)
    if missing_keys:
        if verbose > 0:
            print(
                f'Missing keys {missing_keys} not found in dictionary with keys {dictionary.keys()}')
        return False
    else:
        return True


def assert_nested_dictionary_keys(dictionary, nested_keys_dict, verbose=0):
    missing_keys = []
    missing_nested_keys = []
    for key, nested_keys in nested_keys_dict.items():
        # Check that all top level keys are in the dictionary
        if key not in dictionary.keys():
            missing_keys.append(key)
        for nested_key in nested_keys:
            if nested_key not in dictionary[key].keys():
                missing_nested_keys.append(nested_key)
    if missing_keys:
        if verbose > 0:
            print(
                f'Missing keys {missing_keys} not found in dictionary with keys {dictionary.keys()}')
        return False
    if missing_nested_keys:
        if verbose > 0:
            print(
                f'Nested keys: {missing_nested_keys} not found in dictionary.')
        return False
    else:
        return True
