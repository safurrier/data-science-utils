def param_string_to_kwargs_dict(multiline_param_string):
    """ From a multiline parameter string of the form:
    parameter
    value
    
    Return a kwargs dictionary
    
    E.g. 
    param_string = \"\"\"base_margin_initialize:
    True
    colsample_bylevel:
    1.0
    colsample_bytree:
    0.5\"\"\"
    
    kwargs_param_dict = param_string_to_kwargs_dict(param_string)
    kwargs_param_dict
    {'base_margin_initialize': True,
    'colsample_bylevel': 1.0,
    'colsample_bytree': 0.5,
    'interval': 10}
    """
    params = []
    param_vals = []
    for index, param in enumerate(multiline_param_string.split("\n")):
        if (index == 0) | (index % 2 == 0):
            params.append(param.replace(":", ""))
        else:
            # Get the python dtype of parameter value
            # Cast to Numeric
            try:
                param = int(param)
            except ValueError:
                try:
                    param = float(param)
                except ValueError:
                    pass
            # Check for booleans
            if param == 'True':
                param = True
            if param == 'False':
                param = False            

            param_vals.append(param)
    # Create the dictionary
    kwargs_params = dict(zip(params, param_vals))
    return kwargs_params