import copy
"""
Utility functions to help with the way we represent parameters.
"""

def parameterstofloats(parameters, freeindex, freevalue):
    data = [float(parameter[0]) for parameter in parameters]
    data[freeindex] = freevalue
    return tuple(data)

def parameterstoconstants(parameters, freeindex, freevalue):
    data = [parameter[0] for parameter in parameters]
    data[freeindex].assign(freevalue)
    return tuple(data)

def nextparameters(values, freeindex, oldparams):
    """
    This is probably unnecessarily slow. It will index into
    values each time it is called to find the next parameter
    value. We could think about using a better data structure
    for values.
    """
    current_value = oldparams[freeindex]
    current_index = values.index(current_value)

    if current_index == len(values) - 1:
        # We've reached the last value, no more continuation to do.
        return None
    next_value = values[current_index + 1]

    newparams = list(oldparams)
    newparams[freeindex] = next_value
    return tuple(newparams)

def parameterstostring(parameters, values):
    s = ""
    for (param, value) in zip(parameters, values):
        s += "%s=%14.12e-" % (param[1], value)
    return s[:-1]
