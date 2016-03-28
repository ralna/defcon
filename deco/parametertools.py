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
