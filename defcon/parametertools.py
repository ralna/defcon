class Parameters(object):
    """
    Takes in the parameters and freeindex. Has a bunch of utility
    functions to transform parameters in various ways.
    """

    def __init__(self, parameters, values, freeindex):
        self.parameters = parameters # the output from problem.parameters()
        self.values     = values
        self.constants  = [param[0] for param in parameters]
        self.labels     = [param[1] for param in parameters]
        self.freeindex  = freeindex

    def update_constants(self, values):
        for (val, const) in zip(values, self.constants):
            const.assign(val)

    def floats(self, value=None):
        out = map(float, self.constants)
        if value is not None:
            out[self.freeindex] = value
        return tuple(out)

    def next(self, oldparams):
        """
        This is probably unnecessarily slow. It will index into
        values each time it is called to find the next parameter
        value. We could think about using a better data structure
        for values.
        """
        current_value = oldparams[self.freeindex]
        current_index = self.values.index(current_value)

        if current_index == len(self.values) - 1:
            # We've reached the last value, no more continuation to do.
            return None
        next_value = self.values[current_index + 1]

        newparams = list(oldparams)
        newparams[self.freeindex] = next_value
        return tuple(newparams)

    def previous(self, oldparams):
        current_value = oldparams[self.freeindex]
        current_index = self.values.index(current_value)

        if current_index == 0:
            # We've reached the first value, no more continuation to do.
            return None
        prev_value = self.values[current_index - 1]

        newparams = list(oldparams)
        newparams[self.freeindex] = prev_value
        return tuple(newparams)

    def update_from_string(s):
        subs = s.split('@')
        assert len(subs) == len(self.parameters)

        for (sub, const) in zip(subs, self.constants):
            val = float(sub.split('=')[1])
            const.assign(val)

def make_parameters(parameters, values, free, fixed):
    freeparam = None
    freeindex = None
    for (index, param) in enumerate(parameters):
        label = param[1]
        const = param[0]
        if label in fixed:
            const.assign(fixed[label])

        if label in free:
            freeparam = param
            freeindex = index

    if freeparam is None:
        print("Cannot find %s in parameters %s." % (free.keys()[0], [param[1] for param in self.parameters]))
        assert freeparam is not None

    return Parameters(parameters, values, freeindex)

def parameters_to_string(parameters, values):
    s = ""
    for (param, value) in zip(parameters, values):
        s += "%s=%.15e@" % (param[1], value)
    return s[:-1]

