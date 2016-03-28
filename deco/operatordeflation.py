class DeflationOperator(object):
    pass

class ShiftedDeflation(object):
    def __init__(self, problem, power, shift):
        self.problem = problem
        self.power = power
        self.shift = shift
