import backend
from backend import Function, Vector, assemble, derivative, product

# Make a zero vector from a model, including parallel layout
if backend.__name__ == "dolfin":
    def empty_vector(model):
      b = Vector(model)
      b.zero()
      return b

elif backend.__name__ == "firedrake":
    def empty_vector(model):
      b = model.copy()
      b.dat.zero()
      return b

class DeflationOperator(object):
    """
    Base class for deflation operators.
    """
    def deflate(self, roots):
        self.roots = roots

    def evaluate(self):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError

class ShiftedDeflation(DeflationOperator):
    """
    The shifted deflation operator presented in doi:10.1137/140984798.
    """
    def __init__(self, problem, parameters, power, shift):
        self.problem = problem
        self.parameters = parameters
        self.power = power
        self.shift = shift
        self.roots = []

    def normsq(self, y, root):
        return self.problem.squared_norm(y, root, self.parameters)

    def evaluate(self, y):
        m = 1.0
        for normsq in [assemble(self.normsq(y, root)) for root in self.roots]:
            factor = normsq**(-self.power/2.0) + self.shift
            m *= factor

        return m

    def derivative(self, y):
        if len(self.roots) == 0:
            deta = empty_vector(y.vector())
            return deta

        p = self.power
        factors  = []
        dfactors = []
        dnormsqs = []
        normsqs  = []

        for root in self.roots:
            form = self.normsq(y, root)
            normsqs.append(assemble(form))
            dnormsqs.append(assemble(derivative(form, y)))

        for normsq in normsqs:
            factor = normsq**(-p/2.0) + self.shift
            dfactor = (-p/2.0) * normsq**((-p/2.0) - 1.0)

            factors.append(factor)
            dfactors.append(dfactor)

        eta = product(factors)

        deta = empty_vector(y.vector())

        for (solution, factor, dfactor, dnormsq) in zip(self.roots, factors, dfactors, dnormsqs):
            if backend.__name__ == "firedrake":
                dnormsq = dnormsq.vector()
            deta.axpy(float((eta/factor)*dfactor), dnormsq)

        return deta

