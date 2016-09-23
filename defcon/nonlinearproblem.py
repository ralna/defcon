import backend
if backend.__name__ == "dolfin":

    from backend import derivative, as_backend_type, Function, PETScMatrix, Form
    class GeneralProblem(object):
        def __init__(self, F, y, bcs, J=None, P=None):
            # Firedrake already calls the current Newton state u,
            # so let's mimic this for unity of interface
            self.u = y
            self.comm = y.function_space().mesh().mpi_comm()

            if J is None:
                self.J = derivative(F, y)
            else:
                self.J = J

            if P is None:
                self.P = None
            else:
                self.P = P

            self.F = F
            self.bcs = bcs
