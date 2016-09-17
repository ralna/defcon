import backend
if backend.__name__ == "dolfin":

    from backend import derivative, SystemAssembler, as_backend_type, Function, PETScMatrix, Form
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

            self.ass = SystemAssembler(self.J, F, bcs)
            if self.P is not None:
                self.Pass = SystemAssembler(self.P, F, bcs)

        def init_jacobian(self):
            A = PETScMatrix(self.comm)
            self.ass.init_global_tensor(A, Form(self.J))
            return A

        def init_residual(self):
            b = as_backend_type(Function(self.u.function_space()).vector())
            return b

        def init_preconditioner(self, A):
            if self.P is None: return A
            P = PETscMatrix(self.comm)
            self.Pass.init_global_tensor(P, Form(self.P))
            return P

        def assemble_residual(self, b, x):
            self.ass.assemble(b, x)

        def assemble_jacobian(self, A):
            self.ass.assemble(A)

        def assemble_preconditioner(self, A, P):
            if self.P is None: return A
            self.Pass.assemble(P)
