import backend
from petsc4py import PETSc
if backend.__name__ == "dolfin":
    from backend import as_backend_type, PETScVector, PETScMatrix

    # dolfin lacks a high-level snes frontend like Firedrake,
    # so we're going to put one here and build up what we need
    # to make things happen.
    class SNUFLSolver(object):
        def __init__(self, problem, prefix="", **kwargs):
            self.problem = problem
            u = problem.u
            self.u_dvec = as_backend_type(u.vector())
            self.u_pvec = self.u_dvec.vec()

            comm = u.function_space().mesh().mpi_comm()
            snes = PETSc.SNES().create(comm=comm)
            snes.setOptionsPrefix(prefix)

            # Fix what must be one of the worst defaults in PETSc
            opts = PETSc.Options()
            if (prefix + "snes_linesearch_type") not in opts:
                opts[prefix + "snes_linesearch_type"] = "basic"

            self.b = problem.init_residual()
            snes.setFunction(self.residual, self.b.vec())
            self.A = problem.init_jacobian()
            self.P = problem.init_preconditioner(self.A)
            snes.setJacobian(self.jacobian, self.A.mat(), self.P.mat())
            snes.ksp.setOperators(self.A.mat(), self.P.mat()) # why isn't this done in setJacobian?

            snes.setFromOptions()

            self.snes = snes

        def update_x(self, x):
            """Given a PETSc Vec x, update the storage of our
               solution function u."""

            x.copy(self.u_pvec)
            self.u_dvec.update_ghost_values()

        def residual(self, snes, x, b):
            self.update_x(x)
            b_wrap = PETScVector(b)
            self.problem.assemble_residual(b_wrap, self.u_dvec)

        def jacobian(self, snes, x, A, P):
            self.update_x(x)
            A_wrap = PETScMatrix(A)
            P_wrap = PETScMatrix(P)
            self.problem.assemble_jacobian(A_wrap)
            self.problem.assemble_preconditioner(A_wrap, P_wrap)

        def solve(self):
            # Need a copy for line searches etc. to work correctly.
            x = self.problem.u.copy(deepcopy=True)
            self.snes.solve(None, as_backend_type(x.vector()).vec())

