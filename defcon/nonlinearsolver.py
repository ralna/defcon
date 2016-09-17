import backend
if backend.__name__ == "dolfin":
    from backend import PETScSNESSolver, as_backend_type

    # dolfin lacks a high-level snes frontend like Firedrake,
    # so we're going to put one here and build up what we need
    # to make things happen.
    class SNUFLSolver(object):
        def __init__(self, problem, prefix="", **kwargs):
            y = problem.u
            comm = y.function_space().mesh().mpi_comm()
            low_level_solver = PETScSNESSolver(comm)
            low_level_solver.init(problem, y.vector())

            snes = low_level_solver.snes()
            snes.setOptionsPrefix(prefix)

            self.problem = problem
            self.low_level_solver = low_level_solver
            self.snes = snes

        def solve(self):
            # Need a copy for line searches etc. to work correctly.
            x = self.problem.u.copy(deepcopy=True)
            self.snes.solve(None, as_backend_type(x.vector()).vec())

