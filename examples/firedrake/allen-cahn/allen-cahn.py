# -*- coding: utf-8 -*-
from petsc4py import PETSc
from firedrake import *
from defcon import *

params = {
    "snes_max_it": 100,
    "snes_atol": 1.0e-9,
    "snes_rtol": 0.0,
    "snes_monitor": None,
    "snes_linesearch_type": "basic",
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_package": "mumps"
}

options = PETSc.Options()
for k in params:
    options[k] = params[k]

class AllenCahnProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitSquareMesh(100, 100, "crossed", comm=comm)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        delta = Constant(0)
        return [(delta, "delta", r"$\delta$")]

    def residual(self, y, params, v):
        delta = params[0]

        F = (
            + delta * inner(grad(v), grad(y))*dx
            + 1.0/delta * inner(v, y**3 - y)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        bcs = [DirichletBC(V, +1.0, (1, 2)),
               DirichletBC(V, -1.0, (3, 4))]
        return bcs

    def functionals(self):
        def sqL2(y, params):
            j = assemble(y*y*dx)
            return j

        return [(sqL2, "sqL2", r"$\|y\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        delta = params[0]
        if delta == 0.04: return 3
        else:             return float("inf")

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=AllenCahnProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"delta": [0.04]})
