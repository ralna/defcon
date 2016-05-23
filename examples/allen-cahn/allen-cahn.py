# -*- coding: utf-8 -*-
import sys
from   math import floor

from dolfin import *
from deco   import *

import matplotlib.pyplot as plt

args = [sys.argv[0]] + """
                       --petsc.snes_max_it 50
                       --petsc.snes_atol 1.0e-9
                       --petsc.snes_rtol 0.0
                       --petsc.snes_monitor

                       --petsc.ksp_type preonly
                       --petsc.pc_type lu
                       """.split()
parameters.parse(args)

class AllenCahnProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitSquareMesh(comm, 100, 100, "crossed")

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
        bcs = [DirichletBC(V, +1.0, "x[0] == 0.0 || x[0] == 1"),
               DirichletBC(V, -1.0, "x[1] == 0.0 || x[1] == 1")]
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
    io = FileIO("output")
    dc = DeflatedContinuation(problem=AllenCahnProblem(), io=io, teamsize=1, verbose=True)
    dc.run(free={"delta": [0.04]})
