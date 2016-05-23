# -*- coding: utf-8 -*-
import sys
from   math import degrees, atan2, pi, floor

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

class RootsOfUnityProblem(BifurcationProblem):
    def mesh(self, comm):
        return UnitIntervalMesh(comm, 2)

    def function_space(self, mesh):
        R = FunctionSpace(mesh, "R", 0)
        return MixedFunctionSpace([R, R])

    def parameters(self):
        p = Constant(0)

        return [(p, "p", r"$p$")]

    def residual(self, z, params, v):
        p = params[0]

        (s, d) = split(v)
        (real, imag) = split(z)

        r = sqrt(real**2 + imag**2)
        theta = atan_2(imag, real)

        F = (
              inner(exp(p*ln(r)) * cos(p*theta) - 1, s)*dx
            + inner(exp(p*ln(r)) * sin(p*theta), d)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return []

    def functionals(self):
        def arg(z, params):
            j = degrees(atan2(-z.vector()[1], -z.vector()[0]) + pi)
            return j

        return [(arg, "arg", r"$\mathrm{arg}(z)$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant((-0.9, 0)), V)

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of lambda.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.
        p = params[0]
        return int(floor(p/2.0))*2

    def trivial_solutions(self, V):
        return [interpolate(Constant((1, 0)), V)]

if __name__ == "__main__":
    io = FileIO("output")
    dc = DeflatedContinuation(problem=RootsOfUnityProblem(), io=io, teamsize=1, verbose=True)
    dc.run(free={"p": linspace(2.0, 9.0, 21)})

    dc.bifurcation_diagram("arg")
    plt.title(r"Bifurcation diagram for the roots of unity")
    plt.savefig("bifurcation.pdf")
