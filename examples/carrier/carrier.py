# -*- coding: utf-8 -*-
import sys
from   math import floor

from dolfin import *
from deco   import *

import matplotlib.pyplot as plt

args = [sys.argv[0]] + """
                       --petsc.snes_max_it 100
                       --petsc.snes_type newtonls
                       --petsc.snes_linesearch_type basic
                       --petsc.snes_stol 0.0
                       --petsc.snes_atol 1.0e-9
                       --petsc.snes_rtol 0.0
                       --petsc.snes_monitor
                       --petsc.snes_converged_reason
                       --petsc.snes_linesearch_monitor

                       --petsc.ksp_type preonly

                       --petsc.inner_pc_type lu
                       """.split()
parameters.parse(args)

class CarrierProblem(BifurcationProblem):
    def __init__(self):
        self.bcs = None

        # Awesome asymptotic formulae from Jon Chapman
        self.pitchbfs = [(0.472537/n)**2 for n in range(1,100)]
        self.foldbfs  = [0.08135344292708906, 0.029539186823838406, 0.015428615600424057, 0.00947979, 0.00647056, 0.00470338, 0.00357559, 0.00281121] + [(0.472537/(n + 0.04305/n))**2 for n in range(10, 100)]

    def mesh(self, comm):
        return IntervalMesh(comm, 10000, -1, 1)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        eps   = Constant(0)
        return [(eps, "epsilon", r"$\varepsilon$")]

    def residual(self, y, params, v):
        eps = params[0]
        x = SpatialCoordinate(y.function_space().mesh())[0]

        F = (
            - eps*inner(grad(y), grad(v))*dx
            + 2*(1-x*x) * inner(y, v)*dx
            + inner(y*y, v)*dx
            - inner(Constant(1), v)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        # The boundary conditions are independent of parameters, so only
        # evaluate them once for efficiency.
        if self.bcs is None:
            self.bcs = [DirichletBC(V, 0.0, "on_boundary")]
        return self.bcs

    def functionals(self):
        def signedL2(y, params):
            V = y.function_space()
            j = assemble(y*y*dx)
            g = project(grad(y)[0], V)((-1.0,))
            return j*g

        return [(signedL2, "signedL2", r"$y'(-1) \|y\|^2$")]

    def guesses(self, V, oldparams, oldstates, newparams):
        if oldparams is None:
            newguesses = [interpolate(Constant(1), V)]
        else:
            newguesses = oldstates

        return newguesses

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of eps.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.
        eps = params[0]

        nbifurcations = len([x for x in self.pitchbfs + self.foldbfs if x >= eps])
        return (nbifurcations+1)*2

    def inner_product(self, a, b):
        return inner(a, b)*dx + inner(grad(a), grad(b))*dx

if __name__ == "__main__":
    io = FileIO("output")
    dc = DeflatedContinuation(problem=CarrierProblem(), io=io, teamsize=1, verbose=True)
    dc.run(free={"epsilon": linspace(0.25, 0.20, 51)})

    dc.bifurcation_diagram("signedL2")
    plt.title(r"Solutions of the Carrier problem")
    plt.savefig("bifurcation.pdf")
