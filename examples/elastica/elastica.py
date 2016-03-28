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

                       --petsc.ksp_monitor_cancel
                       --petsc.ksp_type preonly

                       --petsc.inner_pc_type lu
                       """.split()
parameters.parse(args)

class ElasticaProblem(BifurcationProblem):
    def mesh(self, comm):
        return IntervalMesh(comm, 10000, 0, 1)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        lmbda = Constant(0)
        mu    = Constant(0)

        return [(lmbda, "lambda", "λ"),
                (mu,    "mu",     "μ")]

    def residual(self, theta, params, v):
        (lmbda, mu) = params

        F = (
              inner(grad(theta), grad(v))*dx
              - lmbda**2*sin(theta)*v*dx
              + mu*v*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return [DirichletBC(V, 0.0, "on_boundary")]

    def functionals(self):
        def signedL2(theta, params):
            j = sqrt(assemble(inner(theta, theta)*dx))
            g = project(grad(theta)[0], theta.function_space())
            return j*g
        tex = r"\theta'(0) \|\theta\|"

        return [(signedL2, "signedL2", tex)]

    def guesses(self, V, oldparams, oldstates, newparams):
        if oldparams is None:
            newguesses = [Function(V), Function(V)]
            newguesses[0].label = "initial-guess-0"
            newguesses[1].label = "initial-guess-0"
        else:
            newguesses = oldstates
            for (i, soln) in enumerate(oldstates):
                soln.label = "prev-soln-%d" % i

        return newguesses

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of lambda.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.
        (lmbda, mu) = map(float, params)
        n = int(floor((lmbda/pi)))*2
        return n + 1

    def inner_product(self, a, b):
        return inner(a, b)*dx + inner(grad(a), grad(b))*dx

if __name__ == "__main__":
    #io = FileIO("output")
    io = None

    dc = DeflatedContinuation(problem=ElasticaProblem(), io=io, teamsize=2)
    dc.run(free={"lambda": linspace(0, 4*pi, 100)}, fixed={"mu": 0.5})

    #dc.bifurcationdiagram("signedL2", fixed={"mu": 0.5})
    #plt.title(r"Buckling of an Euler elastica, $\mu = 1/2$")
    #plt.grid()
    #plt.savefig("bifurcation.pdf")

    # Maybe you could also do:
    #dc.run(fixed={"lambda": 4*pi}, free={"mu": linspace(0.5, 0.0, 6)})
    #dc.run(fixed={"mu": 0.0}, free={"lambda": linspace(4*pi, 0.0, 100)})

