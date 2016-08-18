# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

import matplotlib.pyplot as plt

args = [sys.argv[0]] + """
                       --petsc.snes_max_it 100
                       --petsc.snes_atol 1.0e-9
                       --petsc.snes_rtol 0.0
                       --petsc.snes_monitor

                       --petsc.ksp_type preonly
                       --petsc.pc_type lu
                       """.split()
parameters.parse(args)

class ElasticaProblem(BifurcationProblem):
    def __init__(self):
        self.bcs = None

    def mesh(self, comm):
        return IntervalMesh(comm, 1000, 0, 1)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        lmbda = Constant(0)
        mu    = Constant(0)

        return [(lmbda, "lambda", r"$\lambda$"),
                (mu,    "mu",     r"$\mu$")]

    def residual(self, theta, params, v):
        (lmbda, mu) = params

        F = (
              inner(grad(theta), grad(v))*dx
              - lmbda**2*sin(theta)*v*dx
              + mu*v*dx
            )

        return F

    def boundary_conditions(self, V, params):
        # The boundary conditions are independent of parameters, so only
        # evaluate them once for efficiency.
        if self.bcs is None:
            self.bcs = [DirichletBC(V, 0.0, "on_boundary")]
        return self.bcs

    def functionals(self):
        def signedL2(theta, params):
            # Argh.
            j = sqrt(assemble(inner(theta, theta)*dx))
            g = project(grad(theta)[0], theta.function_space())
            #return j*g((0.0,))
            return j

        def max(theta, params):
            return theta.vector().max()

        def min(theta, params):
            return theta.vector().min()

        return [(signedL2, "signedL2", r"$\theta'(0) \|\theta\|$"),
                (max, "max", r"$\max{\theta}$"),
                (min, "min", r"$\min{\theta}$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of lambda.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.
        if params[0] < 3.37: return 1

        (lmbda, mu) = params
        n = int(floor((lmbda/pi)))*2
        return n + 1

    def squared_norm(self, a, b, params):
        return inner(a - b, a - b)*dx + inner(grad(a - b), grad(a - b))*dx

if __name__ == "__main__":
    io = FileIO("output")
    dc = DeflatedContinuation(problem=ElasticaProblem(), io=io, teamsize=1, verbose=True)
    dc.run(free={"lambda": linspace(0, 3.9*pi, 200)}, fixed={"mu": 0.5})

    dc.bifurcation_diagram("signedL2", fixed={"mu": 0.5})
    plt.title(r"Buckling of an Euler elastica, $\mu = 1/2$")
    plt.savefig("bifurcation.pdf")

    # Maybe you could also do:
    #dc.run(fixed={"lambda": 4*pi}, free={"mu": linspace(0.5, 0.0, 6)})
    #dc.run(fixed={"mu": 0.0}, free={"lambda": linspace(4*pi, 0.0, 100)})

