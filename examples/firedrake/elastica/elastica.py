# -*- coding: utf-8 -*-
import sys
from   math import floor

from firedrake import *
from defcon import *

import matplotlib.pyplot as plt

class ElasticaProblem(BifurcationProblem):
    def mesh(self, comm):
        return IntervalMesh(1000, 0, 1, comm=comm)

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
        return [DirichletBC(V, 0.0, "on_boundary")]

    def functionals(self):
        def signedL2(theta, params):
            # Argh.
            j = sqrt(assemble(inner(theta, theta)*dx))
            g = project(grad(theta)[0], theta.function_space())
            return j*g((0.0,))

        return [(signedL2, "signedL2", r"$\theta'(0) \|\theta\|$")]

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

    def solver_parameters(self, params, task, **kwargs):
        return {
            "snes_max_it": 100,
            "snes_atol": 1.0e-9,
            "snes_rtol": 0.0,
            "snes_monitor": None,
            "snes_linesearch_type": "basic",
            "ksp_type": "preonly",
            "pc_type": "lu"
         }

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=ElasticaProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"lambda": linspace(0, 3.9*pi, 200), "mu": [0.5]}, freeparam="lambda")

    dc.bifurcation_diagram("signedL2", fixed={"mu": 0.5})
    plt.title(r"Buckling of an Euler elastica, $\mu = 1/2$")
    plt.savefig("bifurcation.pdf")

