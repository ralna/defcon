# -*- coding: utf-8 -*-
import sys
from   math import floor

from petsc4py import PETSc
from firedrake import *
from defcon import *

import matplotlib.pyplot as plt

params = {
    "snes_max_it": 100,
    "snes_atol": 1.0e-9,
    "snes_rtol": 0.0,
    "snes_monitor": None,
    "snes_linesearch_type": "basic",
    "mat_type": "aij",
    "ksp_type": "preonly",
    "ksp_converged_reason": None,
    "ksp_monitor": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_package": "mumps"
}

options = PETSc.Options()
for k in params:
    options[k] = params[k]

class RayleighBenardProblem(BifurcationProblem):
    def __init__(self):
        self.bcs = None

    def mesh(self, comm):
        return RectangleMesh(50, 10, 5.0, 1.0, comm=comm)

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", 2)
        W = FunctionSpace(mesh, "CG", 1)
        T = FunctionSpace(mesh, "CG", 1)
        return MixedFunctionSpace([V, W, T])

    def parameters(self):
        Ra = Constant(1.0)
        Pr = Constant(6.8)

        return [(Ra, "Ra", r"$Ra$"),
                (Pr, "Pr", r"$Pr$")]

    def residual(self, theta, params, test):
        u, p, T = split(theta)
        (Ra, Pr) = params
        v, q, S = split(test)
        g = Constant((0, -1))

        F = (
            inner(grad(u), grad(v))*dx
            + inner(dot(grad(u), u), v)*dx
            - inner(p, div(v))*dx
            - Ra*Pr*inner(T*g, v)*dx
            + 1.e-6 * p * q * dx
            + inner(div(u), q)*dx
            + inner(dot(grad(T), u), S)*dx
            + 1/Pr * inner(grad(T), grad(S))*dx
        )

        return F

    def boundary_conditions(self, Z, params):
        # The boundary conditions are independent of parameters, so only
        # evaluate them once for efficiency.
        if self.bcs is None:
            self.bcs = [
                DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3, 4)),
                DirichletBC(Z.sub(2), Constant(1.0), (1,)),
                DirichletBC(Z.sub(2), Constant(0.0), (2,))
            ]
            
        return self.bcs

    def functionals(self):
        def sqL2(theta, params):
            (u, p, T) = split(theta)
            return assemble(inner(u, u)*dx)

        return [(sqL2, "sqL2", r"$\|u\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def squared_norm(self, a, b, params):
        u1, p1, T1 = split(a)
        u2, p2, T2 = split(b)
        return inner(u1-u2, u1-u2)*dx + inner(T1-T2, T1-T2)*dx


if __name__ == "__main__":
    io = FileIO("output")
    dc = DeflatedContinuation(problem=RayleighBenardProblem(), io=io, teamsize=1, verbose=True)
    dc.run(free={"Ra": linspace(1700, 1780, 1.0)}, fixed={"Pr": 6.8})

    dc.bifurcation_diagram("sqL2", fixed={"Pr": 6.8})
    plt.title(r"Rayleigh-Benard convection, $Pr = 6.8$")
    plt.savefig("bifurcation.pdf")

    # Maybe you could also do:
    #dc.run(fixed={"lambda": 4*pi}, free={"mu": linspace(0.5, 0.0, 6)})
    #dc.run(fixed={"mu": 0.0}, free={"lambda": linspace(4*pi, 0.0, 100)})

