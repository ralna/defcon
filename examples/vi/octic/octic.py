# -*- coding: utf-8 -*-
from defcon import *
from dolfin import *

"""
Minimize

J: R -> R
J(x) = (x^2 (1-x)^2)^2

subject to

x >= -0.8.

Has stationary points at -0.8, -1/root(2), +1/root(2), 1,
and a double root at 0.
"""

class OcticProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = UnitIntervalMesh(comm, 1)
        return mesh

    def function_space(self, mesh):
        R = FunctionSpace(mesh, "R", 0)
        return R

    def parameters(self):
        f = Constant(0)
        return [(f, "f", r"$f$")]

    def energy(self, x, params):
        E = (x**2 * (1 - x**2))**2 * dx
        return E

    def residual(self, x, params, v):
        E = self.energy(x, params)
        F = derivative(E, x, v)
        return F

    def boundary_conditions(self, Z, params):
        return []

    def functionals(self):
        def pointeval(x, params):
            g = x((0.5,))
            return g

        return [
                (pointeval, "pointeval", r"$x$"),
               ]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        return interpolate(Constant(+0.3), Z)

    def number_solutions(self, params):
        return float("inf")

    def solver_parameters(self, params, klass):
        return {
               "snes_max_it": 50,
               "snes_atol": 1.0e-14,
               "snes_rtol": 1.0e-14,
               "snes_monitor": None,
               "snes_linesearch_type": "basic",
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_damping": 1.0,
               "snes_linesearch_monitor": None,
               "ksp_type": "preonly",
               "ksp_monitor": None,
               "ksp_rtol": 1.0e-10,
               "ksp_atol": 1.0e-10,
               "pc_type": "lu",
               "pc_factor_mat_solver_package": "umfpack",
               }

    def bounds(self, R, params):
        l = interpolate(Constant(-0.8), R)
        u = interpolate(Constant(+2), R)
        return (l, u)

if __name__ == "__main__":
    problem = OcticProblem()
    deflation = ShiftedDeflation(problem, power=2, shift=1)
    dc = DeflatedContinuation(problem=problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True, profile=False)
    dc.run(values=dict(f=0))

    if backend.comm_world.rank == 0:
        print()
        params = (0,)
        branches = dc.thread.io.known_branches(params)
        for branch in branches:
            functionals = dc.thread.io.fetch_functionals([params], branch)[0]
            print("Solution: %s" % functionals)
