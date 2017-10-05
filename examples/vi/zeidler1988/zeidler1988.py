# -*- coding: utf-8 -*-
# pg 320, Zeidler (1988)
from defcon import *
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import os

alpha = 0.75
INF = 1e20
lb = Constant((-alpha, -INF, -INF, -INF, -INF))
ub = Constant((+alpha, +INF, +INF, +INF, +INF))

class ZeidlerProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = UnitIntervalMesh(comm, 500)
        self.mesh = mesh
        return mesh

    def function_space(self, mesh):
        Z = VectorFunctionSpace(mesh, "CG", 1, dim=5)
        return Z

    def parameters(self):
        P = Constant(0)
        a = Constant(0)
        g = Constant(0)
        rho = Constant(0)
        return [(P, "P", r"$P$"),
                (a, "a", r"$a$"),
                (g, "g", r"$g$"),
                (rho, "rho", r"$\rho$"),
                ]

    def energy(self, z, params):
        (u, v, w, l1, l2) = split(z)
        (P, a, g, rho) = params
        x = SpatialCoordinate(self.mesh)[0]
        E = (
            + 0.5 * a * inner(w, w)*dx
            - P*inner(v, v)*dx
            - rho*g*u*dx
            )
        return E

    def lagrangian(self, z, params):
        (u, v, w, l1, l2) = split(z)
        L = (
              self.energy(z, params)
            - inner(l1, v - u.dx(0))*dx
            - inner(l2, w - v.dx(0))*dx
            )
        return L

    def residual(self, z, params, v):
        L = self.lagrangian(z, params)
        F = derivative(L, z, v)
        return F

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(0), 0, "on_boundary")]

    def functionals(self):
        def signedenergy(z, params):
            j = assemble(self.energy(z, params))
            g = z((0.5,))[0]
            return j*g

        def energy(z, params):
            j = assemble(self.energy(z, params))
            return j

        return [
                (signedenergy, "signedenergy", r"$u(\frac{1}{2}) E(z)$"),
                (energy, "energy", r"$E(z)$"),
               ]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        return Function(Z)

    def number_solutions(self, params):
        return float("inf")

    def solver_parameters(self, params, klass):
        return {
               "snes_max_it": 10000,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "snes_linesearch_type": "basic",
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_damping": 0.01,
               "snes_linesearch_monitor": None,
               "ksp_type": "preonly",
               "ksp_monitor": None,
               "ksp_rtol": 1.0e-10,
               "ksp_atol": 1.0e-10,
               "pc_type": "lu",
               "pc_factor_mat_solver_package": "umfpack",
               }

    def render(self, params, branchid, solution):
        try:
            os.makedirs('output/figures/%2.6f' % (params[0],))
        except:
            pass

        s = np.linspace(0, 1, 1000)
        u = [solution((s_,))[0] for s_ in s]
        plt.clf()
        plt.plot(s, u, '-b', linewidth=2)
        plt.grid()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$u(x)$')
        plt.ylim([-1, 1])
        plt.title(r'$P = %.3f$' % params[0])
        plt.savefig('output/figures/%2.6f/branchid-%d.png' % (params[0], branchid))

    def postprocess(self, solution, params, branchid, window):
        self.render(params, branchid, solution)
        plt.show()

if __name__ == "__main__":
    eqproblem = ZeidlerProblem()
    viproblem = VIBifurcationProblem(eqproblem, lb, ub)

    dc = DeflatedContinuation(problem=viproblem, teamsize=1, verbose=True, clear_output=True, profile=False)
    #dc.run(values=dict(P=linspace(0, 10, 201), g=-1, a=1, rho=1), freeparam="P")
    dc.run(values=dict(P=5.2, g=-1, a=1, rho=1))
