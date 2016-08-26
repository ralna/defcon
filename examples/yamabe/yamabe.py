# -*- coding: utf-8 -*-
import sys

from defcon import *
from dolfin import *

args = [sys.argv[0]] + """
                       --petsc.snes_max_it 100
                       --petsc.snes_stol 0.0
                       --petsc.snes_rtol 1.0e-10
                       --petsc.snes_trtol 0.0
                       --petsc.snes_monitor

                       --petsc.ksp_monitor_cancel
                       --petsc.ksp_type gmres
                       --petsc.ksp_gmres_restart 100
                       --petsc.ksp_monitor_short
                       --petsc.ksp_max_it 1000
                       --petsc.ksp_atol 1.0e-10
                       --petsc.ksp_rtol 1.0e-10

                       --petsc.pc_type gamg
                       --petsc.pc_gamg_verbose 10
                       --petsc.pc_gamg_type agg
                       --petsc.pc_gamg_coarse_eq_limit 2000
                       --petsc.pc_gamg_agg_nsmooths 4
                       --petsc.pc_gamg_threshold 0.04
                       --petsc.pc_gamg_square_graph 1
                       --petsc.pc_gamg_sym_graph 1
                       --petsc.mg_coarse_pc_type redundant
                       --petsc.mg_coarse_sub_pc_type lu
                       --petsc.mg_levels_pc_type jacobi
                       --petsc.mg_levels_ksp_type chebyshev
                       --petsc.mg_levels_ksp_max_it 10
                       --petsc.gamg_est_ksp_max_it 30
                       """.split()
parameters.parse(args)
parameters["form_compiler"]["quadrature_degree"] = 5
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "quadrature"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

class YamabeProblem(BifurcationProblem):
    def mesh(self, comm):
        return Mesh(comm, "mesh/doughnut.xml")

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        a = Constant(0)
        return [(a, "a", r"$a$")]

    def residual(self, y, params, v):
        a = params[0]

        x = SpatialCoordinate(y.function_space().mesh())
        r = sqrt(x[0]**2 + x[1]**2)
        rho = 1.0/r**3

        F = (
             a*inner(grad(y), grad(v))*dx +
             rho * inner(y**5, v)*dx +
             Constant(-1.0/10.0)*inner(y, v)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        bcs = [DirichletBC(V, Constant(1.0), "on_boundary")]
        return bcs

    def functionals(self):
        def sqL2(y, params):
            j = assemble(y*y*dx)
            return j

        return [(sqL2, "sqL2", r"$\|y\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant(1), V)

if __name__ == "__main__":
    problem = YamabeProblem()
    deflation = ShiftedDeflation(problem, power=1, shift=1.0e-2)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, verbose=True)
    dc.run(free={"a": [8.0]})
