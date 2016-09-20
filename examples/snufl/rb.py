# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

#import matplotlib.pyplot as plt


# 
# args = [sys.argv[0]] + """
#                        --petsc.snes_max_it 100
#                        --petsc.snes_atol 1.0e-9
#                        --petsc.snes_rtol 0.0
#                        --petsc.snes_monitor
#                        --petsc.snes_converged_reason

#                        --petsc.ksp_type preonly
#                        --petsc.pc_type lu
#                        --petsc.pc_factor_mat_solver_package mumps
#                        """.split()
# parameters.parse(args)

args = [sys.argv[0]] + """
                      --petsc.snes_max_it 100
                      --petsc.snes_atol 1.0e-9
                      --petsc.snes_rtol 0.0
                      --petsc.snes_monitor
                      --petsc.snes_converged_reason

                      --petsc.ksp_type fgmres
                      --petsc.ksp_converged_reason
                      --petsc.ksp_monitor
                      --petsc.ksp_gmres_restart 200
                      --petsc.pc_type fieldsplit
                      --petsc.pc_fieldsplit_type schur
                      --petsc.pc_fieldsplit_0_fields 0,1
                      --petsc.pc_fieldsplit_1_fields 2
                      --petsc.fieldsplit_0_ksp_type preonly
                      --petsc.fieldsplit_0_pc_type lu
                      --petsc.fieldsplit_0_pc_factor_mat_solver_package mumps
                      --petsc.fieldsplit_1_ksp_type preonly
                      --petsc.fieldsplit_1_pc_type lu
                      """.split()
parameters.parse(args)

mesh = RectangleMesh(Point(0, 0), Point(5, 1), 50, 50)

Ve = VectorElement("CG", triangle, 2)
Qe = FiniteElement("CG", triangle, 1)
Te = FiniteElement("CG", triangle, 1)
Ze = MixedElement([Ve, Qe, Te])
Z  = FunctionSpace(mesh, Ze)

Ra = Constant(1705)
Pr = Constant(6.8)

z = Function(Z)
w = TestFunction(Z)

(u, p, T) = split(z)
(v, q, S) = split(w)

g = as_vector([0, 1])

F = (
    inner(grad(u), grad(v))*dx
    + inner(dot(grad(u), u), v)*dx
    - inner(p, div(v))*dx
    - Ra*Pr*inner(T*g, v)*dx
    + inner(div(u), q)*dx
    + inner(dot(grad(T), u), S)*dx
    + 1/Pr * inner(grad(T), grad(S))*dx
)

bcs = [
    DirichletBC(Z.sub(0), (0, 0), "on_boundary"),
    DirichletBC(Z.sub(2), 1, "near(x[1], 0.0)"),
    DirichletBC(Z.sub(2), 0, "near(x[1], 1.0)"),
    DirichletBC(Z.sub(1), 0, "x[0] == 0.0 && x[1] == 0.0", "pointwise")
]

prob = nonlinearproblem.GeneralProblem(F, z, bcs)
solver = nonlinearsolver.SNUFLSolver(prob)

solver.solve()
