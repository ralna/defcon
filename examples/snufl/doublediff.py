# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

args = [sys.argv[0]] + """
--petsc.snes_type ksponly
--petsc.ksp_type gmres
--petsc.pc_type fieldsplit
--petsc.pc_fieldsplit_type additive
--petsc.fieldsplit_0_fields 0
--petsc.fieldsplit_1_fields 1
--petsc.fieldsplit_0_ksp_type preonly
--petsc.fieldsplit_0_pc_type lu
--petsc.fieldsplit_1_ksp_type preonly
--petsc.fieldsplit_1_pc_type lu
--petsc.ksp_monitor
""".split()
parameters.parse(args)

mesh = UnitSquareMesh(2, 2)

E1 = FiniteElement("CG", triangle, 1)
E2 = FiniteElement("CG", triangle, 2)

Z  = FunctionSpace(mesh, E1*E2)

z = Function(Z)
w = TestFunction(Z)

u1, u2 = split(z)
v1, v2 = split(w)

f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)

F = (
    inner(grad(u1), grad(v1))*dx + u1*v1*dx
    + inner(grad(u2), grad(v2))*dx + u2*v2*dx
    -f*v1*dx - f*v2*dx
    )

bcs = None


prob = nonlinearproblem.GeneralProblem(F, z, bcs)
solver = nonlinearsolver.SNUFLSolver(prob)

solver.solve()
