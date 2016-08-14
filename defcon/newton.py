from dolfin import *
from petsc4py import PETSc
import sys
from numpy import isnan

# I can't believe this isn't in DOLFIN.
class GeneralProblem(NonlinearProblem):
    def __init__(self, F, y, bcs):
        NonlinearProblem.__init__(self)
        J = derivative(F, y)
        self.ass = SystemAssembler(J, F, bcs)

    def F(self, b, x):
        self.ass.assemble(b, x)

    def J(self, A, x):
        self.ass.assemble(A)

class DeflatedKSP(object):
    def __init__(self, deflation, y, ksp):
        self.deflation = deflation
        self.y = y
        self.ksp = ksp

    def solve(self, ksp, b, dy_pet):
        # Use the inner ksp to solve the original problem
        self.ksp.solve(b, dy_pet)
        deflation = self.deflation

        if deflation is not None:
            dy_vec = PETScVector(dy_pet)

            Edy = -deflation.derivative(self.y).inner(dy_vec)
            minv = 1.0 / deflation.evaluate(self.y)
            tau = (1 + minv*Edy/(1 - minv*Edy))
            dy_pet.scale(tau)

        ksp.setConvergedReason(self.ksp.getConvergedReason())

def newton(F, y, bcs, deflation=None, prefix="", snes_setup=None):

    comm = y.function_space().mesh().mpi_comm()
    solver = PETScSNESSolver(comm)
    snes = solver.snes()
    problem = GeneralProblem(F, y, bcs)

    snes.setOptionsPrefix(prefix)
    PETScOptions.set(prefix + "snes_monitor_cancel")
    solver.init(problem, y.vector())

    if snes_setup is not None:
        snes_setup(snes)

    oldksp = snes.ksp
    defksp = DeflatedKSP(deflation, y, oldksp)
    snes.ksp = PETSc.KSP().createPython(defksp, comm)
    snes.ksp.pc.setType('none')
    snes.ksp.setOperators(*oldksp.getOperators())
    snes.ksp.setUp()

    # Need a copy for line searches etc. to work correctly.
    x = y.copy(deepcopy=True)
    snes.solve(None, as_backend_type(x.vector()).vec())

    success = snes.getConvergedReason() > 0
    return success

