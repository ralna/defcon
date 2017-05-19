from __future__ import absolute_import

from petsc4py import PETSc
from numpy import isnan

import sys

import defcon.backend as backend


# Unfortunately DOLFIN and firedrake are completely different in how they
# do the solve. So we have to branch based on backend here.

if backend.__name__ == "dolfin":
    from defcon.backend import PETScOptions, PETScVector

    def getEdy(deflation, y, dy):
        dy_vec = PETScVector(dy)
        return -deflation.derivative(y).inner(dy_vec)

    def setSnesMonitor(prefix):
        PETScOptions.set(prefix + "snes_monitor_cancel")
        PETScOptions.set(prefix + "snes_monitor")

elif backend.__name__ == "firedrake":
    from defcon.backend import NonlinearVariationalSolver

    def getEdy(deflation, y, dy):
        with deflation.derivative(y).dat.vec_ro as deriv:
            Edy = -deriv.dot(dy)
        return Edy

    def setSnesMonitor(prefix):
        from defcon.backend.petsc import PETSc
        opts = PETSc.Options()
        opts.setValue(prefix + "snes_monitor_cancel", "")
        opts.setValue(prefix + "snes_monitor", "")

else:
    raise ImportError("Unknown backend")

if hasattr(backend, 'ConvergenceError'):
    from defcon.backend import ConvergenceError
else:
    class ConvergenceError(Exception):
        pass

def compute_tau(deflation, state, update_p):
    if deflation is not None:
        Edy = getEdy(deflation, state, update_p)

        minv = 1.0 / deflation.evaluate(state)
        tau = (1 + minv*Edy/(1 - minv*Edy))
        return tau
    else:
        return 1

class DeflatedKSP(object):
    def __init__(self, deflation, y, ksp):
        self.deflation = deflation
        self.y = y
        self.ksp = ksp

    def solve(self, ksp, b, dy_pet):
        # Use the inner ksp to solve the original problem
        self.ksp.solve(b, dy_pet)
        deflation = self.deflation

        tau = compute_tau(deflation, self.y, dy_pet)
        dy_pet.scale(tau)

        ksp.setConvergedReason(self.ksp.getConvergedReason())

    def setUp(self, ksp):
        ksp.setOperators(*self.ksp.getOperators())

    def view(self, ksp, viewer):
        self.ksp.view(viewer)

def newton(F, J, y, bcs, problemclass, solverclass, solver_params,
           teamno, deflation=None, dm=None, prefix=""):
    comm = y.function_space().mesh().mpi_comm()
    problem = problemclass(F, J, y, bcs)
    problem.deflation = deflation

    solver = solverclass(problem, solver_params, prefix=prefix, dm=dm)
    snes = solver.snes

    # all of this is likely defcon-specific and so shouldn't go
    # into the (general-purpose) SNUFLSolver.
    snes.incrementTabLevel(teamno*2)
    setSnesMonitor(prefix)

    oldksp = snes.ksp
    oldksp.incrementTabLevel(teamno*2)
    defksp = DeflatedKSP(deflation, y, oldksp)
    snes.ksp = PETSc.KSP().createPython(defksp, comm)
    snes.ksp.pc.setType('none')

    try:
        solver.solve()
    except ConvergenceError:
        pass
    except:
        import traceback
        traceback.print_exc()
        pass

    success = snes.getConvergedReason() > 0
    iters   = snes.getIterationNumber()
    return (success, iters)
