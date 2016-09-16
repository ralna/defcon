import backend
from petsc4py import PETSc
import sys
from numpy import isnan

# Unfortunately DOLFIN and firedrake are completely different in how they
# do the solve. So we have to branch based on backend here.

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
            Edy = getEdy(deflation, self.y, dy_pet)
                
            minv = 1.0 / deflation.evaluate(self.y)
            tau = (1 + minv*Edy/(1 - minv*Edy))
            dy_pet.scale(tau)

        ksp.setConvergedReason(self.ksp.getConvergedReason())



if backend.__name__ == "dolfin":
    from backend import PETScSNESSolver, PETScOptions, PETScVector, as_backend_type

    def getEdy(deflation, y, dy):
        dy_vec = PETScVector(dy)
        return -deflation.derivative(y).inner(dy_vec)
        
    

    def newton(F, y, bcs, problemclass, teamno, deflation=None, prefix=""):

        comm = y.function_space().mesh().mpi_comm()
        solver = PETScSNESSolver(comm)
        snes = solver.snes()
        problem = problemclass(F, y, bcs)

        snes.setOptionsPrefix(prefix)
        PETScOptions.set(prefix + "snes_monitor_cancel")
        solver.init(problem, y.vector())

        snes.incrementTabLevel(teamno*2)

        oldksp = snes.ksp
        oldksp.incrementTabLevel(teamno*2)
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

elif backend.__name__ == "firedrake":
    from backend import NonlinearVariationalSolver

    def getEdy(deflation, y, dy):
        with deflation.derivative(y).dat.vec_ro as deriv:
            Edy = -deriv.dot(dy)
        return Edy

    def newton(F, y, bcs, problemclass, teamno, deflation=None, prefix=""):

        problem = problemclass(F, y, bcs)
        solver  = NonlinearVariationalSolver(problem, options_prefix=prefix)
        snes = solver.snes
        comm = snes.comm

        snes.incrementTabLevel(teamno*2)

        oldksp = snes.ksp
        oldksp.incrementTabLevel(teamno*2)
        defksp = DeflatedKSP(deflation, y, oldksp)
        snes.ksp = PETSc.KSP().createPython(defksp, comm)
        snes.ksp.pc.setType('none')
        snes.ksp.setOperators(*oldksp.getOperators())
        snes.ksp.setUp()

        try:
            solver.solve()
        except:
            pass

        success = snes.getConvergedReason() > 0
        return success

else:
    raise ImportError("Unknown backend")
