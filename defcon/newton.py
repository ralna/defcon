from dolfin import *
from petsc4py import PETSc
import sys

def printnorm(i, n):
    print "%3d SNES Function norm %1.15e" % (i, n)
    sys.stdout.flush()

def ksp_setup(ksp):
    pass

def newton(F, y, bcs, deflation=None, prefix="", printnorm=printnorm, ksp_setup=ksp_setup):

    # Fetch some SNES options from the PETSc dictionary
    if len(prefix) > 0:
        if prefix[-1] != "_":
            prefix = prefix + "_"

    opts = PETSc.Options()
    atol = opts.getReal(prefix + "snes_atol", default=1.0e-8)
    rtol = opts.getReal(prefix + "snes_rtol", default=1.0e-8)
    maxits = opts.getInt(prefix + "snes_max_it", default=100)
    monitor = opts.getBool(prefix + "snes_monitor", default=True)

    def norm(F, y, bcs):
        b = assemble(F)
        [bc.apply(b, y.vector()) for bc in bcs]
        return b.norm("l2")

    [bc.apply(y.vector()) for bc in bcs]

    dy = Function(y.function_space())
    dyvec = as_backend_type(dy.vector())
    J = derivative(F, y, TrialFunction(y.function_space()))
    hbcs = [DirichletBC(bc) for bc in bcs]; [hbc.homogenize() for hbc in hbcs]
    i = 0

    n0 = norm(F, y, bcs)
    n  = n0
    if monitor: printnorm(i, n)

    success = False

    mpi_comm = y.function_space().mesh().mpi_comm()
    A = PETScMatrix(mpi_comm)
    b = PETScVector(mpi_comm)

    while True:
        if i >= maxits:  break
        if n <= atol:    success = True; break
        if n/n0 <= rtol: success = True; break

        dyvec.zero()
        (A, b) = assemble_system(J, -F, hbcs, A_tensor=A, b_tensor=b)

        ksp = PETSc.KSP().create(comm=b.vec().comm)
        ksp.setOperators(A=A.mat())

        # Set some sensible defaults
        ksp.setType('preonly')
        ksp.pc.setType('lu')
        ksp.pc.setFactorSolverPackage('mumps')

        # Call the user setup routine
        ksp_setup(ksp)

        ksp.setFromOptions()
        ksp.solve(b.vec(), dyvec.vec())
        dyvec.update_ghost_values()

        if deflation is not None:
            Edy = deflation.derivative(y).inner(dyvec)
            minv = 1.0 / deflation.evaluate(y)
            tau = (1 + minv*Edy/(1 - minv*Edy))
            dy.assign(tau * dy)

        y.assign(y + dy)

        i = i + 1
        n = norm(F, y, bcs)
        if monitor: printnorm(i, n)

    return success

