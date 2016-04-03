from petsc4py import PETSc
from utils import empty_vector
from dolfin import *
from math import isnan, isinf

class PetscSnesSolver(object):
  def __init__(self):
    self.initialised = False

  def setup(self, problem, u, monitor=None):
    self.initialised = True

    b = empty_vector(u)
    u_vec = as_backend_type(u).vec()

    dims = (u.local_size(), u.size())
    comm = PETSc.Comm(problem.mesh.mpi_comm())

    class PetscMatShell(object):
      def mult(self, mat, x, y):
        x_wrap = PETScVector(x)
        y_wrap = PETScVector(y)
        problem.Jv(x_wrap, y_wrap, u)

      if hasattr(problem, 'JTv'):
          def multTranspose(self, mat, x, y):
            x_wrap = PETScVector(x)
            y_wrap = PETScVector(y)
            problem.JTv(x_wrap, y_wrap, u)

    class PetscPrecShell(object):
      def apply(self, pc, x, y):
        problem.pc_apply(x, y)

      def view(self, pc, vw):
        if hasattr(problem, 'P'):
            problem.P.view(vw)

    class PetscFunctions(object):
        def residual(self, obj, x, b):
            b_wrap = PETScVector(b)
            x_wrap = PETScVector(x)
            u_wrap = PETScVector(u_vec)

            x_wrap.update_ghost_values()
            x.copy(u_vec)
            u_wrap.update_ghost_values()

            problem.F(b_wrap, x_wrap)

        def jacobian(self, snes, x, J, P):
            x_wrap = PETScVector(x)
            u_wrap = PETScVector(u_vec)

            x_wrap.update_ghost_values()
            x.copy(u_vec)
            u_wrap.update_ghost_values()

            problem.build_cache(x_wrap, snes)

        def objective(self, snes, x):
            b = x.duplicate()
            self.residual(snes, x, b)

            b_norm = b.norm()
            if isnan(b_norm) or isinf(b_norm):
                info_green("Objective returning 1.0e20")
                return 1.0e20
            else:
                info_green("Objective returning %s" % b_norm)
                return b_norm

    pJ = PETSc.Mat()
    pJ.createPython((dims, dims), PetscMatShell(), comm)
    pJ.setUp()

    petscfunctions = PetscFunctions()
    self.snes = PETSc.SNES().create(comm)
    self.snes.setFunction(petscfunctions.residual, as_backend_type(b).vec())
    self.snes.setJacobian(petscfunctions.jacobian, pJ)
    #self.snes.setObjective(petscfunctions.objective)
    self.snes.setName("nld")

    opt = PETSc.Options().getAll()
    if (hasattr(problem, 'lb') or hasattr(problem, 'ub')) and (opt['snes_type'].startswith("vi") or opt['snes_type'] == "composite"):
        assert hasattr(problem, 'lb')
        assert hasattr(problem, 'ub')
        self.snes.setVariableBounds(problem.lb, problem.ub)

    self.snes.setFromOptions()

    self.snes.ksp.pc.setType("python")
    shell = PetscPrecShell()
    self.snes.ksp.pc.setPythonContext(shell)

    # If we're using a nonlinear SNES preconditioner, give it the matrix-free PC too:
    if 'npc_snes_type' in opt:
        self.snes.npc.ksp.pc.setType("python")
        self.snes.npc.ksp.pc.setPythonContext(shell)
        self.snes.npc.setName("npc")

    # Also check for composite SNES:
    if 'snes_type' in opt:
        if opt['snes_type'] == 'composite':
            types = opt['snes_composite_sneses'].split(',')
            for i in range(len(types)):
                subsnes = self.snes.getCompositeSNES(i)
                subsnes.ksp.pc.setType("python")
                subsnes.ksp.pc.setPythonContext(shell)
                subsnes.setFunction(petscfunctions.residual, as_backend_type(b).vec())
                subsnes.setJacobian(petscfunctions.jacobian, pJ)

                if subsnes.getType().startswith("vi"):
                    subsnes.setVariableBounds(problem.lb, problem.ub)

        if opt['snes_type'] == 'ngs':
            from petsctools import set_snes_ngs_coloring_from_mat
            info_red("Setting SNES colouring from matrix!")
            problem.build_cache(u)
            set_snes_ngs_coloring_from_mat(self.snes, as_backend_type(problem._J).mat())


    if monitor is not None:
        self.snes.setMonitor(monitor)
    if hasattr(problem, 'monitor'):
        self.snes.setMonitor(problem.monitor)

  def solve(self, problem, u, monitor=None):

    if not self.initialised:
        self.setup(problem, u, monitor)

    log = PETSc.Log().Stage("Outer SNES solver (%s)" % self.snes.getType())
    log.push()

    problem.set_solver(self.snes)

    u_copy = Vector(u)
    self.snes.solve(None, as_backend_type(u_copy).vec())
    u.zero()
    u.axpy(1.0, u_copy)

    problem.unset_solver()

    self.initialised = False
    log.pop()

    reason = self.snes.getConvergedReason()
    if reason < 0:
        raise ValueError("SNES did not converge")

    return self.snes.getIterationNumber()
