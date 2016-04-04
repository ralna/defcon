from dolfin import *
from utils import empty_vector
from petsc4py import PETSc

class ForwardProblem(NonlinearProblem):
  """The base class for the undeflated forward problem.
  Separates details of
   - residual evaluation
   - Jacobian action
   - Jacobian transpose action
   - preconditioner application

  from PETSc. This is so that we can go matrix-free for the deflated problems."""

  def __init__(self, problem, F, Y, y, bcs=None, power=1, shift=1, bounds=None, P=None):
    """
    The constructor: takes in the form, function space, solution,
    boundary conditions, and deflation parameters."""

    assert isinstance(Y, FunctionSpace)
    self.problem = problem
    self.function_space = Y
    self.mesh = Y.mesh()
    self.comm = PETSc.Comm(self.mesh.mpi_comm())

    NonlinearProblem.__init__(self)
    self.y = y
    self.bcs = bcs

    self._form = F
    self._dF = derivative(F, y)
    self.assembler = SystemAssembler(self._dF, self._form, self.bcs)
    self._J = PETScMatrix()

    # All the known solutions to be deflated
    self.solutions = []

    self.power = power
    self.shift = shift
    self.norms = []
    self.dnorms = []

    self._tmpvec1 = empty_vector(y.vector())
    self._tmpvec2 = empty_vector(y.vector())
    self.residual = empty_vector(y.vector())

    # Sometimes for various problems you want to solve submatrices of the base
    # matrix -- e.g. for active set methods for variational inequalities,
    # and in nonlinear fieldsplits.
    self.eqn_subindices = None
    self.var_subindices = None
    self.inact_subindices = None
    self.pc_prefix = "inner_"

    # in case a fieldsplit preconditioner is requested, and the blocksize
    # isn't set by dolfin
    self.fieldsplit_is = None

    # the near nullspace of the operator
    self.nullsp = None

    if bounds is not None:
        (lb, ub) = bounds
        self.lb = as_backend_type(lb).vec()
        self.ub = as_backend_type(ub).vec()

    # in case you want to use a different matrix to build
    # a preconditioner.
    self.Pmat = None
    if P is not None:
        self.Pmat = P

  def norm(self, y, solution):
    return self.problem.normsq(y, solution)

  def deflate(self, root):
    self.solutions.append(Function(root))

  def deflation(self, rebuild=False):
    """
    Evaluate the deflation factor at a point x.
    """

    eta = 1.0

    # FIXME: or what if we wanted to put the shift outside the loop?
    if rebuild is False:
        assert len(self.norms) == len(self.solutions)
        for norm in self.norms:
            factor = norm**(-self.power/2.0) + self.shift
            eta *= factor
    else:
        for norm in [assemble(self.norm(self.y, solution)) for solution in self.solutions]:
            factor = norm**(-self.power/2.0) + self.shift
            eta *= factor

    return eta

  def build_deflations(self):
    self.norms = [assemble(self.norm(self.y, solution)) for solution in self.solutions]
    self.dnorms = [assemble(derivative(self.norm(self.y, solution), self.y)) for solution in self.solutions]
    self.assembler.assemble(self.residual, self.y.vector())

  def deflation_derivative(self):
    """
    Evaluate the derivative of the deflation factor at a point x.
    """

    if len(self.solutions) == 0:
        deta = empty_vector(self.y.vector())
        deta.zero()
        return deta

    p = self.power
    factors = []
    dfactors = []
    for norm in self.norms:
        factor = norm**(-p/2.0) + self.shift
        dfactor = (-p/2.0) * norm**((-p/2.0) - 1.0)

        factors.append(factor)
        dfactors.append(dfactor)

    eta = product(factors)

    deta = empty_vector(self.y.vector())
    deta.zero()

    for (solution, factor, dfactor, dnorm) in zip(self.solutions, factors, dfactors, self.dnorms):
        deta.axpy((eta/factor)*dfactor, dnorm)

    return deta

  def build_cache(self, x, snes=None):
    if snes is not None:
        snes_prefix = snes.getOptionsPrefix()
        if snes_prefix is not None:
            if len(snes_prefix) > 0:
                if not self.pc_prefix.startswith(snes_prefix):
                    self.pc_prefix = snes_prefix + "inner_"

    if (self.y.vector() - x).norm("l2") >= 1.0e-12:
        print "Hmm. |y - x| == %s" % (self.y.vector() - x).norm("l2")
        #assert (self.y.vector() - x).norm("l2") < 1.0e-12
        self.y.vector().zero()
        self.y.vector().axpy(1.0, x) # agh

    self.assembler.assemble(self._J)

    self.build_deflations()
    self.build_preconditioner()

  def set_nfs_subindices(self, equations, variables):
    self.eqn_subindices = equations
    self.var_subindices = variables

    if equations is not None or variables is not None:
        assert equations is not None
        assert variables is not None
        assert equations.size == variables.size

  def set_inact_subindices(self, inact):
    self.inact_subindices = inact

  def get_submat(self):
    Jmat = mat(self._J)

    if self.eqn_subindices is not None:
        Jmat = Jmat.getSubMatrix(self.eqn_subindices, self.var_subindices)

    if self.inact_subindices is not None:
        Jmat = Jmat.getSubMatrix(self.inact_subindices, self.inact_subindices)

    Pmat = None
    if self.Pmat is not None:
        Pmat = self.Pmat(self.y)
        if self.eqn_subindices is not None:
            Pmat = Pmat.getSubMatrix(self.eqn_subindices, self.var_subindices)

        if self.inact_subindices is not None:
            Pmat = Pmat.getSubMatrix(self.inact_subindices, self.inact_subindices)

    return (Jmat, Pmat)

  def set_pc_prefix(self, prefix):
    if not prefix.endswith("_"):
        prefix = prefix + "_"
    self.pc_prefix = prefix

  def set_fieldsplit_is(self, fieldsplit_is):
    self.fieldsplit_is = fieldsplit_is

  def build_preconditioner(self):
    opts = PETSc.Options().getAll()

    if 'snes_lag_preconditioner' in opts:
      if opts['snes_lag_preconditioner'] == "-1" and hasattr(self, 'P'):
        self.P.setReusePreconditioner(True)
        return

    self.P = PETSc.PC().create(self.comm)

    # Set the operator appropriately, taking the right subsets
    (Jmat, Pmat) = self.get_submat()

    if self.nullsp is not None:
        Jmat.setNearNullSpace(self.nullsp)

    self.P.setOperators(Jmat, Pmat)

    (self._ptmpvec1, self._ptmpvec2) = map(PETScVector, Jmat.createVecs())

    self.P.setType("lu")
    self.P.setFactorSolverPackage("mumps")

    self.P.setOptionsPrefix(self.pc_prefix)

    self.P.setFromOptions()

    if self.P.getType() == "fieldsplit" and Jmat.getBlockSize() <= 1 and self.eqn_subindices is None:
        if self.fieldsplit_is is None:
            self.fieldsplit_is = []
            for i in range(self.function_space.num_sub_spaces()):
                subdofs = SubSpace(self.function_space, i).dofmap().dofs()
                iset = PETSc.IS().createGeneral(subdofs)
                self.fieldsplit_is.append(("%s" % i, iset))

        if self.inact_subindices is None:
            self.P.setFieldSplitIS(*self.fieldsplit_is)
        else:
            # OK. Get the dofs from the inactive set and figure out which split they're from.
            # Agh. VIs make everything so complicated.

            fsets = [set(field[1].getIndices()) for field in self.fieldsplit_is]
            inact_fieldsplit_is = {}
            for field in self.fieldsplit_is:
                inact_fieldsplit_is[int(field[0])] = []

            # OK. Suppose you have a 6x6 matrix and the splits are [1, 2, 3, 4]; [5, 6].
            # Suppose further that the inactive indices are [2, 3, 4, 5]. Then what we
            # want to produce for the fieldsplit of the reduced problem is [1, 2, 3]; [4].
            # In other words, we add the *index* of the inactive dof to the reduced split
            # if the inactive dof is in the full split.

            # We also need the offset of how many dofs all earlier processes own.
            from mpi4py import MPI as MPI4
            offset = MPI4.COMM_WORLD.exscan(self.inact_subindices.getLocalSize())
            if offset is None: offset = 0

            for (i, idx) in enumerate(self.inact_subindices.getIndices()):
                for (j, fset) in enumerate(fsets):
                    if idx in fset:
                        inact_fieldsplit_is[j].append(offset + i)
                        break

            inact_input = []
            for j in inact_fieldsplit_is:
                iset = PETSc.IS().createGeneral(inact_fieldsplit_is[j], comm=self.comm)
                inact_input.append(("%s" % j, iset))

            for (orig_data, vi_data) in zip(self.fieldsplit_is, inact_input):
                orig_iset = orig_data[1]
                vi_iset   = vi_data[1]

                if orig_iset.getSizes() == vi_iset.getSizes():
                    nullsp = orig_iset.query("nearnullspace")
                    if nullsp is not None:
                        vi_iset.compose("nearnullspace", nullsp)

            self.P.setFieldSplitIS(*inact_input)

    if self.eqn_subindices is not None:
        nullsp = self.eqn_subindices.query("nearnullspace")
        if nullsp is not None:
            op = self.P.getOperators()[0]
            op.setNearNullSpace(nullsp)

    self.preconditioner_hook() # argh, such a hack.

    self.P.setUp()

  def preconditioner_hook(self):
    pass

  def F(self, b, x):
    self.assembler.assemble(self._tmpvec1, self.y.vector())

    b.zero()
    b.axpy(self.deflation(rebuild=True), self._tmpvec1)

  def J(self, A, x):
    assert False
    self.build_cache(x)
    return

  def Jv(self, v, Jonv, x):
    self._J.mult(v, self._tmpvec1)

    Jonv.zero()
    Jonv.axpy(self.deflation(), self._tmpvec1)

    derivative = self.deflation_derivative()
    Jonv.axpy(derivative.inner(v), self.residual)

  def JTv(self, v, JTonv, x):
    # FIXME: implement for the deflated problem also
    assert self.solutions == []
    self._J.transpmult(v, JTonv)

  def pc_apply(self, x, y):
    x_wrap = PETScVector(x)
    y_wrap = PETScVector(y)
    self.Pv(x_wrap, y_wrap)

  def Pv(self, v, Ponv):
    self._ptmpvec1.zero()
    self._ptmpvec1.axpy(1.0, v)

    self.P.apply(as_backend_type(self._ptmpvec1).vec(), as_backend_type(self._ptmpvec2).vec())

    beta = 1.0/self.deflation()
    Ponv.zero()
    Ponv.axpy(beta, self._ptmpvec2)

    derivative = self.deflation_derivative()
    if self.eqn_subindices is not None:
        derivative_v = vec(derivative)
        subderivative = derivative_v.getSubVector(self.eqn_subindices)
        derivative_ = PETScVector(subderivative.copy())
        derivative_v.restoreSubVector(self.eqn_subindices, subderivative)
        derivative = derivative_

        residual_v = vec(self.residual)
        subresidual = residual_v.getSubVector(self.eqn_subindices)
        residual_ = PETScVector(subresidual.copy())
        residual_v.restoreSubVector(self.eqn_subindices, subresidual)
        residual = residual_
    else:
        residual = self.residual

    tmp_derivative = derivative.inner(self._ptmpvec2)

    if abs(tmp_derivative) > 0:
        self._ptmpvec2.zero()
        self._ptmpvec2.axpy(1.0, self.residual)

        self.P.apply(as_backend_type(self._ptmpvec2).vec(), as_backend_type(self._ptmpvec1).vec())
        denom = 1 + beta * derivative.inner(self._ptmpvec1)
        Ponv.axpy(-beta * beta * tmp_derivative / denom, self._ptmpvec1)

  def set_solver(self, snes):
    self.snes = snes

  def get_solver(self):
    return self.snes

  def unset_solver(self):
    del self.snes

  def set_near_nullspace(self, nullsp, constant=False):
    assert self.eqn_subindices is None
    assert self.inact_subindices is None

    vecs = map(vec, nullsp)
    self.nullsp = PETSc.NullSpace().create(vectors=vecs, constant=constant, comm=self.comm)

# These functions transfer between FEniCS' wrapper classes around
# Vec and Mat and petsc4py's wrappers around Vec and Mat.
def vec(z):
    if isinstance(z, dolfin.cpp.Function):
        return as_backend_type(z.vector()).vec()
    else:
        return as_backend_type(z).vec()

mat = lambda x: as_backend_type(x).mat()

class VIForwardProblem(ForwardProblem):
    def __init__(self, F, Y, y, bcs=None, power=1, shift=1, bounds=None, P=None):
        ForwardProblem.__init__(self, F, Y, y, bcs, power=power, shift=shift, bounds=bounds, P=P)

    def build_cache(self, x, snes=None):
        ForwardProblem.build_cache(self, x, snes)
        self.pc_set_up = False

    def set_solver(self, snes):
        """ Agh. This is where PETSc gets very complicated. We need to know the SNES so that we
            can ask it for its inactive set and create the right preconditioner. Now, if you
            just have one SNES of type VINEWTON?SLS, then it's straightforward. But what about
            composition? What if you have a SNESCOMPOSITE and it's one of the subsneses that knows
            your inactive set? What if it's a nonlinear preconditioner? So here we have to
            go looking. """

        self.snes = snes

        if snes.getType() == "composite":
            for i in range(snes.getCompositeNumber()):
                subsnes = snes.getCompositeSNES(i)
                if subsnes.getType().startswith("vi"):
                    self.snes = subsnes

    def build_preconditioner(self):
        # this routine is called by ForwardProblem.build_cache,
        # but we can't build the preconditioner at cache time
        # because we need the active set. So we do it ourselves inside Pv.
        pass

    def _build_preconditioner(self):
        if self.snes.getType().startswith("vi"):
            inact = self.snes.getVIInactiveSet()
            self.set_inact_subindices(inact)
        else:
            self.set_inact_subindices(None)

        ForwardProblem.build_preconditioner(self)
        self.pc_set_up = True

    def Pv(self, v, Ponv):
        x = vec(v)

        if not self.pc_set_up:
            self._build_preconditioner()

        y = vec(Ponv)

        assert x.size == y.size
        if self.inact_subindices is not None:
            assert x.size == self.inact_subindices.size

        # FIXME: do I not need to implement the deflation preconditioner
        # low-rank update stuff here?
        self.P.apply(x, y)
