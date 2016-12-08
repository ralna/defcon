# -*- coding: utf-8 -*-
import backend
import iomodule
import branchio
import nonlinearproblem, nonlinearsolver

class BifurcationProblem(object):
    """
    A base class for bifurcation problems.

    This object is overridden by the user to implement his particular problem.
    """

    def mesh(self, comm):
        """
        This method loads/builds the mesh with a given communicator.

        *Arguments*
          comm (MPI communicator)
            The MPI communicator to use in building the mesh.

            Typically, the MPI communicator passed in here will be shared among
            a small number of the processors (called a team). Each team solves a
            PDE independently.
        *Returns*
          mesh (:py:class:`dolfin.Mesh`)
        """
        raise NotImplementedError

    def coarse_meshes(self, comm):
        """
        This method supplies additional coarsenings of the mesh for use
        in multigrid methods.

        Return

        [coarsest_mesh, next_finer, next_finer, ...]

        where the last mesh returned is the coarsening of the mesh returned by
        the .mesh() method.

        Of course, you may wish to make the mesh returned by .mesh() by refining
        the meshes returned here; defcon guarantees that this method is called
        after .mesh(), so construct the hierarchy of meshes in .mesh(), return the
        finest, and return the remainder here.

        The reason why this is split into two routines is because many users
        will not want to use multigrid, and this API design is simpler for them.
        """
        return []

    def function_space(self, mesh):
        """
        This method creates the function space for the prognostic variables of
        the problem.

        *Arguments*
          mesh (:py:class:`dolfin.Mesh`)
            The mesh to use.
        *Returns*
          functionspace (:py:class:`dolfin.FunctionSpace`)
        """
        raise NotImplementedError

    def parameters(self):
        """
        This method returns a list of tuples. Each tuple contains (Constant,
        asciiname, tex). For example, if there is one parameter

        lmbda = Constant(...)

        to be varied, this routine should return

        [(lmbda, "lambda", r"$\lambda$")]

        You may need to add

        # -*- coding: utf-8 -*-

        to the top of the Python script to use UTF characters.

        The values in the Constants are irrelevant; they are initialised in the
        continuation.

        *Returns*
          params
            A list of [(Constant, asciiname, tex), ...]
        """
        raise NotImplementedError

    def residual(self, state, params, test):
        """
        This method defines the PDE to be solved: if you would solve the PDE via

        solve(F == 0, state, ...)

        then this method should return F.

        The parameters will be varied internally by the continuation algorithm,
        i,e.  this will not be called multiple times for multiple parameters.

        *Arguments*
          state (:py:class:`dolfin.Function`)
            a Function in the FunctionSpace
          params (tuple of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
          test  (:py:class:`dolfin.TestFunction`)
            the test function to use in defining the residual
        *Returns*
          residual (:py:class:`ufl.form.Form`)
            the form for the residual
        """
        raise NotImplementedError

    def jacobian(self, F, state, params, test, trial):
        """
        This method defines the Jacobian to use in Newton's method.

        If you would solve the PDE via

        solve(F == 0, state, ...)

        then this method should return derivative(F, state, trial).

        The parameters will be varied internally by the continuation algorithm,
        i,e.  this will not be called multiple times for multiple parameters.

        *Arguments*
          F (:py:class:`ufl.form.Form`)
            the output of BifurcationProblem.residual
          state (:py:class:`dolfin.Function`)
            a Function in the FunctionSpace
          params (tuple of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
          test (:py:class:`dolfin.TestFunction`)
            the test function to use in defining the Jacobian
          trial (:py:class:`dolfin.TrialFunction`)
            the trial function to use in defining the Jacobian
        *Returns*
          jacobian (:py:class:`ufl.form.Form`)
            the form for the Jacobian
        """
        return backend.derivative(F, state, trial)

    def objective(self, F, state, params):
        """
        WARNING: this method is currently unused, pending the implementation
        of necessary features in PETSc.

        This method computes the norm of the residual.

        The default is to compute the l_2 norm of the assembled
        residual vector. This has the advantage of speed, but is
        mesh-dependent. For mesh-independent termination criteria
        the appropriate dual norm should be coded here.

        Most users will not override this method; if you're not sure
        you want it, you don't need it.

        *Arguments*
          F (:py:class:`ufl.form.Form`)
            the output of BifurcationProblem.residual
          state (:py:class:`dolfin.Function`)
            a Function in the FunctionSpace
          params (tuple of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
        *Returns*
          objective (:py:class:`float`)
            the dual norm of the residual
        """
        pass

    def boundary_conditions(self, function_space, params):
        """
        This method returns a list of DirichletBC objects to impose on the
        problem.

        *Arguments*
          function_space (:py:class:`dolfin.FunctionSpace`)
            the function space returned by self.function_space()
          params (list of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
        *Returns*
          bcs (list of :py:class:`dolfin.DirichletBC`)
        """
        raise NotImplementedError

    def functionals(self):
        """
        This method returns a list of functionals. Each functional is a tuple
        consisting of a callable, an ascii name, and a tex label.  The callable
        J is called via

          j = J(state, params)

        and should return a float.

        For example, this routine might consist of

        def functionals(self):
            def L2norm(state, param):
                return assemble(inner(state, state)*dx)**0.5
            return [(L2norm, "L2norm", r"\|y\|")]

        *Returns*
          functionals (list of tuples)
        """
        raise NotImplementedError

    def number_initial_guesses(self, params):
        """
        Return the number of initial guesses we wish to search from at the
        very first deflation step, i.e. when initialising the continuation.
        """
        raise NotImplementedError

    def initial_guess(self, function_space, params, n):
        """
        Return the n^{th} initial guess.
        """
        raise NotImplementedError

    def number_solutions(self, params):
        """
        If the number of solutions for a given set of parameters is analytically
        known, then this function can return it. In this case the continuation
        algorithm will stop looking when it has found that many solutions.
        Otherwise the routine should return float("inf").

        *Arguments*
          params (tuple of :py:class:`float`)
            parameters to use, in the same order returned by parameters()

        *Returns*
          nsolutions (int)
            number of solutions, known from analysis, or float("inf")
        """
        return float("inf")

    def squared_norm(self, state1, state2, params):
        """
        This method computes the squared-norm between two vectors in the Hilbert
        space defined in the function_space method.

        This is used to define the norm used in deflation, and to test two
        functions for equality, among other things.

        The default is
            def squared_norm(self, state1, state2, params)
                return inner(state1 - state2, state1 - state2)*dx

        *Arguments*
        """
        return backend.inner(state1 - state2, state1 - state2)*backend.dx

    def trivial_solutions(self, function_space, params, freeindex):
        """
        This method returns any trivial solutions of the problem,
        i.e. solutions u such that f(u, \lambda) = 0 for all \lambda.
        These will be deflated at every computation.

        freeindex is the index into params that is being modified by this
        current continuation run. This is useful if the trivial solutions
        depend on the values of parameters in params other than freeindex.
        """
        return []

    def monitor(self, params, branchid, solution, functionals):
        """
        This method is called whenever a solution is computed.
        The user can specify custom processing tasks here.
        """
        pass

    def io(self, prefix="output"):
        """
        Return an IO object that defcon will use to save solutions and functionals.

        The default is usually a good choice.
        """

        try:
            import h5py
            #assert hasattr(backend.HDF5File, 'set_mpi_atomicity')
            return branchio.BranchIO(prefix)
        except (ImportError, AssertionError):
            return iomodule.SolutionIO(prefix)

    def save_pvd(self, y, pvd):
        """
        Save the function y to a PVD file.

        The default is

        pvd << y
        """
        pvd << y

    def nonlinear_problem(self, F, J, y, bcs):
        """
        The class used to assemble the nonlinear problem.

        Most users will never need to override this: it's only useful if you
        want to do something unusual in the assembly process.
        """
        if backend.__name__ == "dolfin":
            return nonlinearproblem.GeneralProblem(F, y, bcs, J=J, problem=self)
        else:
            return backend.NonlinearVariationalProblem(F, y, bcs, J=J)

    def solver(self, problem, solver_params, prefix="", **kwargs):
        """
        The class used to solve the nonlinear problem.

        Users might want to override this if they want to customize
        how the nonlinear solver is set up. For example, look at the
        hyperelasticity demo to see how this is used to pass a near-nullspace
        into an algebraic multigrid preconditioner.
        """

        if backend.__name__ == "dolfin":
            return nonlinearsolver.SNUFLSolver(
                problem, prefix=prefix,
                solver_parameters=solver_params,
                **kwargs
            )
        else:
            return backend.NonlinearVariationalSolver(
                problem, options_prefix=prefix,
                solver_parameters=solver_params,
                **kwargs
            )

    def compute_stability(self, params, branchid, solution, hint=None):
        """
        This method allows the user to compute whether a solution on a given branch is
        stable.

        Stability means different things to different problems. For example, in an unconstrained
        energy minimisation problem, investigating stability involves checking the positive
        definiteness of the Hessian at a critical point; in constrained minimisation, one needs
        to check the inertia of the Hessian and verify that the number of negative eigenvalues
        is as expected, given the constraints.

        With this in mind, defcon does not attempt to compute the stability of solutions;
        it just provides a hook for you to calculate whatever test you wish.

        It is often the case that some results from one stability analysis can be used
        to accelerate the calculation of the next. For example, in SLEPc one can
        set an initial guess for the eigenspace, and if it is a good guess the algorithms
        converge much faster. This is the purpose of the hint input. For the first
        calculation on a given branch, it is None, but on subsequent calculations it
        is passed along from the outputs of one calculation to the inputs of the next.

        This routine should return the following: a dictionary like

        d = {"stable": is_stable,
             "eigenvalues":    [list of eigenvalues],
             "eigenfunctions": [list of eigenfunctions],
             "hint": hint}

        is_stable can be anything that can be evaluated with literal_eval in Python.
        Its value has meaning only to you. The reason why this is not True/False
        is because stability is more subtle than that -- for example, one could
        pass None to indicate that the computation failed. In other problems
        (such as the Gross-Pitaevskii equation in quantum mechanics) there are
        different ways a problem can be unstable; these could be distinguished
        by returning "exponential" or "oscillatory".

        eigenvalues is a list of real or complex numbers. These will be saved
        by the I/O object. The list can be empty.

        eigenfunctions is a list of Functions. These will also be saved by
        the I/O object. Again, the list can be empty.

        hint is any data that you wish to pass on to the next calculation.

        For an example of this in action, see the elastica demo.
        """
        pass

    def solver_parameters(self, params, klass):
        """
        Returns a dictionary with the PETSc options to configure
        the backend nonlinear solver.  Users should
        override this method in their own subclasses to set
        solver/preconditioner preferences.

        params is the set of continuation parameters, which is present
        so that users could adapt the solver strategy depending on the
        parameter regime if needed.

        klass is one of {DeflationTask, ContinuationTask, StabilityTask, ArclengthTask},
        and allows for the user to tune the solver parameters as required."""
        return {}

    def transform_guess(self, oldparams, newparams, state):
        """
        When performing deflation, it's sometimes useful to modify the
        initial guess in some way (e.g. Hermite promotion in the case
        of a nonlinear Schroedinger equation, or perturbation in the case
        of perfect Z_2 symmetry). This method provides exactly such a hook.
        It should modify state in place, not return anything.
        """
        pass

    def branch_found(self, task):
        """
        This hook is experimental. Its interface will probably change.

        When a new branch is discovered, this method is called. It is called
        with the ContinuationTask representing the new branch.

        This method should return a list of additional tasks to perform,
        such as continuation tasks in another parameter dimension. The
        taskids of these additional tasks should be task.taskid + 1,
        task.taskid + 2, ...
        """

        return []
