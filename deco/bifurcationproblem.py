# -*- coding: utf-8 -*-
import dolfin

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
        asciiname, symbol). For example, if there is one parameter

        lmbda = Constant(...)

        to be varied, this routine should return

        [(lmbda, "lambda", "λ")]

        You may need to add

        # -*- coding: utf-8 -*-

        to the top of the Python script to use UTF characters.

        The values in the Constants are irrelevant; they are initialised in the
        continuation.

        *Returns*
          params
            A list of [(Constant, asciiname, symbol), ...]
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
        """
        raise NotImplementedError

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

    def guesses(self, function_space, oldparams, oldstates, newparams):
        """
        Given the solutions oldstates corresponding to the parameter values
        oldparams, construct a list of guesses for the parameter values
        newparams.

        In the simplest case, this just returns oldstates again.

        There is one special case that must be handled. If oldparams = None and
        oldstates = None, then this routine should return the initial guesses
        to be used for the initial solve (when no solutions are available).

        Each guess in the returned list should have a label attribute
        with a string describing its origin (e.g. prev-soln-5).

        *Arguments*
          function_space (:py:class:`dolfin.FunctionSpace`)
            the function space returned by function_space()
          oldparams (tuple of :py:class:`float`)
            old parameters to use, in the same order returned by parameters()
          oldstates (list of :py:class:`dolfin.Function`)
            solutions known at the old parameter values
          newparams (tuple of :py:class:`float`)
            new parameters to use, in the same order returned by parameters()

        *Returns*
          newguesses (list of :py:class:`dolfin.Function`)
            guesses for the new parameter values
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

    def inner_product(self, state1, state2):
        """
        This method computes the inner product of two vectors in the Hilbert
        space defined in the function_space method.

        This is used to define the norm used in deflation, and to test two
        functions for equality, among other things.

        The default is
            def inner_product(self, state1, state2)
                return inner(state1, state2)*dx

        *Arguments*
        """
        return dolfin.inner(state1, state2)*dolfin.dx

    def trivial_solutions(self, function_space):
        """
        This method returns any trivial solutions of the problem,
        i.e. solutions u such that f(u, \lambda) = 0 for all \lambda.
        These will be deflated at every computation.
        """
        return []

