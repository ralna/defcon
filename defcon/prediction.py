# Implement tangent predictor for continuation steps. These
# are intended for use in the BifurcationProblem.predict method.

import defcon.backend as backend
from ufl import derivative
from defcon.newton import newton
from defcon.tasks import TangentPredictionTask
from defcon.variationalinequalities import VIBifurcationProblem

def tangent(problem, solution, oldparams, newparams, hint=None):
    oldparams = map(backend.Constant, oldparams)
    chgparams = map(backend.Constant, (new - old for (new, old) in zip(newparams, oldparams)))

    Z = solution.function_space()
    v = backend.TestFunction(Z)
    w = backend.TrialFunction(Z)

    # FIXME: cache the symbolic calculation once, it can be expensive sometimes
    du = backend.Function(Z)

    F = problem.residual(solution, oldparams, v)
    G = derivative(F, solution, du) + sum(derivative(F, oldparam, chgparam) for (oldparam, chgparam) in zip(oldparams, chgparams))

    J = problem.jacobian(G, du, chgparams, v, w)

    # FIXME: figure out if the boundary conditions depend on
    # the parameters, and set the boundary conditions on the update
    dubcs = problem.boundary_conditions(Z, newparams)
    [dubc.homogenize() for dubc in dubcs]

    dm = problem._dm

    # FIXME: make this optional
    teamno = 0

    task = TangentPredictionTask(map(float, oldparams), map(float, newparams))

    # FIXME: there's probably a more elegant way to do this.
    # Or should we use one semismooth Newton step? After all
    # we already have a Newton linearisation.
    if isinstance(problem, VIBifurcationProblem):
        # If we're dealing with a VI, we need to enforce the appropriate
        # bound constraints on the update problem. Essentially, we need
        # that
        # lb <= u + du <= ub
        # so
        # lb - u <= du <= ub - u

        orig = problem.problem
        state = solution.split(deepcopy=True)[0]
        class FixTheBounds(object):
            def bounds(self, Z, params):
                (lb, ub) = orig.bounds(Z, newparams)
                lb.vector().axpy(-1.0, state.vector())
                ub.vector().axpy(-1.0, state.vector())

                return (lb, ub)

            def __getattr__(self, attr):
                return getattr(orig, attr)

        problem.problem = FixTheBounds()

    (success, iters) = newton(G, J, du, dubcs,
                              problem.nonlinear_problem,
                              chgparams,
                              problem.solver,
                              problem.solver_parameters(oldparams, task),
                              teamno, deflation=None, dm=dm)

    if isinstance(problem, VIBifurcationProblem):
        # Restore the original bounds calculations
        problem.problem = orig

    if not success:
        # Should we raise an Exception here? After all, this is only an auxiliary
        # problem.
        raise ValueError("Tangent linearisation failed")

    solution.assign(solution + du)

    return None
