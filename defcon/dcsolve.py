# Just solve a PDE specified in defcon format
# for a given set of parameters.

import defcon.backend as backend
from defcon.newton import newton
from defcon.tasks import DeflationTask
from defcon.mg import create_dm

def dcsolve(problem, params, comm=backend.comm_world, guess=None, deflation=None):

    mesh = problem.mesh(comm)
    Z = problem.function_space(mesh)

    if guess is None:
        z = problem.initial_guess(Z, params, 0)
    elif isinstance(guess, int):
        z = problem.initial_guess(Z, params, guess)
    elif isinstance(guess, backend.Function):
        z = guess.copy(deepcopy=True)

    v = backend.TestFunction(Z)
    w = backend.TrialFunction(Z)

    F = problem.residual(z, params, v)
    J = problem.jacobian(F, z, params, v, w)
    bcs = problem.boundary_conditions(Z, params)

    task = DeflationTask(0, None, None, 0, params)
    dm = create_dm(Z, problem)
    teamno = 0 # FIXME: make this optional

    (success, iters) = newton(F, J, z, bcs,
                              params,
                              problem,
                              problem.solver_parameters(params, task),
                              teamno, deflation=deflation, dm=dm)

    if success:
        return (success, iters, z)
    else:
        return (success, iters, None)
