# Just solve a PDE specified in defcon format
# for a given set of parameters.

import defcon.backend as backend
from defcon.newton import newton
from defcon.tasks import DeflationTask
from defcon.mg import create_dm

def dcsolve(problem, params, comm=backend.comm_world, guess=None, deflation=None, sp=None):

    if isinstance(guess, backend.Function):
        mesh = guess.function_space().mesh()
    else:
        mesh = problem.mesh(comm)

    if isinstance(guess, backend.Function):
        Z = guess.function_space()
    else:
        Z = problem.function_space(mesh)

    nguesses = problem.number_initial_guesses(params)
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

    if sp is None:
        sp = problem.solver_parameters(params, task)

    (success, iters) = newton(F, J, z, bcs,
                              params,
                              problem,
                              sp,
                              teamno, deflation=deflation, dm=dm)

    if success:
        return (success, iters, z)
    else:
        return (success, iters, None)
