from defcon import *
from hyperelasticity import HyperelasticityProblem
from mpi4py import MPI
from petsc4py import PETSc

def test_elastica():
    problem = HyperelasticityProblem()
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True)
    values = list(arange(0.0, 0.05, 0.005)) + [0.05]
    dc.run(values={"eps": values})

    io = problem.io()
    V = problem.function_space(problem.mesh(MPI.COMM_SELF))
    io.setup(problem.parameters(), problem.functionals(), V)

    if backend.comm_world.rank == 0:
        final = (values[-1], 0.05)
        branches = io.known_branches(final)
        assert len(branches) == 3

        stabilities = io.fetch_stability(final, branches)
        print("stabilities: %s" % stabilities)
        assert sum(stab["stable"] for stab in stabilities) == 2 # Two stable, one unstable
