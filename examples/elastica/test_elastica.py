from defcon import *
from elastica import ElasticaProblem
from math import pi
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc

def test_elastica():
    problem = ElasticaProblem()
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True)
    values = linspace(0, 1.1*pi, 10)
    dc.run(values={"lambda": values, "mu": [0.5]}, freeparam="lambda")

    io = problem.io()

    # More totally unnecessary API breakages, quite frustrating
    try:
        V = problem.function_space(problem.mesh(PETSc.Comm(MPI.COMM_SELF)))
    except TypeError:
        V = problem.function_space(problem.mesh(MPI.COMM_SELF))

    io.setup(problem.parameters(), problem.functionals(), V)

    if backend.comm_world.rank == 0:
        final = (values[-1], 0.5)
        branches = io.known_branches(final)
        assert len(branches) == 3

        stabilities = io.fetch_stability(final, branches)
        print("stabilities: %s" % stabilities)
        assert sum(stab["stable"] for stab in stabilities) == 2 # Two stable, one unstable

    # Check that this doesn't crash
    dc.bifurcation_diagram("signedL2", fixed={"mu": 0.5})
    plt.title(r"Buckling of an Euler elastica, $\mu = 1/2$")
    plt.savefig("bifurcation.pdf")
