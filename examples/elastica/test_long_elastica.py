from defcon import *
from elastica import ElasticaProblem
from math import pi
import matplotlib.pyplot as plt
import pytest

@pytest.mark.skipif(backend.comm_world.size < 5, reason="Needs to run on more cores")
def test_elastica():
    problem = ElasticaProblem()
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True, logfiles=True)
    values = linspace(0, 3.9*pi, 200)
    dc.run(values={"lambda": values, "mu": [0.5]}, freeparam="lambda")

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    if backend.comm_world.rank == 0:
        final = (values[-1], 0.5)
        branches = io.known_branches(final)
        assert len(branches) == 7

        stabilities = io.fetch_stability(final, branches)
        assert sum(stabilities) == 2 # Two stable, one unstable

        # Now check *all* stabilities have been computed
        for value in values:
            param = (value, 0.5)
            branches = io.known_branches(param)
            stabilities = io.fetch_stability(param, branches)
            assert len(branches) == len(stabilities)
            assert sum(stabilities) <= 2

    # Check that this doesn't crash
    dc.bifurcation_diagram("signedL2", fixed={"mu": 0.5})
    plt.title(r"Buckling of an Euler elastica, $\mu = 1/2$")
    plt.savefig("bifurcation.pdf")
