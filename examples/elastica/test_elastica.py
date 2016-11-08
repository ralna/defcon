from defcon import *
from elastica import ElasticaProblem
from math import pi

def test_elastica():
    problem = ElasticaProblem()
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True)
    values = linspace(0, 1.1*pi, 10)
    dc.run(values={"lambda": values, "mu": [0.5]}, freeparam="lambda")

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    import backend
    if backend.comm_world.rank == 0:
        final = (values[-1], 0.5)
        branches = io.known_branches(final)
        assert len(branches) == 3

        stabilities = io.fetch_stability(final, branches)
        assert sum(stabilities) == 2 # Two stable, one unstable
