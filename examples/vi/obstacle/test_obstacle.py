from defcon import *
from obstacle import ObstacleProblem, lb, ub

def test_obstacle():
    eqproblem = ObstacleProblem()
    viproblem = VIBifurcationProblem(eqproblem, lb, ub)
    dc = DeflatedContinuation(viproblem, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"f": -10.0})

    io = viproblem.io()
    io.setup(viproblem.parameters(), viproblem.functionals(), None)

    if backend.comm_world.rank == 0:
        params = (-10.0,)
        assert len(io.known_branches(params)) == 1

        functionals = io.fetch_functionals([params], 0)[0]
        assert 0.65 < functionals[0] < 0.66
