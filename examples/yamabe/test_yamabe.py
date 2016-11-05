from defcon import *
from yamabe import YamabeProblem

def test_yamabe():
    problem = YamabeProblem()
    deflation = ShiftedDeflation(problem, power=1, shift=1.0e-2)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"a": [8.0]})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    import backend
    if backend.comm_world.rank == 0:
        assert len(io.known_branches((8.0,))) == 7
