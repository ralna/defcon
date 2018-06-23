from defcon import *
from tinloi42 import TinLoiProblem, N, F

def test_tinloi42():
    problem = TinLoiProblem(F, N)
    deflation = ShiftedDeflation(problem, power=1, shift=1)
    dc = DeflatedContinuation(problem, deflation=deflation, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"q": 1})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    if backend.comm_world.rank == 0:
        params = (1,)
        assert len(io.known_branches(params)) == 2
