from defcon import *
from painleve import PainleveProblem

def test_allen_cahn():
    problem = PainleveProblem()
    dc = DeflatedContinuation(problem, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"a": [6.0]})

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    assert len(io.known_branches((6.0,))) == 2
