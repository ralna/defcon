"""
Bimatrix game with three equilibria.

Drawn from

@article{aggarwal1973,
year={1973},
journal={Mathematical Programming},
volume={4},
number={1},
doi={10.1007/BF01584663},
title={On the generation of all equilibrium points for bimatrix games through the {Lemke--Howson} Algorithm},
author={Aggarwal, V.},
pages={233--234},
}
"""

from dolfin import *
from defcon import *

N = 4
def F(z, params):
    x1 = z[0]
    x2 = z[1]
    y1 = z[2]
    y2 = z[3]

    F1 = 30*y1 + 20*y2 - 1
    F2 = 10*y1 + 25*y2 - 1
    F3 = 30*x1 + 20*x2 - 1
    F4 = 10*x1 + 25*x2 - 1
    F = [F1, F2, F3, F4]

    return F

class AggarwalProblem(ComplementarityProblem):
    def parameters(self):
        lamda = Constant(0)
        return [(lamda, "lambda", r"$\lambda$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant([0, 0, 0.1, 0]), V)

    def number_solutions(self, params):
        return 3

    def solver_parameters(self, params, klass):
        args = {
               "snes_max_it": 100,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_stol": 0.0,
               "snes_converged_reason": None,
               "snes_monitor": None,
               "snes_linesearch_damping": 1.0,
               "snes_linesearch_maxstep": 1.0,
               "snes_linesearch_type": "l2",
               "ksp_type": "preonly",
               "pc_type": "svd",
               }
        return args

if __name__ == "__main__":
    problem = AggarwalProblem(F, N)
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True)
    dc.run(values={"lambda": 0})
