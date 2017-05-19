"""
Nonconvex quadratic programming example with unusual central path.

Drawn from

N. I. M. Gould
"The state-of-the-art in numerical methods for quadratic programming"
19th Biennial Conference on Numerical Analysis, Dundee, Scotland.

ftp://ftp.numerical.rl.ac.uk/pub/talks/nimg.dundee.28VI01.ps.gz
"""

from dolfin import *
from defcon import *

N = 4
def F(x):

    F1 = -4*(x[0] - 0.25) + 3*x[2] + x[3]
    F2 = +4*(x[1] - 0.50) + x[2] + x[3]
    F3 = 1.5 - 3*x[0] - x[1]
    F4 = 1.0 - x[0]   - x[1]
    F = [F1, F2, F3, F4]

    return F

class GouldProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = UnitIntervalMesh(comm, 1)
        return mesh

    def function_space(self, mesh):
        Re = FiniteElement("R", interval, 0)
        Ve = MixedElement([Re]*N)
        V = FunctionSpace(mesh, Ve)

        return V

    def parameters(self):
        lamda = Constant(0)
        return [(lamda, "lambda", r"$\lambda$")]

    def residual(self, z, params, v):

        f = inner(as_vector(F(z)), v)*dx
        return f

    def boundary_conditions(self, V, params):
        return []

    def functionals(self):
        def fetch_component(i):
            def func(z, params):
                return z.vector().array()[i]
            return (func, "z[%d]" % i, r"z_{%d}" % i)
        return [fetch_component(i) for i in range(N)]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant((0.3, 0.3, 0.3, 0.3)), V)

    def number_solutions(self, params):
        return 3

    def squared_norm(self, a, b, params):
        return inner(a - b, a - b)*dx

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
               "mat_mumps_icntl_24": 1,
               "mat_mumps_icntl_13": 1
               }
        return args

lb = Constant((0, 0, 0, 0))
ub = Constant((1e20, 1e20, 1e20, 1e20))

if __name__ == "__main__":
    eqproblem = GouldProblem()
    viproblem = VIBifurcationProblem(eqproblem, lb, ub)
    dc = DeflatedContinuation(viproblem, teamsize=1, clear_output=True)
    dc.run(values={"lambda": 0})
