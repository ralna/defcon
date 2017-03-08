# -*- coding: utf-8 -*-
from defcon import *
from dolfin import *

def plus(x):
    return conditional(gt(x, 0), x, 0)
def minus(x):
    return conditional(lt(x, 0), x, 0)

class Obstacle(Expression):
    def eval(self, values, x):
        if x[0] < -0.5:
            values[0] = -0.2
            return
        if -0.5 <= x[0] <= 0.0:
            values[0] = -0.4
            return
        if 0.0 <= x[0] < 0.5:
            values[0] = -0.6
            return
        if 0.5 <= x[0] <= 1.0:
            values[0] = -0.8
            return
c = Constant(100)

class ObstacleProblem(BifurcationProblem):
    def mesh(self, comm):
        return RectangleMesh(comm, Point(-1, -1), Point(1, 1), 64, 64, "crossed")

    def coarse_meshes(self, comm):
        return [RectangleMesh(comm, Point(-1, -1), Point(1, 1), 16, 16, "crossed"), RectangleMesh(comm, Point(-1, -1), Point(1, 1), 32, 32, "crossed")]

    def function_space(self, mesh):
        Vele = FiniteElement("CG", triangle, 1)
        Dele = FiniteElement("CG", triangle, 1)
        Zele = MixedElement([Vele, Dele])
        Z = FunctionSpace(mesh, Zele)

        # Construct BCs here for efficiency
        self._bcs = [DirichletBC(Z.sub(0), 0.0, "on_boundary")]

        # And the obstacle:
        D = FunctionSpace(mesh, Dele)
        self.psi = interpolate(Obstacle(element=Dele), D)

        return Z

    def parameters(self):
        f = Constant(0)
        return [(f, "f", r"$f$")]

    def residual(self, z, params, w):
        (u, lmbda) = split(z)
        (v, mu)    = split(w)
        f = params[0]

        psi = self.psi

        F = (
              inner(grad(u), grad(v))*dx
            - inner(lmbda, v)*dx
            - inner(f, v)*dx
            + inner(lmbda, mu)*dx
            - inner(plus(lmbda - (u - psi)), mu)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return self._bcs

    def functionals(self):
        def uL2(z, params):
            u = split(z)[0]
            j = assemble(u*u*dx)
            return j

        return [(uL2, "uL2", r"$\|u\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        return 1

    def save_pvd(self, z, pvd):
        u = z.split(deepcopy=True)[0]
        u.rename("Solution", "Solution")

        pvd << u

    def solver_parameters(self, params, klass):
        return {
               "snes_max_it": 50,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "ksp_monitor": None,
               "ksp_rtol": 1.0e-10,
               "ksp_atol": 1.0e-10,
               "pc_type": "lu",
               "pc_factor_mat_solver_package": "mumps",
               }

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=ObstacleProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"f": -10})
