# Configuration and mesh taken from D. N. Arnold,
# http://www.math.umn.edu/~arnold/8446.s12/programs/
import sys
from   math import floor

from defcon import *
from dolfin import *

from petsc4py import PETSc

class StokesProblem(BifurcationProblem):
    def mesh(self, comm):

        # Load coarsest mesh
        self.meshes = [Mesh(comm, "mesh/mesh.xml.gz")]

        # Refine mesh a few times
        for i in range(4):
            cmesh = self.meshes[-1]
            fmesh = refine(cmesh, redistribute=False)
            self.meshes.append(fmesh)

        return fmesh

    def coarse_meshes(self, comm):
        return self.meshes[:-1]

    def function_space(self, mesh):
        Ve = VectorElement("CG", mesh.ufl_cell(), 2)
        Qe = FiniteElement("CG", mesh.ufl_cell(), 1)
        Ze = MixedElement([Ve, Qe])
        Z = FunctionSpace(mesh, Ze)
        return Z

    def parameters(self):
        # Dummy parameter: we don't use it
        f = Constant(0)
        return [(f, "f", r"$f$")]

    def residual(self, z, params, w):
        (u, p) = split(z)
        (v, q) = split(w)

        tol = 1.0e-6
        F = (
              inner(grad(u), grad(v))*dx
            + div(v)*p*dx
            + div(u)*q*dx
            - tol*p*q*dx
            )

        return F

    def boundary_conditions(self, Z, params):
        tol = 1.0e-10

        # top and bottom: no slip
        bc0val = Constant((0, 0))
        bc0loc = "on_boundary && ((fabs(x[0]) > -2 + {tol}) && (x[0] < 2.0 - {tol}))".format(tol=tol)
        bc0    = DirichletBC(Z.sub(0), bc0val, bc0loc)

        # left: quadratic inflow
        bc1val = Expression(("x[1]-x[1]*x[1]", "0.0"), degree=2, mpi_comm=Z.mesh().mpi_comm())
        bc1loc = "on_boundary && (x[0] < -2.0 + {tol})".format(tol=tol)
        bc1    = DirichletBC(Z.sub(0), bc1val, bc1loc)

        # right: slower quadratic outflow
        bc2val = Expression(("(1-x[1]*x[1])/8.", "0.0"), degree=2, mpi_comm=Z.mesh().mpi_comm())
        bc2loc = "on_boundary && (x[0] > 2.0 - {tol})".format(tol=tol)
        bc2    = DirichletBC(Z.sub(0), bc2val, bc2loc)

        return [bc0, bc1, bc2]

    def functionals(self):
        def L2(z, params):
            (u, p) = split(z)
            j = sqrt(assemble(inner(u, u)*dx))
            return j

        return [(L2, "L2", r"$\|u\|$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        return Function(Z)

    def number_solutions(self, params):
        return 1

    def solver_parameters(self, params, klass):
        args = {
               "snes_max_it": 10,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "snes_view": None,
               "ksp_type": "richardson",
               "ksp_monitor_true_residual": None,
               "ksp_atol": 1.0e-10,
               "ksp_rtol": 1.0e-10,
               "pc_type": "lu",
               "pc_factor_mat_solver_package": "mumps",
               "pc_mg_galerkin": None,
               }
        return args

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=StokesProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"f": [1.0]})
