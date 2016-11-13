from defcon import *
from dolfin import *

from petsc4py import PETSc

class LaplaceProblem(BifurcationProblem):
    def mesh(self, comm):
        mesh = Mesh(comm, "mesh/canal.xml.gz")
        facets = MeshFunction("size_t", mesh, "mesh/facets.xml.gz")
        self.facets = facets
        return mesh

    def function_space(self, mesh):
        Ve = VectorElement("CG", mesh.ufl_cell(), 2)
        Qe = FiniteElement("CG", mesh.ufl_cell(), 1)
        Ze = MixedElement([Ve, Qe])
        Z  = FunctionSpace(mesh, Ze)
        self.Z = Z
        return Z

    def parameters(self):
        # Dummy parameter, not used
        f = Constant(0)
        return [(f, "f", r"$f$")]

    def residual(self, z, params, w):
        (u, p) = split(z)
        (v, q) = split(w)

        F = (
              inner(grad(u), grad(v)) * dx
            - div(v) * p * dx
            - q * div(u) * dx
            )

        return F

    def boundary_conditions(self, Z, params):
        facets = self.facets

        zerovector = Constant((0.0, 0.0))
        inflow = Expression(("0.25 * (4 - x[1] * x[1])", "0.0"), mpi_comm=Z.mesh().mpi_comm(), degree=2)
        bcs = [DirichletBC(Z.sub(0), inflow, facets, 2),
               DirichletBC(Z.sub(0), (0, 0), facets, 1),
               DirichletBC(Z.sub(0), (0, 0), facets, 4)]
        return bcs

    def functionals(self):
        def L2(z, params):
            (u, _) = split(z)
            j = sqrt(assemble(inner(u, u)*dx))
            return j

        return [(L2, "L2", r"$\|u\|$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

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

    def save_pvd(self, z, pvd):
        (u, p) = z.split(deepcopy=True)
        u.rename("Velocity", "Velocity")
        p.rename("Pressure", "Pressure")
        pvd << u

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=LaplaceProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"f": [1.0]})
