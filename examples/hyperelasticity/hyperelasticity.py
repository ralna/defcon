# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

from Probe import Probe # borrowed from Mikael Mortensen's fenicstools
from numpy import array
from petsc4py import PETSc

args = [sys.argv[0]] + """
                       --petsc.snes_max_it 100
                       --petsc.snes_atol 1.0e-7
                       --petsc.snes_rtol 1.0e-10
                       --petsc.snes_monitor

                       --petsc.ksp_type gmres
                       --petsc.ksp_monitor_cancel
                       --petsc.ksp_converged_reason
                       --petsc.ksp_max_it 2000
                       --petsc.pc_type gamg
                       """.split()
parameters.parse(args)

class HyperelasticityProblem(BifurcationProblem):
    def mesh(self, comm):
        return RectangleMesh(comm, Point(0, 0), Point(1, 0.1), 50, 50)

    def function_space(self, mesh):
        V = VectorFunctionSpace(mesh, "CG", 1)

        # Construct rigid body modes used in algebraic multigrid preconditioner later on
        rbms = [Constant((0, 1)),
                Constant((1, 0)),
                Expression(("-x[1]", "x[0]"), mpi_comm=mesh.mpi_comm())]
        self.rbms = [interpolate(rbm, V) for rbm in rbms]

        return V

    def parameters(self):
        eps = Constant(0)

        return [(eps, "eps", r"$\epsilon$")]

    def residual(self, u, params, v):
        B   = Constant((0.0, -1000)) # Body force per unit volume

        # Kinematics
        I = Identity(2)             # Identity tensor
        F = I + grad(u)             # Deformation gradient
        C = F.T*F                   # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ic = tr(C)
        J  = det(F)

        # Elasticity parameters
        E, nu = 1000000.0, 0.3
        mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

        # Total potential energy
        Energy = psi*dx - dot(B, u)*dx #- dot(T, u)*ds
        F = derivative(Energy, u, v)

        return F

    def boundary_conditions(self, V, params):
        eps = params[0]
        left  = CompiledSubDomain("(std::abs(x[0])       < DOLFIN_EPS) && on_boundary", mpi_comm=V.mesh().mpi_comm())
        right = CompiledSubDomain("(std::abs(x[0] - 1.0) < DOLFIN_EPS) && on_boundary", mpi_comm=V.mesh().mpi_comm())

        bcl = DirichletBC(V, (0.0,  0.0), left)
        bcr = DirichletBC(V, (-eps, 0.0), right)

        return [bcl, bcr]

    def functionals(self):
        def pointeval(u, params):
            # use the Probe class from 'fenicstools' to evaluate the
            # solution at the point (0.25, 0.05)

            probe = Probe(array([0.25, 0.05]), u.function_space())
            probe(u)
            j = probe[0][1]
            return j

        return [(pointeval, "pointeval", r"$u(0.25, 0.05)$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant((0, 0)), V)

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of eps.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.
        eps = params[0]
        if eps < 0.03:
            return 1
        if eps < 0.07:
            return 3
        if eps < 0.12:
            return 5
        if eps < 0.18:
            return 7
        if eps < 0.20:
            return 9
        return float("inf")

    def squared_norm(self, a, b):
        return inner(a - b, a - b)*dx + inner(grad(a - b), grad(a - b))*dx

    def configure_snes(self, snes):
        # Set the rigid body modes for use in AMG

        if snes.ksp.pc.type != "preonly":
            # Convert rigid body modes (computed in self.function_space above) to PETSc Vec
            vec = lambda x: as_backend_type(x.vector()).vec()
            rbms = map(vec, self.rbms)

            # Create the PETSc nullspace
            nullsp = PETSc.NullSpace().create(vectors=rbms, constant=False, comm=snes.comm)

            (A, P) = snes.ksp.getOperators()
            A.setNearNullSpace(nullsp)
            P.setNearNullSpace(nullsp)

if __name__ == "__main__":
    io = FileIO("output")
    dc = DeflatedContinuation(problem=HyperelasticityProblem(), io=io, teamsize=1, verbose=True)
    params = list(arange(0.0, 0.2, 0.001)) + [0.2]
    dc.run(free={"eps": params})
