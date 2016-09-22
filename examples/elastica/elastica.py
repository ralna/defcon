# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

import matplotlib.pyplot as plt

args = [sys.argv[0]] + """
                       --petsc.snes_max_it 100
                       --petsc.snes_atol 1.0e-9
                       --petsc.snes_rtol 0.0
                       --petsc.snes_monitor

                       --petsc.ksp_type preonly
                       --petsc.pc_type lu

                       --petsc.eps_type krylovschur
                       --petsc.eps_target -1
                       --petsc.eps_monitor_all
                       --petsc.eps_converged_reason
                       --petsc.eps_nev 1
                       --petsc.st_type sinvert
                       --petsc.st_ksp_type preonly
                       --petsc.st_pc_type lu
                       """.split()
parameters.parse(args)

from petsc4py import PETSc
from slepc4py import SLEPc # for stability calculations

class ElasticaProblem(BifurcationProblem):
    def __init__(self):
        self.bcs = None

    def mesh(self, comm):
        return IntervalMesh(comm, 1000, 0, 1)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        lmbda = Constant(0)
        mu    = Constant(0)

        return [(lmbda, "lambda", r"$\lambda$"),
                (mu,    "mu",     r"$\mu$")]

    def residual(self, theta, params, v):
        (lmbda, mu) = params

        F = (
              inner(grad(theta), grad(v))*dx
              - lmbda**2*sin(theta)*v*dx
              + mu*v*dx
            )

        return F

    def boundary_conditions(self, V, params):
        # The boundary conditions are independent of parameters, so only
        # evaluate them once for efficiency.
        if self.bcs is None:
            self.bcs = [DirichletBC(V, 0.0, "on_boundary")]
        return self.bcs

    def functionals(self):
        def signedL2(theta, params):
            j = sqrt(assemble(inner(theta, theta)*dx))
            g = project(grad(theta)[0], theta.function_space())
            return j*g((0.0,))

        def max(theta, params):
            return theta.vector().max()

        def min(theta, params):
            return theta.vector().min()

        return [(signedL2, "signedL2", r"$\theta'(0) \|\theta\|$"),
                (max, "max", r"$\max{\theta}$"),
                (min, "min", r"$\min{\theta}$")]

    def trivial_solutions(self, V, params, freeindex):
        # check we're continuing in lambda:
        if freeindex == 0:
            # check if mu is 0
            if params[1] == 0.0:
                # return the trivial solution
                return [Function(V)]
        return []

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant(1), V)

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of lambda.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.

        (lmbda, mu) = params
        n = int(floor((lmbda/pi)))*2 # this is the exact formula for mu = 0, but works for mu = 0.5 also

        if mu == 0: return max(n, 1) # don't want the trivial solution
        else:       return n + 1

    def squared_norm(self, a, b, params):
        return inner(a - b, a - b)*dx + inner(grad(a - b), grad(a - b))*dx

    def compute_stability(self, params, branchid, theta, hint=None):
        V = theta.function_space()
        trial = TrialFunction(V)
        test  = TestFunction(V)

        bcs = self.boundary_conditions(V, params)
        comm = V.mesh().mpi_comm()

        F = self.residual(theta, map(Constant, params), test)
        J = derivative(F, theta, trial)
        b = inner(Constant(1), test)*dx # a dummy linear form, needed to construct the SystemAssembler

        # Build the LHS matrix
        A = PETScMatrix(comm)
        asm = SystemAssembler(J, b, bcs)
        asm.assemble(A)

        # Build the mass matrix for the RHS of the generalised eigenproblem
        B = PETScMatrix(comm)
        asm = SystemAssembler(inner(test, trial)*dx, b, bcs)
        asm.assemble(B)
        [bc.zero(B) for bc in bcs]

        # Create the SLEPc eigensolver
        eps = SLEPc.EPS.create(comm=comm)
        eps.setOperators(A.mat(), B.mat())
        eps.setWhichEigenpairs(eps.Which.SMALLEST_MAGNITUDE)
        eps.setProblemType(eps.ProblemType.GHEP)
        eps.setFromOptions()

        # If we have a hint, use it - it's the eigenfunctions from the previous solve
        if hint is not None:
            initial_space = [vec(x) for x in hint]
            eps.setInitialSpace(initial_space)

        # Solve the eigenproblem
        eps.solve()

        eigenvalues = []
        eigenfunctions = []
        eigenfunction = Function(V, name="Eigenfunction")

        for i in range(eps.getConverged()):
            lmbda = eps.getEigenvalue(i)
            eigenvalues.append(lmbda)

            eps.getEigenvector(i, vec(eigenfunction))
            eigenfunctions.append(eigenfunction.copy(deepcopy=True))

        if min(eigenvalues) < 0:
            is_stable = False
        else:
            is_stable = True

        d = {"stable": is_stable,
             "eigenvalues": eigenvalues,
             "eigenfunctions": eigenfunctions,
             "hint": eigenfunctions}

        return d

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=ElasticaProblem(), teamsize=1, verbose=True)
    dc.run(free={"lambda": linspace(0, 3.9*pi, 200)}, fixed={"mu": 0.5})

    dc.bifurcation_diagram("signedL2", fixed={"mu": 0.5})
    plt.title(r"Buckling of an Euler elastica, $\mu = 1/2$")
    plt.savefig("bifurcation.pdf")

    # Maybe you could also do:
    #dc.run(fixed={"lambda": 4*pi}, free={"mu": linspace(0.5, 0.0, 6)})
    #dc.run(fixed={"mu": 0.0}, free={"lambda": linspace(4*pi, 0.0, 100)})

