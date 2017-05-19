from defcon.bifurcationproblem import BifurcationProblem
from defcon.newton import compute_tau
from backend import *
from petsc4py import PETSc
from numpy import where, array, int32

def vec(x):
    if isinstance(x, Function):
        x = x.vector()
    return as_backend_type(x).vec()

class VIBifurcationProblem(object):
    def __init__(self, problem, lb, ub):
        """
        Construct a BifurcationProblem for analysing variational inequalities.

        Inputs: problem is the BifurcationProblem associated with the *unconstrained*
        rootfinding problem. lb and ub are lower and upper bounds respectively;
        they will be interpolated into the state function space associated with the
        underlying problem.
        """
        self.problem = problem
        self.lb = lb
        self.ub = ub

        self.function_spaces = {}
        self.lbs = {}
        self.ubs = {}
        self.is_state = {}
        self.is_lb    = {}
        self.is_ub    = {}
        self.zeros    = {}

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def residual(self, z, params, w):
        (u, _, _) = split(z)
        (v, _, _) = split(w)
        return self.problem.residual(u, params, v)

    def function_space(self, mesh):
        D = self.problem.function_space(mesh)
        ub = vec(interpolate(self.ub, D))
        lb = vec(interpolate(self.lb, D))
        zero = vec(Function(D))

        key = mesh.num_cells()
        self.function_spaces[key] = D
        self.lbs[key] = lb
        self.ubs[key] = ub
        self.zeros[key] = zero

        De = D.ufl_element()
        Ze = MixedElement([De, De, De]) # PDE solution, multiplier for lower bound, multiplier for upper bound
        Z  = FunctionSpace(mesh, Ze)

        comm = mesh.mpi_comm()
        is_state = PETSc.IS().createGeneral(Z.sub(0).dofmap().dofs(), comm=comm)
        is_lb = PETSc.IS().createGeneral(Z.sub(1).dofmap().dofs(), comm=comm)
        is_ub = PETSc.IS().createGeneral(Z.sub(2).dofmap().dofs(), comm=comm)

        self.is_state[key] = is_state
        self.is_lb[key] = is_lb
        self.is_ub[key] = is_ub

        return Z

    def boundary_conditions(self, Z, params):
        return self.problem.boundary_conditions(Z.sub(0), params)

    def functionals(self):
        orig = self.problem.functionals()
        out  = []
        for (func, name, latex) in orig:
            def newfunc(z, params):
                u = z.split()[0]
                return func(u, params)

            out.append((newfunc, name, latex))

        return out

    def initial_guess(self, Z, params, n):
        V = Z.sub(0).collapse()
        z = Function(Z)
        u = self.problem.initial_guess(V, params, n)
        assign(z.sub(0), u)
        return z

    def solver(self, problem, solver_params, prefix="", **kwargs):
        base = self.problem.solver(problem, solver_params, prefix=prefix, **kwargs)
        snes = base.snes

        u_dvec = as_backend_type(problem.u.vector())
        mesh = problem.u.function_space().mesh()
        comm = mesh.mpi_comm()
        key  = mesh.num_cells()

        lb = self.lbs[key]
        ub = self.ubs[key]
        is_state  = self.is_state[key]
        is_lb = self.is_lb[key]
        is_ub = self.is_ub[key]
        zero  = self.zeros[key]
        deflation = problem.deflation

        class MySolver(object):
            def step(iself, snes, X, F, Y):
                Xorig = X.copy()

                J = snes.getJacobian()[0]
                snes.computeJacobian(X, J)

                u = X.getSubVector(is_state)
                mu_lb = X.getSubVector(is_lb)
                mu_ub = X.getSubVector(is_ub)

                du = Y.getSubVector(is_state)
                dmu_lb = Y.getSubVector(is_lb)
                dmu_ub = Y.getSubVector(is_ub)

                lb_active_dofs   = where(mu_lb.array_r - (u.array_r - lb.array_r) >  0)[0].astype('int32')
                lb_active_is = PETSc.IS().createGeneral(lb_active_dofs, comm=comm)
                lb_inactive_dofs = where(mu_lb.array_r - (u.array_r - lb.array_r) <= 0)[0].astype('int32')
                lb_inactive_is = PETSc.IS().createGeneral(lb_inactive_dofs, comm=comm)

                ub_active_dofs   = where(mu_ub.array_r - (ub.array_r - u.array_r) >  0)[0].astype('int32')
                ub_active_is = PETSc.IS().createGeneral(ub_active_dofs, comm=comm)
                ub_inactive_dofs = where(mu_ub.array_r - (ub.array_r - u.array_r) <= 0)[0].astype('int32')
                ub_inactive_is = PETSc.IS().createGeneral(ub_inactive_dofs, comm=comm)

                inactive_dofs = array(list(set(lb_inactive_dofs).intersection(set(ub_inactive_dofs))), dtype=int32)
                inactive_is   = PETSc.IS().createGeneral(inactive_dofs, comm=comm)

                du_lb_active = du.getSubVector(lb_active_is)
                du_ub_active = du.getSubVector(ub_active_is)
                du_inactive  = du.getSubVector(inactive_is)

                # Set du where the lower bound is active.

                lb_active = lb.getSubVector(lb_active_is)
                u_lb_active = u.getSubVector(lb_active_is)
                lb_active.copy(du_lb_active) # where lower bound is active: du = lb
                du_lb_active.axpy(-1.0, u_lb_active) #                      du = lb - u
                du_lb_active.scale(-1.0)             # PETSc has a really weird convention
                u.restoreSubVector(lb_active_is, u_lb_active)

                # Set du where the upper bound is active.

                ub_active = ub.getSubVector(ub_active_is)
                u_ub_active = u.getSubVector(ub_active_is)
                ub_active.copy(du_ub_active) # where upper bound is active: du = ub
                du_ub_active.axpy(-1.0, u_ub_active) #                      du = ub - u
                du_ub_active.scale(-1.0)             # PETSc has a really weird convention
                u.restoreSubVector(ub_active_is, u_ub_active)

                # Solve the PDE where the constraints are inactive.
                M = J.createSubMatrix(is_state, is_state)
                M_inact = M.createSubMatrix(inactive_is, inactive_is)
                M_lb    = M.createSubMatrix(inactive_is, lb_active_is)
                M_ub    = M.createSubMatrix(inactive_is, ub_active_is)

                F_u     = F.getSubVector(is_state)
                F_inact = F_u.getSubVector(inactive_is)

                rhs = F_inact.copy()
                tmp = F_inact.duplicate()

                # Need to add and subtract slack variables here
                mu_lb_inactive = mu_lb.getSubVector(inactive_is)
                mu_ub_inactive = mu_ub.getSubVector(inactive_is)
                rhs.axpy(-1.0, mu_ub_inactive)
                rhs.axpy(+1.0, mu_lb_inactive)
                mu_ub.restoreSubVector(inactive_is, mu_ub_inactive)
                mu_lb.restoreSubVector(inactive_is, mu_lb_inactive)

                M_lb.mult(du_lb_active, tmp)
                rhs.axpy(-1.0, tmp)
                M_ub.mult(du_ub_active, tmp)
                rhs.axpy(-1.0, tmp)

                ksp = PETSc.KSP().create(comm=comm)
                ksp.setOperators(M_inact)
                ksp.setType("preonly")
                ksp.pc.setType("lu")
                ksp.pc.setFactorSolverPackage("mumps")
                ksp.setFromOptions()
                ksp.setUp()
                ksp.solve(rhs, du_inactive)

                del rhs
                del M_ub, M_lb, M_inact

                F_u.restoreSubVector(inactive_is, F_inact)

                lb.restoreSubVector(lb_active_is, lb_active)
                ub.restoreSubVector(ub_active_is, ub_active)

                du.restoreSubVector(inactive_is,  du_inactive)
                du.restoreSubVector(ub_active_is, du_ub_active)
                du.restoreSubVector(lb_active_is, du_lb_active)

                # Now set the slacks. First, the lower bound.
                mu_lb.copy(dmu_lb)

                dmu_lb_active = dmu_lb.getSubVector(lb_active_is)
                mu_lb_active = mu_lb.getSubVector(lb_active_is)
                dmu_lb_active.axpy(-1.0, mu_lb_active)
                mu_lb.restoreSubVector(lb_active_is, mu_lb_active)

                # dmu_lb should be whatever it needs to be so that
                # the Newton equation is satisfied where the lower bound
                # is active.
                M_lb = M.createSubMatrix(lb_active_is, None)
                F_lb = F_u.getSubVector(lb_active_is)

                M_lb.mult(du, dmu_lb_active)   # dmu_lb = J.du
                dmu_lb_active.axpy(-1.0, F_lb) # dmu_lb = -F + J.du on lb-active part

                dmu_lb.restoreSubVector(lb_active_is, dmu_lb_active)
                F_u.restoreSubVector(lb_active_is, F_lb)
                del M_lb

                # Now set the upper bound.

                # First, the change on the inactive part is whatever's required
                # to ensure the slack is zero there.
                mu_ub.copy(dmu_ub)

                dmu_ub_active = dmu_ub.getSubVector(ub_active_is)
                mu_ub_active = mu_ub.getSubVector(ub_active_is)
                dmu_ub_active.axpy(-1.0, mu_ub_active)
                mu_ub.restoreSubVector(ub_active_is, mu_ub_active)

                # dmu_ub should be whatever it needs to be so that
                # the Newton equation is satisfied where the upper bound
                # is active.
                M_ub = M.createSubMatrix(ub_active_is, None)
                F_ub = F_u.getSubVector(ub_active_is)

                M_ub.mult(du, dmu_ub_active)   # dmu_ub = J.du
                dmu_ub_active.scale(-1.0)      # dmu_ub = -J.du
                dmu_ub_active.axpy(+1.0, F_ub) # dmu_ub = F - J.du on ub-active part

                dmu_ub.restoreSubVector(ub_active_is, dmu_ub_active)

                F_u.restoreSubVector(ub_active_is, F_ub)
                del M_ub

                F.restoreSubVector(is_state, F_u)

                Y.restoreSubVector(is_ub, dmu_ub)
                Y.restoreSubVector(is_lb, dmu_lb)
                Y.restoreSubVector(is_state, du)

                X.restoreSubVector(is_ub, mu_ub)
                X.restoreSubVector(is_lb, mu_lb)
                X.restoreSubVector(is_state, u)

                tau = compute_tau(deflation, problem.u, Y)
                Y.scale(tau)

        (f, (fenicsresidual, args, kargs)) = snes.getFunction()

        def plus(X, out):
            out.pointwiseMax(X, zero)

        def newresidual(snes, X, F):
            fenicsresidual(snes, X, F)

            F_u = F.getSubVector(is_state)
            F_lb = F.getSubVector(is_lb)
            F_ub = F.getSubVector(is_ub)

            u = X.getSubVector(is_state)
            mu_lb = X.getSubVector(is_lb)
            mu_ub = X.getSubVector(is_ub)

            tmp_state = F_u.duplicate()
            tmp_c1 = F_lb.duplicate()
            tmp_c2 = F_lb.duplicate()

            # Add the extra components to the u residual.

            F_u.axpy(-1.0, mu_lb)            # F = J'(u) - mu_lb
            F_u.axpy(+1.0, mu_ub)            # F = J'(u) + mu_ub - mu_lb

            # Now set the residual for lb.

            u.copy(tmp_c1)             # tmp1 = u
            tmp_c1.axpy(-1.0, lb)      # tmp1 = u - a
            tmp_c1.scale(-1.0)         # tmp1 = -(u - a)
            tmp_c1.axpy(+1.0, mu_lb)   # tmp1 = mu_lb - (u - a)
            plus(tmp_c1, tmp_c2)       # tmp2 = (mu_lb - (u - a))_+
            tmp_c2.scale(-1.0)         # tmp2 = - (mu_lb - (u - a))_+
            tmp_c2.axpy(+1.0, mu_lb)   # tmp2 = mu_lb - (mu_lb - (u - a))_+
            tmp_c2.copy(F_lb)

            # Now set the residual for ub.

            ub.copy(tmp_c1)            # tmp1 = b
            tmp_c1.axpy(-1.0, u)       # tmp1 = b - u
            tmp_c1.scale(-1.0)         # tmp1 = -(b - u)
            tmp_c1.axpy(+1.0, mu_ub)   # tmp1 = mu - (b - u)
            plus(tmp_c1, tmp_c2)       # tmp2 = (mu - (b - u))_+
            tmp_c2.scale(-1.0)         # tmp2 = - (mu - (b - u))_+
            tmp_c2.axpy(+1.0, mu_ub)   # tmp2 = mu - (mu - (b - u))_+
            tmp_c2.copy(F_ub)

            X.restoreSubVector(is_ub, mu_ub)
            X.restoreSubVector(is_lb, mu_lb)
            X.restoreSubVector(is_state,  u)

            #print "|F_u|: ", F_u.norm(PETSc.NormType.NORM_2)
            #print "|F_lb|: ", F_lb.norm(PETSc.NormType.NORM_2)
            #print "|F_ub|: ", F_ub.norm(PETSc.NormType.NORM_2)

            F.restoreSubVector(is_ub, F_ub)
            F.restoreSubVector(is_lb, F_lb)
            F.restoreSubVector(is_state, F_u)

            res = F.norm(PETSc.NormType.NORM_2)

        snes.setType("python")
        snes.setPythonContext(MySolver())
        snes.setFunction(newresidual, f)

        return base

    def save_pvd(self, z, pvd):
        u = z.split()[0]
        self.problem.save_pvd(u, pvd)

