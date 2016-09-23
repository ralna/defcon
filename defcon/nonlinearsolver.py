import backend
from petsc4py import PETSc
if backend.__name__ == "dolfin":
    from backend import as_backend_type, PETScVector, PETScMatrix, \
        MixedElement, VectorElement, Function, FunctionSpace, \
        SystemAssembler, Form
    import numpy, weakref

    def funcspace2ises(fs):
        uflel = fs.ufl_element()
        if isinstance(uflel, MixedElement) \
           and not isinstance(uflel, VectorElement):
            splitdofs = [V.dofmap().dofs() for V in fs.split()]
            ises = [PETSc.IS().createGeneral(sd) for sd in splitdofs]
            return tuple(ises)
        else:
            return (PETSc.IS().createGeneral(fs.dofmap().dofs()),)


    def create_subdm(dm, fields, *args, **kwargs):
        W = dm.getAttr('__fs__')
        if len(fields) == 1:
            f = int(fields[0])
            subel = W.sub(f).ufl_element()
            subspace = FunctionSpace(W.mesh(), subel)
            subdm = funcspace2dm(subspace)
            iset = PETSc.IS().createGeneral(W.sub(f).dofmap().dofs())
            return iset, subdm
        else:
            sub_el = MixedElement(
                [W.sub(int(f)).ufl_element() for f in fields]
            )
            subspace = FunctionSpace(W.mesh(), sub_el)
            subdm = funcspace2dm(subspace)

            bigises = funcspace2ises(W)

            alldofs = numpy.concatenate(
                [W.sub(int(f)).dofmap().dofs() for f in fields])
            iset = PETSc.IS().createGeneral(sorted(alldofs))
            
        return iset, subdm 


    def create_field_decomp(dm, *args, **kwargs):
        W = dm.getAttr('__fs__')
        Wsubs = [Wsub.collapse() for Wsub in W.split()]
        names = [Wsub.name() for Wsub in Wsubs]
        dms = [funcspace2dm(Wsub) for Wsub in Wsubs]
        return names, funcspace2ises(W), dms

        
    # This code is needed to set up shell dm's that hold the index
    # sets and allow nice field-splitting to happen
    def funcspace2dm(func_space):
        # We need to do different things based on whether
        # we have a mixed element or not
        comm = func_space.mesh().mpi_comm()

        # this way the dm knows the function space it comes from
        dm = PETSc.DMShell().create(comm=comm)
        dm.setAttr('__fs__', func_space)

        # this gives the dm a template to create vectors inside snes
        
        dm.setGlobalVector(
            as_backend_type(Function(func_space).vector()).vec()
        )

        # if we have a mixed function space, then we need to tell PETSc
        # how to divvy up the different parts of the function space.
        # This is not needed for non-mixed elements.
        ufl_el = func_space.ufl_element()
        if isinstance(ufl_el, MixedElement) \
           and not isinstance(ufl_el, VectorElement):
            dm.setCreateSubDM(create_subdm)
            dm.setCreateFieldDecomposition(create_field_decomp)

        return dm


    # dolfin lacks a high-level snes frontend like Firedrake,
    # so we're going to put one here and build up what we need
    # to make things happen.
    class SNUFLSolver(object):
        def __init__(self, problem, prefix="", **kwargs):
            self.problem = problem
            u = problem.u
            self.u_dvec = as_backend_type(u.vector())
            self.u_pvec = self.u_dvec.vec()

            comm = u.function_space().mesh().mpi_comm()
            self.comm = comm
            snes = PETSc.SNES().create(comm=comm)
            snes.setOptionsPrefix(prefix)

            # Fix what must be one of the worst defaults in PETSc
            opts = PETSc.Options()
            if (prefix + "snes_linesearch_type") not in opts:
                opts[prefix + "snes_linesearch_type"] = "basic"

            J, F, bcs, P = problem.J, problem.F, problem.bcs, problem.P

            self.ass = SystemAssembler(J, F, bcs)
            if P is not None:
                self.Pass = SystemAssembler(P, F, bcs)
                
            self.b = self.init_residual()
            snes.setFunction(self.residual, self.b.vec())
            self.A = self.init_jacobian()
            self.P = self.init_preconditioner(self.A)
            snes.setJacobian(self.jacobian, self.A.mat(), self.P.mat())
            snes.ksp.setOperators(self.A.mat(), self.P.mat()) # why isn't this done in setJacobian?

            snes.setDM(funcspace2dm(u.function_space()))

            snes.setFromOptions()

            self.snes = snes

        def init_jacobian(self):
            A = PETScMatrix(self.comm)
            self.ass.init_global_tensor(A, Form(self.problem.J))
            return A

        def init_residual(self):
            b = as_backend_type(Function(self.problem.u.function_space()).vector())
            return b

        def init_preconditioner(self, A):
            if self.problem.P is None: return A
            P = PETscMatrix(self.comm)
            self.Pass.init_global_tensor(P, Form(self.P))
            return P

        def update_x(self, x):
            """Given a PETSc Vec x, update the storage of our
               solution function u."""

            x.copy(self.u_pvec)
            self.u_dvec.update_ghost_values()

        def residual(self, snes, x, b):
            self.update_x(x)
            b_wrap = PETScVector(b)
            self.ass.assemble(b_wrap, self.u_dvec)

        def jacobian(self, snes, x, A, P):
            self.update_x(x)
            A_wrap = PETScMatrix(A)
            P_wrap = PETScMatrix(P)
            self.ass.assemble(A_wrap)
            if self.problem.P is not None:
                self.Pass.assemble(P_wrap)

        def solve(self):
            # Need a copy for line searches etc. to work correctly.
            x = self.problem.u.copy(deepcopy=True)
            xv = as_backend_type(x.vector()).vec()

            try:
                self.snes.solve(None, xv)
            except:
                import traceback
                traceback.print_exc()
                pass
