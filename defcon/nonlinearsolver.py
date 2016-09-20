import backend
from petsc4py import PETSc
if backend.__name__ == "dolfin":
    from backend import as_backend_type, PETScVector, PETScMatrix, \
        MixedElement, VectorElement, Function
    import numpy, weakref
    
    def funcspace2ises(mfs, idx=None):
        # returns the PETSc index set associated with the idx'th component
        # of a mixed function space 
        assert isinstance(mfs.ufl_element(), MixedElement) and not isinstance(mfs.ufl_element(), VectorElement)
        if idx is None: # ids of whole space
            return PETSc.IS().createGeneral(mfs.dofmap().dofs())
        else: # ids of a particular subspace
            return PETSc.IS().createGeneral(mfs.sub(idx).dofmap().dofs())

    
    def create_subdm(dm, fields, *args, **kwargs):
        W = dm.getAttr('__fs__')
        if len(fields) == 1:
            field = fields[0]
            subdm = funcspace2dm(W.sub[field])
            iset = funcspace2ises(W, field)
            return iset, subdm
        else:
            sub_el = MixedElement(
                [W.sub[f].ufl_element() for f in fields]
            )
            subspace = FunctionSpace(W.mesh(), sub_el).collapse()
            subdm = funcspace2dm(subspace, True)
            
            iset = PETsc.IS().createGeneral(
                numpy.concatenate(
                    [funcspace2ises(W, field).indices for f in fields]
                )
            )
            
        return iset, subdm 
        
    def create_field_decomp(dm, *args, **kwargs):
        W = dm.getAttr('__fs__')
        Wsubs = [Wsub.collapse() for Wsub in W.split()]
        names = [Wsub.name() for Wsub in Wsubs]
        dms = [funcspace2dm(Wsub) for Wsub in Wsubs]
        return names, funcspace2ises(W), dms

        
    # This code is needed to set up shell dm's that hold the index
    # sets and allow nice field-splitting to happen
    def funcspace2dm(func_space, is_sub=False):
        print "creating dm for ", func_space
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

        print "done creating dm"
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
            snes = PETSc.SNES().create(comm=comm)
            snes.setOptionsPrefix(prefix)

            # Fix what must be one of the worst defaults in PETSc
            opts = PETSc.Options()
            if (prefix + "snes_linesearch_type") not in opts:
                opts[prefix + "snes_linesearch_type"] = "basic"

            self.b = problem.init_residual()
            snes.setFunction(self.residual, self.b.vec())
            self.A = problem.init_jacobian()
            self.P = problem.init_preconditioner(self.A)
            snes.setJacobian(self.jacobian, self.A.mat(), self.P.mat())
            snes.ksp.setOperators(self.A.mat(), self.P.mat()) # why isn't this done in setJacobian?

            snes.setDM(funcspace2dm(u.function_space()))

            snes.setFromOptions()

            self.snes = snes

            

            
        def update_x(self, x):
            """Given a PETSc Vec x, update the storage of our
               solution function u."""

            x.copy(self.u_pvec)
            self.u_dvec.update_ghost_values()

        def residual(self, snes, x, b):
            self.update_x(x)
            b_wrap = PETScVector(b)
            self.problem.assemble_residual(b_wrap, self.u_dvec)

        def jacobian(self, snes, x, A, P):
            self.update_x(x)
            A_wrap = PETScMatrix(A)
            P_wrap = PETScMatrix(P)
            self.problem.assemble_jacobian(A_wrap)
            self.problem.assemble_preconditioner(A_wrap, P_wrap)

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
