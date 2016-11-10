# Code to support geometric multigrid in defcon.
import backend
from petsc4py import PETSc

if backend.__name__ == "dolfin":
    from backend import as_backend_type, Function, MixedElement, VectorElement, \
                        FunctionSpace

# Set up multigrid support
def create_dm(V, problem=None):
    comm = V.mesh().mpi_comm()
    coarse_meshes = problem.coarse_meshes(comm)
    coarse_fs = []
    for coarse_mesh in coarse_meshes:
        coarse_fs.append(problem.function_space(coarse_mesh))

    all_meshes = coarse_meshes + [V.mesh()]
    all_fs     = coarse_fs + [V]
    all_dms    = [create_fs_dm(W) for W in all_fs]

    return all_dms[-1]

# This code is needed to set up shell DM's that hold the index
# sets and allow nice field-splitting to happen.
def create_fs_dm(V):
    # firedrake does its own MG, we have nothing to do with it
    if backend.__name__ == "firedrake":
        return None

    comm = V.mesh().mpi_comm()

    # this way the dm knows the function space it comes from
    dm = PETSc.DMShell().create(comm=comm)
    dm.setAttr('__fs__', V)

    # this gives the dm a template to create vectors inside snes

    dm.setGlobalVector(as_backend_type(Function(V).vector()).vec())

    # if we have a mixed function space, then we need to tell PETSc
    # how to divvy up the different parts of the function space.
    # This is not needed for non-mixed elements.
    ufl_el = V.ufl_element()
    if isinstance(ufl_el, MixedElement) and not isinstance(ufl_el, VectorElement):
        dm.setCreateSubDM(create_subdm)
        dm.setCreateFieldDecomposition(create_field_decomp)

    return dm

# This provides PETSc the information needed to decompose
# the field -- the set of names (currently blank, allowing petsc
# to simply enumerate them), the tuple of index sets, and the
# dms for the resulting subspaces.
def create_field_decomp(dm, *args, **kwargs):
    W = dm.getAttr('__fs__')
    Wsubs = [Wsub.collapse() for Wsub in W.split()]
    names = [None for Wsub in Wsubs]
    dms = [funcspace2dm(Wsub) for Wsub in Wsubs]
    return (names, funcspace_to_index_sets(W), dms)

# For a non-mixed function space, this converts the array of dofs
# into a PETSc IS.
# For a mixed (but not vector) function space, it returns a tuple
# of the PETSc IS'es for each field.
def funcspace_to_index_sets(fs):
    uflel = fs.ufl_element()
    comm = fs.mesh().mpi_comm()
    if isinstance(uflel, MixedElement) and not isinstance(uflel, VectorElement):
        splitdofs = [V.dofmap().dofs() for V in fs.split()]
        ises = [PETSc.IS().createGeneral(sd, comm=comm)
                for sd in splitdofs]
        return tuple(ises)
    else:
        return (PETSc.IS().createGeneral(fs.dofmap().dofs(), comm=comm),)

# since field splitting occurs by having DM shells indicate
# which dofs belong to which field, we need to create DMs for
# the relevant subspaces in order to have recursive field splitting.
def create_subdm(dm, fields, *args, **kwargs):
    W = dm.getAttr('__fs__')
    comm = W.mesh().mpi_comm()
    if len(fields) == 1:
        f = int(fields[0])
        subel = W.sub(f).ufl_element()
        subspace = FunctionSpace(W.mesh(), subel)
        subdm = funcspace2dm(subspace)
        iset = PETSc.IS().createGeneral(W.sub(f).dofmap().dofs(), comm)
        return iset, subdm
    else:
        subel = MixedElement([W.sub(int(f)).ufl_element() for f in fields])
        subspace = FunctionSpace(W.mesh(), subel)
        subdm = create_dm(subspace)

        alldofs = numpy.concatenate(
            [W.sub(int(f)).dofmap().dofs() for f in fields])
        iset = PETSc.IS().createGeneral(sorted(alldofs), comm=comm)

    return (iset, subdm)

