from __future__ import absolute_import

# Various minor fixes for firedrake API incompatibility
import defcon.backend as backend
from petsc4py import PETSc

def function_space_dimension(V):
    if backend.__name__ == "dolfin":
        return V.dim()
    elif backend.__name__ == "firedrake":
        return V.dof_dset.layout_vec.getSize()

def make_comm(comm):
    # Garth has arbitrarily broken the API for no good reason
    # whatsoever, with no opportunity for discussion. Wonderful.

    if backend.__name__ == "dolfin" and hasattr(backend, "has_pybind11") and backend.has_pybind11():
        return comm
    elif backend.__name__ == "dolfin":
        return PETSc.Comm(comm)
    elif backend.__name__ == "firedrake":
        return comm
