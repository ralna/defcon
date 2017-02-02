from __future__ import absolute_import

# Various minor fixes for firedrake API incompatibility
import defcon.backend as backend

def function_space_dimension(V):
    if backend.__name__ == "dolfin":
        return V.dim()
    elif backend.__name__ == "firedrake":
        return V.dof_dset.layout_vec.getSize()
