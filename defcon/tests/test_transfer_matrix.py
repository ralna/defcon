from defcon.mg import create_transfer_matrix
from backend import *
import pytest

vec = lambda u: as_backend_type(u.vector()).vec()

def xtest_scalar_p1():
    meshc = UnitCubeMesh(2, 2, 2)
    meshf = UnitCubeMesh(3, 4, 5)

    Vc = FunctionSpace(meshc, "CG", 1)
    Vf = FunctionSpace(meshf, "CG", 1)

    u = Expression("x[0] + 2*x[1] + 3*x[2]", degree=1)
    uc = interpolate(u, Vc)
    uf = interpolate(u, Vf)

    mat = create_transfer_matrix(Vc, Vf).mat()
    Vuc = Function(Vf)
    mat.mult(vec(uc), vec(Vuc))
    as_backend_type(Vuc.vector()).update_ghost_values()

    diff = Function(Vf)
    diff.assign(Vuc - uf)
    assert diff.vector().norm("l2") < 1.0e-12

def xtest_scalar_p2():
    meshc = UnitCubeMesh(2, 2, 2)
    meshf = UnitCubeMesh(3, 4, 5)

    Vc = FunctionSpace(meshc, "CG", 2)
    Vf = FunctionSpace(meshf, "CG", 2)

    u = Expression("x[0]*x[2] + 2*x[1]*x[0] + 3*x[2]", degree=2)
    uc = interpolate(u, Vc)
    uf = interpolate(u, Vf)

    mat = create_transfer_matrix(Vc, Vf).mat()
    Vuc = Function(Vf)
    mat.mult(vec(uc), vec(Vuc))
    as_backend_type(Vuc.vector()).update_ghost_values()

    diff = Function(Vf)
    diff.assign(Vuc - uf)
    assert diff.vector().norm("l2") < 1.0e-12

@pytest.mark.skip("Vector elements not working yet")
def test_vector_p1():
    meshc = UnitCubeMesh(2, 2, 2)
    meshf = UnitCubeMesh(3, 4, 5)

    Vc = VectorFunctionSpace(meshc, "CG", 1)
    Vf = VectorFunctionSpace(meshf, "CG", 1)

    u = Expression(("x[0] + 2*x[1] + 3*x[2]", "x[2] * 4*x[0]", "x[1]"), degree=1)
    uc = interpolate(u, Vc)
    uf = interpolate(u, Vf)

    mat = create_transfer_matrix(Vc, Vf).mat()
    Vuc = Function(Vf)
    mat.mult(vec(uc), vec(Vuc))
    as_backend_type(Vuc.vector()).update_ghost_values()

    diff = Function(Vf)
    diff.assign(Vuc - uf)
    assert diff.vector().norm("l2") < 1.0e-12
