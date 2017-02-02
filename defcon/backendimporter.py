from __future__ import absolute_import

import sys

def import_backend():
    """Import a backend modulem, tweak its parameters and
    return it; currently either dolfin or firedrake."""

    use_dolfin = True
    use_firedrake = False

    if "dolfin" in sys.modules and "firedrake" not in sys.modules:
        use_dolfin = True

    elif "firedrake" in sys.modules and "dolfin" not in sys.modules:
        use_dolfin = False
        use_firedrake = True

    elif "firedrake" in sys.modules and "dolfin" in sys.modules:
        # both loaded, don't know what to do
        raise ImportError("Import exactly one of dolfin or firedrake before importing defcon.")

    else: # nothing loaded, default to DOLFIN
        use_dolfin = True

    if use_dolfin:
        import dolfin
        assert dolfin.has_petsc4py()

        dolfin.set_log_level(dolfin.ERROR)

        dolfin.parameters["form_compiler"]["representation"] = "uflacs"
        dolfin.parameters["form_compiler"]["optimize"]     = True
        dolfin.parameters["form_compiler"]["cpp_optimize"] = True
        dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

        # I have to *force* DOLFIN to initialise PETSc.
        # Otherwise, it will do it in the workers, using COMM_WORLD,
        # and deadlock. Yikes.
        dolfin.SubSystemsManager.init_petsc()

        # PETSc has recently implemented a new divergence tolerance,
        # which regularly breaks my deflation code. Disable it.
        dolfin.PETScOptions.set("snes_divergence_tolerance", -1)

        return dolfin

    elif use_firedrake:
        # firedrake imported, no dolfin
        import firedrake

        firedrake.parameters["pyop2_options"]["lazy_evaluation"] = False

        from firedrake.petsc import PETSc
        opts = PETSc.Options()
        opts.setValue("snes_divergence_tolerance", -1)

        return firedrake


backend = import_backend()

# Monkey-patch modules so that user can import from a backend
sys.modules['defcon.backend'] = backend
