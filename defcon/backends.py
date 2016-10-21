def import_backend():
    import sys

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

        # Check dolfin version
        if dolfin.__version__.startswith("1") or dolfin.__version__.startswith("2016.1.0"):
            raise ImportError("Your version of DOLFIN is too old. DEFCON needs the development version of DOLFIN, 2016.2.0+.")

        dolfin.set_log_level(dolfin.ERROR)
        sys.modules['backend'] = dolfin

        dolfin.parameters["form_compiler"]["representation"] = "uflacs"
        dolfin.parameters["form_compiler"]["optimize"]     = True
        dolfin.parameters["form_compiler"]["cpp_optimize"] = True
        dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

        # I have to *force* DOLFIN to initialise PETSc.
        # Otherwise, it will do it in the workers, using COMM_WORLD,
        # and deadlock. Yikes.
        dolfin.PETScOptions.set("dummy", 1)
        dolfin.PETScOptions.clear("dummy")

    elif use_firedrake:
        # firedrake imported, no dolfin
        import firedrake
        sys.modules['backend'] = firedrake

        firedrake.parameters["assembly_cache"]["enabled"] = False
        firedrake.parameters["pyop2_options"]["lazy_evaluation"] = False
