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

        dolfin.set_log_level(dolfin.ERROR)
        sys.modules['backend'] = dolfin

        dolfin.parameters["form_compiler"]["representation"] = "uflacs"
        dolfin.parameters["form_compiler"]["optimize"]     = True
        dolfin.parameters["form_compiler"]["cpp_optimize"] = True
        dolfin.parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"

        # I have to *force* DOLFIN to initialise PETSc.
        # Otherwise, it will do it in the workers, using COMM_WORLD,
        # and deadlock. Yikes.
        dolfin.PETScOptions.set("-dummy", 1)
        dolfin.PETScOptions.clear("-dummy")

        # PETSc has recently implemented a new divergence tolerance,
        # which regularly breaks my deflation code. Disable it.
        dolfin.PETScOptions.set("snes_divergence_tolerance", -1)

    elif use_firedrake:
        # firedrake imported, no dolfin
        import firedrake
        sys.modules['backend'] = firedrake
        import backend

        firedrake.parameters["pyop2_options"]["lazy_evaluation"] = False

        from firedrake.petsc import PETSc
        opts = PETSc.Options()
        opts.setValue("snes_divergence_tolerance", -1)

        # firedrake lacks a HDF5File, let's make one

        from backend import Function
        from firedrake.petsc import PETSc
        from pyop2.mpi import COMM_WORLD, dup_comm, free_comm
        from firedrake import hdf5interface as h5i
        import numpy as np
        import os, os.path

        FILE_READ = PETSc.Viewer.Mode.READ
        FILE_WRITE = PETSc.Viewer.Mode.WRITE
        FILE_UPDATE = PETSc.Viewer.Mode.APPEND

        class HDF5File(object):

            """An object to mimic the DOLFIN HDF5File class.

            This checkpoint object is capable of writing :class:`~.Function`\s
            to disk in parallel (using HDF5) and reloading them on the same
            number of processes and a :func:`~.Mesh` constructed identically.

            :arg basename: the base name of the checkpoint file.
            :arg single_file: Should the checkpoint object use only a single
                 on-disk file (irrespective of the number of stored
                 timesteps)?  See :meth:`~.DumbCheckpoint.new_file` for more
                 details.
            :arg mode: the access mode (one of :data:`~.FILE_READ`,
                 :data:`~.FILE_CREATE`, or :data:`~.FILE_UPDATE`)
            :arg comm: (optional) communicator the writes should be collective
                 over.

            This object can be used in a context manager (in which case it
            closes the file when the scope is exited).

            .. note::

               This object contains both a PETSc ``Viewer``, used for storing
               and loading :class:`~.Function` data, and an :class:`h5py:File`
               opened on the same file handle.  *DO NOT* call
               :meth:`h5py:File.close` on the latter, this will cause
               breakages.

            """
            def __init__(self, comm, filename, mode):
                self.comm = dup_comm(comm or COMM_WORLD)

                if mode == 'r':
                    self.mode = FILE_READ
                elif mode == 'w':
                    self.mode = FILE_WRITE
                elif mode == 'a':
                    self.mode = FILE_UPDATE

                self._single = True
                self._made_file = False
                self._filename = filename
                self._fidx = 0
                self.new_file()

                print "Opened filename %s with mode %s." % (filename, mode)

            def new_file(self):
                """Open a new on-disk file for writing checkpoint data.

                :arg name: An optional name to use for the file, an extension
                     of ``.h5`` is automatically appended.

                If ``name`` is not provided, a filename is generated from the
                ``basename`` used when creating the :class:`~.DumbCheckpoint`
                object.  If ``single_file`` is ``True``, then we write to
                ``BASENAME.h5`` otherwise, each time
                :meth:`~.DumbCheckpoint.new_file` is called, we create a new
                file with an increasing index.  In this case the files created
                are::

                    BASENAME_0.h5
                    BASENAME_1.h5
                    ...
                    BASENAME_n.h5

                with the index incremented on each invocation of
                :meth:`~.DumbCheckpoint.new_file` (whenever the custom name is
                not provided).
                """
                self.close()
                name = self._filename

                import os
                exists = os.path.exists(name)
                if self.mode == FILE_READ and not exists:
                    raise IOError("File '%s' does not exist, cannot be opened for reading" % name)
                mode = self.mode
                if mode == FILE_UPDATE and not exists:
                    mode = FILE_CREATE

                # Create the directory if necessary
                dirname = os.path.dirname(name)
                try:
                    os.makedirs(dirname)
                except OSError:
                    pass

                self._vwr = PETSc.ViewerHDF5().create(name, mode=mode,
                                                      comm=self.comm)
                if self.mode == FILE_READ:
                    nprocs = self.attributes('/')['nprocs']
                    if nprocs != self.comm.size:
                        raise ValueError("Process mismatch: written on %d, have %d" %
                                         (nprocs, self.comm.size))
                else:
                    self.attributes('/')['nprocs'] = self.comm.size

            @property
            def vwr(self):
                """The PETSc Viewer used to store and load function data."""
                if hasattr(self, '_vwr'):
                    return self._vwr
                self.new_file()
                return self._vwr

            @property
            def h5file(self):
                """An h5py File object pointing at the open file handle."""
                if hasattr(self, '_h5file'):
                    return self._h5file
                self._h5file = h5i.get_h5py_file(self.vwr)
                return self._h5file

            def close(self):
                """Close the checkpoint file (flushing any pending writes)"""
                if hasattr(self, "_vwr"):
                    self._vwr.destroy()
                    del self._vwr
                if hasattr(self, "_h5file"):
                    self._h5file.flush()
                    del self._h5file

            def flush(self):
                """Flush any pending writes."""
                self._h5file.flush()

            def write(self, function, path):
                """Store a function in the checkpoint file.

                :arg function: The function to store.
                :arg path: the path to store the function under.
                """
                if self.mode is FILE_READ:
                    raise IOError("Cannot store to checkpoint opened with mode 'FILE_READ'")
                if not isinstance(function, backend.Function):
                    raise ValueError("Can only store functions")
                group = os.path.dirname(path)
                name  = os.path.basename(path)
                with function.dat.vec_ro as v:
                    self.vwr.pushGroup(group)
                    oname = v.getName()
                    v.setName(name)
                    v.view(self.vwr)
                    v.setName(oname)
                    self.vwr.popGroup()

            def read(self, function, path):
                """Store a function from the checkpoint file.

                :arg function: The function to load values into.
                :arg path: the path under which the function is stored.
                """
                if not isinstance(function, backend.Function):
                    raise ValueError("Can only load functions")
                group = os.path.dirname(path)
                name  = os.path.basename(path)

                with function.dat.vec as v:
                    self.vwr.pushGroup(group)
                    oname = v.getName()
                    v.setName(name)
                    v.load(self.vwr)
                    v.setName(oname)
                    self.vwr.popGroup()

            def attributes(self, obj):
                return self.h5file[obj].attrs

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

            def __del__(self):
                self.close()
                if hasattr(self, "comm"):
                    free_comm(self.comm)
                    del self.comm

        backend.HDF5File = HDF5File

