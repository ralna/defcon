"""
A module that implements the I/O backend for deflated continuation.

FIXME: I've tried to write this in a modular way so that it is possible to
implement more efficient/scalable backends at a later time.
"""

import backend
if backend.__name__ == "dolfin":
    from backend import HDF5File, Function
elif backend.__name__ == "firedrake":
    from backend import Function
    from firedrake.petsc import PETSc
    from pyop2.mpi import COMM_WORLD, dup_comm, free_comm
    from firedrake import hdf5interface as h5i
    import numpy as np
    import os.path

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

from parametertools import parameterstostring

import os
import glob
import time
import numpy as np
from ast import literal_eval

class IO(object):
    """
    Base class for I/O implementations.
    """

    def __init__(self, directory):
        self.directory = directory

    def setup(self, parameters, functionals, function_space):
        self.parameters = parameters
        self.functionals = functionals
        self.function_space = function_space

    def save_solution(self, solution, params, branchid):
        raise NotImplementedError

    def fetch_solutions(self, params, branchids):
        raise NotImplementedError

    def fetch_functionals(self, params, branchids):
        raise NotImplementedError

    def known_branches(self, params):
        raise NotImplementedError

    def known_parameters(self, fixed, branchid):
        raise NotImplementedError

    def max_branch(self):
        raise NotImplementedError

class FileIO(IO):
    """ I/O Module that uses HDF5 files to store the solutions and functionals. 
        We create one HDF5 file for each parameter value, with groups that contain the solutions for each branch.
        The file has the form: f = self.directory/params-(x1=...).hdf5/solution-0, solution-1, ...
        The functionals are stored as attributes for the file, so f.attrs.keys() = [functional-0, functional-1, ...]
    """

    def __init__(self, directory):
        self.directory = directory

        # Create the output directory.
        try: os.mkdir(directory) 
        except OSError: pass

    def dir(self, branchid):
        return self.directory + os.path.sep + "branch-%s.hdf5" % branchid

    def known_params_file(self, branchid, params, mode):
	# Records the existence of a solution with branchid for params.
        g = file(self.directory + os.path.sep + "branch-%s.txt" % branchid, mode)
        g.write(str(params)+';')
        g.flush()
        g.close()

    def save_solution(self, solution, funcs, params, branchid):
        """ Save a solution to the file branch-branchid.hdf5. """
        # Urgh... we need to check if the file already exists to decide if we use write mode or append mode. HDF5File's 'a' mode fails if the file doesn't exist.
        # This behaviour is different from h5py's 'a' mode, which can create a file if it doesn't exist and modify otherwise.
        if os.path.exists(self.dir(branchid)): mode='a'
        else: mode = 'w'

        with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(branchid), mode) as f:
            # First save the solution.
            f.write(solution, "/" + parameterstostring(self.parameters, params))

            # Now save the functionals.
            s = parameterstostring(self.functionals, funcs)
            f.attributes(parameterstostring(self.parameters, params))["functional"] = s

            # Flush and save the file.
            f.flush()

        # Make a note that we've discovered a solution on this branch and for these parameters. 
        if os.path.exists(self.directory + os.path.sep + "branch-%s.txt" % branchid): mode='a'
        else: mode = 'w'

        self.known_params_file(branchid, params, mode)
     
    def fetch_solutions(self, params, branchids):
        """ Fetch solutions for a particular parameter value for each branchid in branchids. """
        solns = []
        for branchid in branchids:
            failcount = 0
            while True:
                try:
                    with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(branchid), 'r') as f:
                        soln = Function(self.function_space)
                        f.read(soln, "/" + parameterstostring(self.parameters, params))
                        f.flush()
                    solns.append(soln)
                    break
  
                # We failed to open/read the file. Shouldn't happen, but just in case.
                except Exception:
                    print "Loading file %s failed. Sleeping for a second and trying again." % self.dir(branchid)
                    failcount += 1
                    if failcount == 5:
                        print "Argh. Tried 5 times to load file %s. Raising exception." % self.dir(branchid)
                        raise
                    time.sleep(1)
        return solns

    def known_branches(self, params):
        """ Load the branches we know about for this particular parameter value, by opening the file and seeing which groups it has. """
        saved_branch_files = glob.glob(self.directory + os.path.sep + "*.txt")
        branches = []
        for branch_file in saved_branch_files:
            pullData = open(branch_file, 'r').read().split(';')
            if str(params) in pullData: 
                branches.append(int(branch_file.split('-')[-1].split('.')[0]))
        return set(branches)
     
    def fetch_functionals(self, params, branchid):
        """ Gets the functionals back. Output [[all functionals...]]. """
        funcs = []
        with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(branchid), 'r') as f:
            for param in params: 
                newfuncs = [float(line.split('=')[-1]) for line in f.attributes(parameterstostring(self.parameters, param))["functional"].split('@')]
                funcs.append(newfuncs)
        return funcs

    def known_parameters(self, fixed, branchid):
        """ Returns a list of known parameters for a given branch. """
        fixed_indices = []
        fixed_values = []
        for key in fixed:
            fixed_values.append(fixed[key])
            # find the index
            for (i, param) in enumerate(self.parameters):
                if param[1] == key:
                    fixed_indices.append(i)
                    break

        seen = []

        pullData = open(self.directory + os.path.sep + "branch-%s.txt" % branchid, 'r').read().split(';')[0:-1]
        saved_params = [tuple([float(param) for param in literal_eval(params)]) for params in pullData]
        
        for param in saved_params:
            should_add = True
            for (index, value) in zip(fixed_indices, fixed_values):
                if param[index] != value:
                    should_add = False
                    break

            if should_add:
                seen.append(param)

        return seen

    def max_branch(self):
        saved_branch_files = glob.glob(self.directory + os.path.sep + "*.hdf5")
        branchids = [int(branch_file.split('-')[-1].split('.')[0]) for branch_file in saved_branch_files]
        return max(branchids)

