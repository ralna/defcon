"""
A module that implements the I/O backend for deflated continuation.

FIXME: I've tried to write this in a modular way so that it is possible to
implement more efficient/scalable backends at a later time.
"""

import backend
import json
import tempfile
import shutil

from backend import HDF5File, Function, File
from parametertools import parameters_to_string

import os
import glob
import time
import numpy as np
from ast import literal_eval
from mpi4py import MPI
from petsc4py import PETSc
import shutil

class IO(object):
    """
    Base class for I/O implementations.
    """

    def __init__(self, directory):
        self.directory = directory

        tmpdir = "tmp"
        try:
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)
        except OSError:
            pass
        self.tmpdir = tmpdir

    def __del__(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def construct(self, comm):
        pass

    def setup(self, parameters, functionals, function_space):
        self.parameters = parameters
        self.functionals = functionals
        self.function_space = function_space

        # Argh, why do we need two communicators, from two libraries,
        # written by the same person ... ?
        if function_space is not None:
            # petsc4py comm
            self.pcomm = function_space.mesh().mpi_comm()
            self.mcomm = self.pcomm.tompi4py()
        else:
            self.mcomm = MPI.COMM_SELF
            self.pcomm = PETSc.Comm(self.mcomm)

    def clear(self):
        if os.path.exists(self.directory):
            tmpd = tempfile.mkdtemp(dir=os.getcwd())
            shutil.move(self.directory, tmpd + os.path.sep + self.directory)
            shutil.rmtree(tmpd, ignore_errors=True)

    def save_solution(self, solution, funcs, params, branchid):
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

    def save_arclength(self, params, branchid, ds, data):
        raise NotImplementedError

    def save_stability(self, stable, eigenvalues, eigenfunctions, params, branchid):
        raise NotImplementedError

    def fetch_stability(self, params, branchids):
        raise NotImplementedError

class BranchIO(IO):
    """ 
    I/O module that uses one HDF5File per branch.
    """

    def __init__(self, directory):
        self.directory = directory

        # Create the output directory.
        try: os.mkdir(directory) 
        except OSError: pass

    def dir(self, branchid):
        return self.directory + os.path.sep + "branch-%s.hdf5" % branchid

    def known_params_file(self, branchid, params, mode):
        """ Records the existence of a solution with branchid for params. """
        g = file(self.directory + os.path.sep + "branch-%s.txt" % branchid, mode)
        g.write(str(params)+';')
        g.close()

    def save_solution(self, solution, funcs, params, branchid):
        """ Save a solution to the file branch-branchid.hdf5. Also save the functionals as an attribute in the file."""
        # Urgh... we need to check if the file already exists to decide if we use write mode or append mode. HDF5File's 'a' mode fails if the file doesn't exist.
        # This behaviour is different from h5py's 'a' mode, which can create a file if it doesn't exist and modify otherwise.
        if os.path.exists(self.dir(branchid)): mode='a'
        else: mode = 'w'

        with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(branchid), mode) as f:
            # First save the solution.
            f.write(solution, "/" + parameters_to_string(self.parameters, params))

            # Now save the functionals.
            s = parameters_to_string(self.functionals, funcs)
            f.attributes(parameters_to_string(self.parameters, params))["functional"] = s

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
                        f.read(soln, "/" + parameters_to_string(self.parameters, params))
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
        """ Gets functionals for a particular branchid, one for each param in params. """
        funcs = []
        with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(branchid), 'r') as f:
            for param in params: 
                newfuncs = [float(line.split('=')[-1]) for line in f.attributes(parameters_to_string(self.parameters, param))["functional"].split('@')]
                funcs.append(newfuncs)
        return funcs

    def known_parameters(self, fixed, branchid):
        """ Returns a list of known parameters for a given branch. """
        pullData = open(self.directory + os.path.sep + "branch-%s.txt" % branchid, 'r').read().split(';')[0:-1]
        saved_params = [tuple([float(param) for param in literal_eval(params)]) for params in pullData]
        return saved_params

    def max_branch(self):
        """ Returns the index of the maximum branch we have found. """
        saved_branch_files = glob.glob(self.directory + os.path.sep + "*.hdf5")
        branchids = [int(branch_file.split('-')[-1].split('.')[0]) for branch_file in saved_branch_files]
        return max(branchids)

class SolutionIO(IO):
    """An I/O class that saves one HDF5File per solution found."""
    def dir(self, params):
        return self.directory + os.path.sep + parameters_to_string(self.parameters, params) + os.path.sep

    def save_solution(self, solution, funcs, params, branchid):
        with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(params) + "solution-%d.h5" % branchid, 'w') as f:
            f.write(solution, "/solution")

        # wait for the file to be written
        size = 0
        while True:
            try:
                size = os.stat(self.dir(params) + "solution-%d.h5" % branchid).st_size
            except OSError:
                pass
            if size > 0: break
            #print "Waiting for %s to have nonzero size" % (self.dir(params) + "solution-%d.xml.gz" % branchid)
            time.sleep(0.1)

        f = file(self.dir(params) + "functional-%d.txt" % branchid, "w")
        s = parameters_to_string(self.functionals, funcs).replace('@', '\n') + '\n'
        f.write(s)

    def fetch_solutions(self, params, branchids):
        solns = []
        for branchid in branchids:
            filename = self.dir(params) + "solution-%d.h5" % branchid
            failcount = 0
            while True:
                try:
                    with HDF5File(self.function_space.mesh().mpi_comm(), filename, 'r') as f:
                        soln = Function(self.function_space)
                        f.read(soln, "/solution")
                        f.flush()
                    break
                except Exception:
                    print "Loading file %s failed. Sleeping for a second and trying again." % filename
                    failcount += 1
                    if failcount == 10:
                        print "Tried 10 times to load file %s. Raising exception." % filename
                        raise
                    time.sleep(1)

            solns.append(soln)
        return solns

    def fetch_functionals(self, params, branchid):
        funcs = []
        for param in params:
            f = file(self.dir(param) + "functional-%d.txt" % branchid, "r")
            func = []
            for line in f:
                func.append(float(line.split('=')[-1]))
            funcs.append(func)
        return funcs

    def known_branches(self, params):
        filenames = glob.glob(self.dir(params) + "solution-*.h5")
        branches = [int(filename.split('-')[-1][:-3]) for filename in filenames]
        return set(branches)

    def known_parameters(self, fixed, branchid):
        fixed_indices = []
        fixed_values = []
        for key in fixed:
            fixed_values.append(fixed[key])
            # find the index
            for (i, param) in enumerate(self.parameters):
                if param[1] == key:
                    fixed_indices.append(i)
                    break

        seen = set()
        filenames = glob.glob(self.directory + "/*/solution-%d.h5" % branchid)
        saved_param_dirs = [x.replace("output", "").split('/')[1] for x in filenames]
        saved_params = [tuple([float(x.split('=')[-1]) for x in dirname.split('/')[-1].split('@')]) for dirname in saved_param_dirs]

        for param in saved_params:
            should_add = True
            for (index, value) in zip(fixed_indices, fixed_values):
                if param[index] != value:
                    should_add = False
                    break

            if should_add:
                seen.add(param)

        return sorted(list(seen))

    def max_branch(self):
        filenames = glob.glob(self.directory + "/*/solution-*.h5")
        branches = [int(filename.split('-')[-1][:-3]) for filename in filenames]
        return max(branches)

    def save_stability(self, stable, eigenvalues, eigenfunctions, params, branchid):
        assert len(eigenvalues) == len(eigenfunctions)

        if len(eigenvalues) > 0:
            with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(params) + "eigenfunctions-%d.h5" % branchid, 'w') as f:
                for (i, (eigval, eigfun)) in enumerate(zip(eigenvalues, eigenfunctions)):
                    f.write(eigfun, "/eigenfunction-%d" % i)
                    f.attributes("/eigenfunction-%d" % i)['eigenvalue'] = eigval

                f.attributes('/eigenfunction-0')['number_eigenvalues'] = len(eigenvalues)

            # wait for the file to be written
            size = 0
            while True:
                try:
                    size = os.stat(self.dir(params) + "eigenfunctions-%d.h5" % branchid).st_size
                except OSError:
                    pass
                if size > 0: break
                #print "Waiting for %s to have nonzero size" % (self.dir(params) + "solution-%d.xml.gz" % branchid)
                time.sleep(0.1)

        # Trick from Lawrence Mitchell: POSIX guarantees that mv is atomic
        f = tempfile.NamedTemporaryFile("w", delete=False, dir=self.tmpdir)
        s = str(stable)
        f.write(s)
        os.fsync(f.file.fileno())
        f.close()
        filename = self.dir(params) + "stability-%d.txt" % branchid
        os.rename(f.name, filename)

    def save_arclength(self, params, freeindex, branchid, ds, data):
        try:
            if not os.path.exists(self.directory + os.path.sep + "arclength"):
                os.makedirs(self.directory + os.path.sep + "arclength")
        except OSError:
            pass

        # Trick from Lawrence Mitchell: POSIX guarantees that mv is atomic
        f = tempfile.NamedTemporaryFile("w", delete=False, dir=self.tmpdir)
        json.dump(data, f.file, indent=4)
        f.file.flush()
        os.fsync(f.file.fileno())
        f.close()
        os.rename(f.name, self.directory + os.path.sep + "arclength/params-%s-freeindex-%s-branchid-%s-ds-%.14e.json" % (parameters_to_string(self.parameters, params), freeindex, branchid, ds))

    def fetch_stability(self, params, branchids):
        stables = []
        for branchid in branchids:
            f = file(self.dir(params) + "stability-%d.txt" % branchid, "r")
            stable = literal_eval(f.read())
            stables.append(stable)
        return stables

# Some code to remap C- and Python-level stdout/stderr
def remap_c_streams(stdout_filename, stderr_filename):
    import sys
    import ctypes

    if os.path.isfile(stdout_filename):
        os.remove(stdout_filename)
    if os.path.isfile(stderr_filename):
        os.remove(stderr_filename)

    sys.stdout = open(stdout_filename, "a+")
    sys.stderr = open(stderr_filename, "a+")

    try:
        libc = ctypes.CDLL("libc.so.6")
    except:
        return

    stdout_c = libc.fdopen(1, "w")
    libc.freopen(stdout_filename, "a+", stdout_c)
    stderr_c = libc.fdopen(2, "w")
    libc.freopen(stderr_filename, "a+", stderr_c)
