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
import atexit
import collections

class IO(object):
    """
    Base class for I/O implementations.
    """

    def __init__(self, directory):
        self.directory = directory
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            pass

        tmpdir = os.path.abspath("tmp")
        try:
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)
        except OSError:
            pass
        self.tmpdir = tmpdir

        self.worldcomm = backend.comm_world

        atexit.register(self.cleanup)

    def cleanup(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def parameter_map(self):
        """
        Return an object that stores the map of parameter tuple -> list of known branches.
        """
        return collections.defaultdict(list)

    def close_parameter_map(self):
        pass

    def construct(self, worldcomm):
        self.worldcomm = worldcomm

    def setup(self, parameters, functionals, function_space):
        self.parameters = parameters
        self.functionals = functionals
        self.function_space = function_space

        # Argh, why do we need two communicators, from two libraries,
        # written by the same person ... ?
        if function_space is not None:
            # petsc4py comm
            self.pcomm = PETSc.Comm(function_space.mesh().mpi_comm())
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

class SolutionIO(IO):
    """An I/O class that saves one HDF5File per solution found."""
    def dir(self, params):
        return self.directory + os.path.sep + parameters_to_string(self.parameters, params) + os.path.sep

    def save_solution(self, solution, funcs, params, branchid):
        try:
            os.makedirs(self.dir(params))
        except OSError:
            pass

        tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=self.tmpdir, suffix=".h5")
        tmp.close()
        with HDF5File(self.function_space.mesh().mpi_comm(), tmp.name, 'w') as f:
            f.write(solution, "/solution")
        os.rename(tmp.name, self.dir(params) + "solution-%d.h5" % branchid)
        assert os.path.exists(self.dir(params) + "solution-%d.h5" % branchid)

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
