from __future__ import absolute_import, print_function

"""
A module that implements the I/O backend for deflated continuation.

FIXME: I've tried to write this in a modular way so that it is possible to
implement more efficient/scalable backends at a later time.
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import json
import tempfile
import shutil
import sys
import os
import glob
import time
from ast import literal_eval
import atexit
import collections

from defcon.parametertools import parameters_to_string
import defcon.backend as backend
from defcon.backend import HDF5File, Function, File


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
        self.made_tmp = False
        try:
            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)
                self.made_tmp = True
        except OSError:
            pass
        self.tmpdir = tmpdir

        self.worldcomm = backend.comm_world

        atexit.register(self.cleanup)

    def cleanup(self):
        if self.made_tmp:
            shutil.rmtree(self.tmpdir, ignore_errors=True)

    def log(self, msg, warning=False):
        if self.pcomm.rank != 0: return

        if warning:
            fmt = RED = "\033[1;37;31m%s\033[0m\n"
        else:
            fmt = GREEN = "\033[1;37;32m%s\033[0m\n"

        timestamp = "[%s] " % time.strftime("%H:%M:%S")

        if warning:
            sys.stderr.write(fmt % (timestamp + msg))
            sys.stderr.flush()
        else:
            sys.stdout.write(fmt % (timestamp + msg))
            sys.stdout.flush()

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

        tmpname = os.path.join(self.tmpdir, 'solution-%d.h5' % branchid)
        with HDF5File(comm=self.function_space.mesh().mpi_comm(), filename=tmpname, file_mode='w') as f:
            f.write(solution, "/solution")
        if self.pcomm.rank == 0:
            os.rename(tmpname, self.dir(params) + "solution-%d.h5" % branchid)

            f = open(self.dir(params) + "functional-%d.txt" % branchid, "w")
            s = parameters_to_string(self.functionals, funcs).replace('@', '\n') + '\n'
            f.write(s)

        self.pcomm.Barrier()
        assert os.path.exists(self.dir(params) + "solution-%d.h5" % branchid)

    def fetch_solutions(self, params, branchids):
        solns = []
        for branchid in branchids:
            filename = self.dir(params) + "solution-%d.h5" % branchid
            failcount = 0
            while True:
                try:
                    with HDF5File(comm=self.function_space.mesh().mpi_comm(), filename=filename, file_mode='r') as f:
                        soln = Function(self.function_space)
                        f.read(soln, "/solution")
                        f.flush()
                    break
                except Exception:
                    print("Loading file %s failed. Sleeping for a second and trying again." % filename)
                    failcount += 1
                    if failcount == 10:
                        print("Tried 10 times to load file %s. Raising exception." % filename)
                        raise
                    time.sleep(1)

            solns.append(soln)
        return solns

    def fetch_functionals(self, params, branchid):
        funcs = []
        for param in params:
            f = open(self.dir(param) + "functional-%d.txt" % branchid, "r")
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
        saved_params = [tuple([float(x.split('=')[-1]) for x in filename.split('/')[-2].split('@')]) for filename in filenames]

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
            with HDF5File(comm=self.function_space.mesh().mpi_comm(), filename=(self.dir(params) + "eigenfunctions-%d.h5" % branchid), file_mode='w') as f:
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
                #print("Waiting for %s to have nonzero size" % (self.dir(params) + "solution-%d.xml.gz" % branchid))
                time.sleep(0.1)

        # Trick from Lawrence Mitchell: POSIX guarantees that mv is atomic
        f = tempfile.NamedTemporaryFile("w", delete=False, dir=self.tmpdir)
        s = str(stable)
        f.write(s)
        os.fsync(f.file.fileno())
        f.close()
        filename = self.dir(params) + "stability-%d.txt" % branchid
        if self.pcomm.rank == 0:
            os.rename(f.name, filename)

    def save_arclength(self, params, freeindex, branchid, ds, sign, data):
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
        if self.pcomm.rank == 0:
            os.rename(f.name, self.directory + os.path.sep + "arclength/params-%s-freeindex-%s-branchid-%s-ds-%.14e-sign-%d.json" % (parameters_to_string(self.parameters, params), freeindex, branchid, ds, sign))

    def fetch_stability(self, params, branchids,fetch_eigenfunctions=False):
        stabs = []
        for branchid in branchids:
            stab = {}
            filename = self.dir(params) + "eigenfunctions-%d.h5" % branchid
            fs = open(self.dir(params) + "stability-%d.txt" % branchid, "r")
            is_stable = literal_eval(fs.read())
            stab["stable"] =  is_stable
            stab["hint"] = None
            failcount=0
            evals = []
            eigfs = []
            while True:
                try:
                    with HDF5File(comm=self.function_space.mesh().mpi_comm(), filename=filename, file_mode='r') as f:
                        # Get Number of Eigenvalues 
                        num_evals= f.attributes('/eigenfunction-0')['number_eigenvalues']
                        #  create function space for eigenfunctions
                        efunc = Function(self.function_space)
                        # Iterate through each eigenvalue and 
                        for i in range(num_evals):
                            eigval=f.attributes("/eigenfunction-%d" % i)['eigenvalue']
                            evals.append(eigval)
                            if fetch_eigenfunctions:
                                f.read(efunc, "/eigenfunction-%d" % i)
                                f.flush()
                                eigfs.append(efunc)
                    break
                except Exception:
                    print("Loading file %s failed. Sleeping for a second and trying again." % filename)
                    failcount += 1
                    if failcount == 10:
                        print("Tried 10 times to load file %s. Raising exception." % filename)
                        raise
                    time.sleep(1)
            
            stab["eigenvalues"] = evals
            stab["eigenfunctions"] = eigfs
            stabs.append(stab)
        return stabs

# Some code to remap C- and Python-level stdout/stderr
def remap_c_streams(stdout_filename, stderr_filename):
    import sys
    import os

    if os.path.isfile(stdout_filename):
        os.remove(stdout_filename)
    if os.path.isfile(stderr_filename):
        os.remove(stderr_filename)

    new_stdout = open(stdout_filename, "a+")
    new_stderr = open(stderr_filename, "a+")

    os.dup2(new_stdout.fileno(), sys.stdout.fileno())
    os.dup2(new_stderr.fileno(), sys.stderr.fileno())

    sys.stdout = new_stdout
    sys.stderr = new_stderr
