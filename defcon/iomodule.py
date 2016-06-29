"""
A module that implements the I/O backend for deflated continuation.

FIXME: I've tried to write this in a modular way so that it is possible to
implement more efficient/scalable backends at a later time.
"""

from dolfin import *
from parametertools import parameterstostring

import os
import glob
import time
import numpy as np

class IO(object):
    """
    Base class for I/O implementations.
    """

    def setup(self, parameters, functionals, function_space):
        self.parameters = parameters
        self.functionals = functionals
        self.function_space = function_space

    def save_solution(self, solution, params, branchid):
        raise NotImplementedError

    def fetch_solutions(self, params, branchids):
        raise NotImplementedError

    def delete_solutions(self):
        raise NotImplementedError

    def save_functionals(self, functionals, params, branchid):
        raise NotImplementedError

    def fetch_functionals(self, params, branchids):
        raise NotImplementedError

    def known_branches(self, params):
        raise NotImplementedError

    def known_parameters(self, fixed):
        raise NotImplementedError

    def max_branch(self):
        raise NotImplementedError



class FileIO(IO):
    def __init__(self, directory):
        self.directory = directory

    def dir(self, params):
        return self.directory + os.path.sep + parameterstostring(self.parameters, params) + os.path.sep

    def save_solution(self, solution, params, branchid):
        f = HDF5File(self.function_space.mesh().mpi_comm(), self.dir(params) + "solution-%d.hdf5" % branchid, 'w')
        f.write(solution, self.dir(params) + "solution-%d.hdf5" % branchid)
        f.close()
        del f

        # wait for the file to be written
        size = 0
        while True:
            try:
                size = os.stat(self.dir(params) + "solution-%d.hdf5" % branchid).st_size
            except OSError:
                pass
            if size > 0: break
            #print "Waiting for %s to have nonzero size" % (self.dir(params) + "solution-%d.xml.gz" % branchid)
            time.sleep(0.1)

    def fetch_solutions(self, params, branchids):
        solns = []
        for branchid in branchids:
            filename = self.dir(params) + "solution-%d.hdf5" % branchid
            failcount = 0
            while True:
                try:
                    soln = Function(self.function_space)
                    f = HDF5File(self.function_space.mesh().mpi_comm(), filename, 'r')
                    f.read(soln, filename)
                    f.close()
                    del f
                    break
                except Exception:
                    print "WTF? Loading file %s failed. Sleeping for a second and trying again." % filename
                    failcount += 1
                    if failcount == 5:
                        print "Argh. Tried 5 times to load file %s. Raising exception." % filename
                        raise
                    time.sleep(1)

            solns.append(soln)
        return solns

    def known_branches(self, params):
        filenames = glob.glob(self.dir(params) + "solution-*.hdf5")
        branches = [int(filename.split('-')[-1][:-5]) for filename in filenames] # The -5 is because ".hdf5" has 5 chars. Different filenames mean changing this value. 
        return set(branches)

    def save_functionals(self, funcs, params, branchid):
        f = file(self.dir(params) + "functional-%d.txt" % branchid, "w")
        s = parameterstostring(self.functionals, funcs).replace('@', '\n') + '\n'
        f.write(s)

    def fetch_functionals(self, params, branchids):
        funcs = []
        for branchid in branchids:
            f = file(self.dir(params) + "functional-%d.txt" % branchid, "r")
            func = []
            for line in f:
                func.append(float(line.split('=')[-1]))
            funcs.append(func)
        return funcs

    def known_parameters(self, fixed):

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
        saved_param_dirs = glob.glob(self.directory + "/*")
        saved_params = [tuple([float(x.split('=')[-1]) for x in dirname.split('/')[-1].split('@')]) for dirname in saved_param_dirs]

        for param in saved_params:
            should_add = True
            for (index, value) in zip(fixed_indices, fixed_values):
                if param[index] != value:
                    should_add = False
                    break

            if should_add:
                seen.add(param)

        return seen

    def max_branch(self):
        filenames = glob.glob(self.directory + "/*/solution-*.hdf5")
        branches = [int(filename.split('-')[-1][:-5]) for filename in filenames]
        return max(branches)


class DictionaryIO(IO):
    """
    I/O module using dictionaries to store the solutions and functionals. 
    TODO: Solutions are also written to the disk when saved, in HDF5 format.
    """
    #FIXME: Gives different solutions to the FileIO. Why????
    def __init__(self, nparams):
        self.nparams = nparams
        self.sols = dict()
        self.funcs = dict()
        
    def save_solution(self, solution, params, branchid):
        print solution
        if branchid in self.sols:
            self.sols[branchid][params] = solution
        else:
            self.sols[branchid] = {params:solution}
            
    def fetch_solutions(self, params, branchids):
        #FIXME: Send solutions one at a time. Perhaps use an iterator?
        solns = []
        for branchid in branchids:
            try:
                soln =  self.sols[branchid][params]
                solns.append(soln)
                break
            except Exception:
                raise
        return solns

    def delete_solutions(self):
        raise NotImplementedError

    def save_functionals(self, functionals, params, branchid):
        if branchid in self.funcs:
            self.funcs[branchid][params] = functionals
        else:
            self.funcs[branchid] = {params:functionals}

    def fetch_functionals(self, params, branchids):
        funcns = []
        for branchid in branchids:
            funcns.append(self.funcs[branchid][params])
        return funcns

    def delete_functionals(self):
        raise NotImplementedError

    def known_branches(self, params):
        branchids = []
        for key in self.sols.keys():
            if params in self.sols[key]:
                branchids.append(key)
        return set(branchids)

    def known_parameters(self, fixed):
        # Duplicated from FileIO. Problems?
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
        saved_params = [x for x in self.sols[key].keys() for key in self.sols.keys()]

        for param in saved_params:
            should_add = True
            for (index, value) in zip(fixed_indices, fixed_values):
                if param[index] != value:
                    should_add = False
                    break

            if should_add:
                seen.add(param)

        return seen

    def max_branch(self):
        return len(self.sols)-1 # -1 as branch labelling starts from 0. 
