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
import h5py as h5

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
        self.maxbranch = -1 

        # File directorys aren't necessarily created automatically. FIXME: Find some way to remove this?
        try: os.mkdir(directory) 
        except OSError: pass

    def dir(self, params):
        """ Directories for storing functionals. """
        return self.directory + os.path.sep + parameterstostring(self.parameters, params)

    def save_solution(self, solution, params, branchid):
        # We create one HDF5 file for each parameter value, with groups that contain the solutions for each branch.
        # The file has the form: self.directory/params-(x1=...).hdf5/solution-0, solution-1, solution-2, etc.

        filename = self.dir(params) + ".hdf5"

        # Urgh... we need to check if the file already exists to decide if we use write mode or append mode...
        if os.path.exists(filename): mode='a'
        else: mode = 'w'

        with HDF5File(self.function_space.mesh().mpi_comm(), filename, mode) as f:
            f.write(solution, "/solution-%d" % branchid)
            f.flush()
     
        # This is potentially a new branch, so let's update the maxbranch variable if need be. 
        self.maxbranch = max(branchid, self.maxbranch)

    def fetch_solutions(self, params, branchids):
        solns = []
        for branchid in branchids:
            failcount = 0
            solns = []
            while True:
                # FIXME: For some crazy reason, this doesn't work if we remove the try/except clause, even though we never go in the except part. WTF?
                try:
                    filename = self.dir(params) + ".hdf5"
                    with HDF5File(self.function_space.mesh().mpi_comm(), filename, 'r') as f:
                        soln = Function(self.function_space)
                        f.read(soln, "solution-%d" % branchid)
                        f.flush()
                    solns.append(soln)
                    break
  
                except Exception:
                    print "WTF? Loading file %s failed. Sleeping for a second and trying again." % filename
                    failcount += 1
                    if failcount == 5:
                        print "Argh. Tried 5 times to load file %s. Raising exception." % filename
                        raise
                    time.sleep(1)

        return solns

    def known_branches(self, params):
        # Load the branches we know about for this particular parameter value, by opening the file and seeing which groups it has. 
        # FIXME: can we remove the except clause?? Maybe tidy up a bit, use the with/as clause for better protection. 
        try:
            f = h5.File(self.dir(params) + ".hdf5", 'r')
            names = f.keys()
            branches = [int(name.split('-')[-1]) for name in names]
            f.close()
        except Exception:
            branches = set([])
        return set(branches)
     
    #FIXME: incorporate the functionals into the data structure.   
    def save_functionals(self, funcs, params, branchid):
        # Urgh, we have to make the directory if it doesn't exist
        try:
            os.mkdir(self.dir(params))
        except OSError:
            pass

        f = file(self.dir(params) + os.path.sep + "functional-%d.txt" % branchid, "w")
        s = parameterstostring(self.functionals, funcs).replace('@', '\n') + '\n'
        f.write(s)

    def fetch_functionals(self, params, branchids):
        funcs = []
        for branchid in branchids:
            f = file(self.dir(params) + os.path.sep + "functional-%d.txt" % branchid, "r")
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

        saved_param_files = glob.glob(self.directory + "/*.hdf5")
        saved_params = [tuple([float(x.split('=')[-1]) for x in dirname[0:-5].split('/')[-1].split('@')]) for dirname in saved_param_files] 

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
        return self.maxbranch


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
