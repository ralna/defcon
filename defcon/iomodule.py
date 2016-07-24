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
from ast import literal_eval

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
        return self.directory + os.path.sep + "branch-%d.hdf5" % branchid

    def known_params_file(self, branchid, params, mode):
        g = file(self.directory + os.path.sep + "branch-%s.txt" % branchid, mode)
        g.write(str(params)+';')
        g.flush()
        g.close()
        
    def save_solution(self, solution, params, branchid):
        """ Save a solution to the file branch-branchid.hdf5. """
        # Urgh... we need to check if the file already exists to decide if we use write mode or append mode. HDF5File's 'a' mode fails if the file doesn't exist.
        # This behaviour is different from h5py's 'a' mode, which can create a file if it doesn't exist and modify otherwise.
        if os.path.exists(self.dir(branchid)): mode='a'
        else: mode = 'w'

        # Open file and write the solution. Flush afterwards to ensure it is written to disk. 
        with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(branchid), mode) as f:
            f.write(solution, "/%s" % parameterstostring(self.parameters, params))
            f.flush()
            f.close()

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
                        f.read(soln, parameterstostring(self.parameters, params))
                        f.flush()
                        f.close()
                    solns.append(soln)
                    break
  
                # We failed to open/read the file. Shouldn't happen, but just in case.
                except Exception:
                    print "WTF? Loading file %s failed. Sleeping for a second and trying again." % self.dir(params)
                    failcount += 1
                    if failcount == 5:
                        print "Argh. Tried 5 times to load file %s. Raising exception." % self.dir(params)
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
     
    def save_functionals(self, funcs, params, branchid):
        """ Stores the functionals as attribute 'functional-branchid' of the /solution-branchid group of the appropriate file. """
        s = parameterstostring(self.functionals, funcs)
        with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(branchid), 'a') as f:
            f.attributes(parameterstostring(self.parameters, params))["functional"] = s
            f.flush()
            f.close()

    def fetch_functionals(self, params, branchids):
        """ Gets the functionals back. Output [[all functionals...]]. """
        funcs = []
        for branchid in branchids:
            with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(branchid), 'r') as f:
                try: 
                    newfuncs = [float(line.split('=')[-1]) for line in f.attributes(parameterstostring(self.parameters, params))["functional"].split('@')]
                    funcs.append(newfuncs)
                    f.flush()
                except Exception: pass
            f.close()       
        return funcs

    def known_parameters(self, fixed):
        """ Returns a list of known parameters. """
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

        # FIXME: this doesn't quite work. 
        saved_branch_files = glob.glob(self.directory + os.path.sep + "*.txt")
        all_keys = []
        saved_params = []
        for branch_file in saved_branch_files:
            pullData = open(branch_file, 'r').read().split(';')
            all_params = []
            for param in pullData: 
                if len(param) > 0: all_params.append(param)
            saved_params += [tuple([float(param) for param in literal_eval(params)]) for params in all_params]
        

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
        saved_branch_files = glob.glob(self.directory + os.path.sep + "*.hdf5")
        branchids = [int(branch_file.split('-')[-1].split('.')[0]) for branch_file in saved_branch_files]
        return max(branchids)

