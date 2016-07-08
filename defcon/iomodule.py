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

    def plot_to_file(self, x, y, branchid):
        raise NotImplementedError



class FileIO(IO):
    """ I/O Module that uses HDF5 files to store the solutions and functionals. 
        We create one HDF5 file for each parameter value, with groups that contain the solutions for each branch.
        The file has the form: f = self.directory/params-(x1=...).hdf5/solution-0, solution-1, ...
        The functionals are stored as attributes for the file, so f.attrs.keys() = [functional-0, functional-1, ...]
    """

    def __init__(self, directory):
        self.directory = directory
        self.maxbranch = -1 

        # Create the output directory.
        try: os.mkdir(directory) 
        except OSError: pass

    def dir(self, params):
        return self.directory + os.path.sep + parameterstostring(self.parameters, params) + ".hdf5"

    def save_solution(self, solution, params, branchid):
        """ Save a solution to the file params.hdf5. """
        # Urgh... we need to check if the file already exists to decide if we use write mode or append mode. HDF5File's 'a' mode fails if the file doesn't exist.
        # This behaviour is different from h5py's 'a' mode, which can create a file if it doesn't exist and modify otherwise.
        if os.path.exists(self.dir(params)): mode='a'
        else: mode = 'w'

        # Opne file and white the solution. Flush afterwards to ensure it is written to disk. 
        with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(params), mode) as f:
            f.write(solution, "/solution-%d" % branchid)
            f.flush()
     
        # This is potentially a new branch, so let's update the maxbranch file if need be.
        # FIXME: We need the file because for some reason the variable doesn't persist (we get -1 if we call from BifurcationDiagram). Figure out some other way to do this?
        if branchid > self.maxbranch:
            self.maxbranch = branchid
            g = file(self.directory + os.path.sep + "maxbranch", 'w')
            g.write(str(self.maxbranch))
            g.flush()
            g.close()

    def fetch_solutions(self, params, branchids):
        """ Fetch solutions for a particular parameter value for each branchid in branchids. """
        solns = []
        for branchid in branchids:
            failcount = 0
            while True:
                try:
                    with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(params), 'r') as f:
                        soln = Function(self.function_space)
                        f.read(soln, "solution-%d" % branchid)
                        f.flush()
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
        try:
            f = h5.File(self.dir(params), 'r')
            names = f.keys()
            branches = [int(name.split('-')[-1]) for name in names]
            f.close()
        except Exception:
            branches = []
        return set(branches)
     
    def save_functionals(self, funcs, params, branchid):
        """ Stores the functionals as attribute 'functional-branchid' of the /solution-branchid group of the appropriate file. """
        s = parameterstostring(self.functionals, funcs)
        with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(params), 'a') as f:
            f.attributes("/solution-%d" % branchid)["functional-%d" % branchid] = s
            f.flush()


    def fetch_functionals(self, params, branchids):
        """ Gets the functionals back. Output [[all functionals...]]. """
        funcs = []
        try:
            with HDF5File(self.function_space.mesh().mpi_comm(), self.dir(params), 'r') as f:
                for branchid in branchids:
                    newfuncs = [float(line.split('=')[-1]) for line in f.attributes("/solution-%d" % branchid)["functional-%d" % branchid].split('@')]
                    funcs.append(newfuncs)
        except Exception: raise
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

        # Pull up each file and then break them apart to get a list of tuples of diferent parameter values.
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
        """ Read the single number in the maxbranch file. """
        g = file(self.directory + os.path.sep + "maxbranch", 'r')
        s = int(g.read())
        g.close()
        return s

    def plot_to_file(self, x, y, branchid):
        """ Writes a pair of points to the file 'points_to_plot', so the external gui can read them in. 
            Points are written to 10 decimal places of accuracy. """
        g = file(self.directory + os.path.sep + "points_to_plot", 'a') # append mode so we don't overwrite the previous contents. 
        g.write("%.10f,%.10f,%d \n" % (x, y, branchid)) # change '.10' to alter the decimal precision.
        g.flush()
        g.close()
