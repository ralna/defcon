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

class FileIO(IO):
    def __init__(self, directory):
        self.directory = directory

    def dir(self, params):
        return self.directory + os.path.sep + parameterstostring(self.parameters, params) + os.path.sep

    def save_solution(self, solution, params, branchid):
        File(self.function_space.mesh().mpi_comm(), self.dir(params) + "solution-%d.xml.gz" % branchid) << solution
        assert os.stat(self.dir(params) + "solution-%d.xml.gz" % branchid).st_size > 0

    def fetch_solutions(self, params, branchids):
        solns = []
        for branchid in branchids:
            filename = self.dir(params) + "solution-%d.xml.gz" % branchid
            failcount = 0
            while True:
                try:
                    soln = Function(self.function_space, filename)
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
        filenames = glob.glob(self.dir(params) + "solution-*.xml.gz")
        branches = [int(filename.split('-')[-1][:-7]) for filename in filenames]
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