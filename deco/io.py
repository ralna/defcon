"""
A module that implements the I/O backend for deflated continuation.

I've tried to write this in a modular way so that it is possible to
implement more efficient/scalable backends at a later time.
"""

import os
import glob

from dolfin import *
from parametertools import parameterstostring

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
        File(self.dir(params) + "solution-%d.xml.gz" % branchid) << solution

    def fetch_solutions(self, params, branchids):
        solns = [Function(self.function_space, self.dir(params) + "solution-%d.xml.gz" % branchid) for branchid in branchids]

    def known_branches(self, params):
        filenames = glob.glob(self.dir(params) + "solution-*.xml.gz")
        branches = [int(filename.split('-')[-1][-7]) for filename in filenames]
        return set(branches)

    def save_functionals(self, funcs, params, branchid):
        f = file(self.dir(params) + "functional-%d.txt" % branchid, "w")
        s = parameterstostring(self.functionals, funcs).replace('-', '\n')
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
