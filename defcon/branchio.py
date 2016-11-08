from backend import HDF5File, Function
import h5py # FIXME: remove dependency on h5py, eventually

import cPickle as pickle
import os, os.path
import iomodule
import glob
import collections
import sys
import time
import traceback
import tempfile

from ast import literal_eval
from numpy import array

# If you're storing one branch per hdf5 file, the
# hardest part is figuring out which parameters have which
# branches. I'm going to use a special HDF5 file for this
# purpose.

def paramstokey(params): return "(" + ", ".join("%.15e" % x for x in params) + ")"
def keytoparams(key): return literal_eval(key)

class ParameterMap(object):
    def __init__(self, directory):
        self.path = os.path.join(directory, "parameter_map.pck")
        try:
            f = open(self.path, "r")
            self.dict = pickle.load(f)
        except:
            self.dict = collections.defaultdict(list)

    def close(self):
        with open(self.path, "w") as f:
            pickle.dump(self.dict, f)

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value

class BranchIO(iomodule.SolutionIO):
    def __init__(self, directory):
        iomodule.SolutionIO.__init__(self, directory)
        self.pm = None

    def parameter_map(self):
        if self.pm is None:
            self.pm = ParameterMap(self.directory)
        return self.pm

    def close_parameter_map(self):
        self.pm.close()
        self.pm = None

    def known_branches(self, params):
        if self.worldcomm.rank != 0:
            raise ValueError("Can only be requested on master process")

        if self.pm is None:
            self.pm = ParameterMap(self.directory)
        known = self.pm[params]
        return known

    def solution_filename(self, branchid):
        return os.path.join(self.directory, "solution-" + str(branchid) + ".h5")

    def stability_filename(self, branchid):
        return os.path.join(self.directory, "stability-" + str(branchid) + ".h5")

    def save_solution(self, solution, funcs, params, branchid):
        key = paramstokey(params)
        fname = self.solution_filename(branchid)

        tmpfile = tempfile.NamedTemporaryFile("w", delete=False, dir=self.tmpdir)
        tmpfile.close()
        tmpname = tmpfile.name

        if os.path.exists(fname):
            mode = "a"
            exists = True

            # I can't believe I have to do this to get filesystem synchronisation right.
            os.rename(fname, tmpname)
        else:
            mode = "w"
            exists = False

        with HDF5File(self.pcomm, tmpname, mode) as f:
            if self.pcomm.size > 1:
                f.set_mpi_atomicity(True)
            f.write(solution, key + "/solution")

            attrs = f.attributes(key)
            for (func, value) in zip(self.functionals, funcs):
                attrs[func[1]] = value
        self.pcomm.Barrier()

        # Trick from Lawrence Mitchell: POSIX guarantees that mv is atomic
        os.rename(tmpname, fname)

        # Belt-and-braces: sleep until the path exists
        while not os.path.exists(fname):
            time.sleep(0.1)

        return

    def fetch_solutions(self, params, branchids):
        key = paramstokey(params)
        solns = []
        for branchid in branchids:
            failcount = 0
            filename = self.solution_filename(branchid)

            while True:
                try:
                    with HDF5File(self.pcomm, filename, "r") as f:
                        solution = Function(self.function_space)
                        f.read(solution, key + "/solution")
                        solns.append(solution)
                        break
                except Exception:
                    # This should never happen. I'm going for the belt-and-braces approach here.
                    self.log("Could not load %s from file %s; backtrace and exception follows" % (key, filename), warning=True)
                    traceback.print_stack()
                    traceback.print_exc()
                    failcount += 1
                    if failcount == 10:
                        self.log("Failed 10 times. Raising exception.", warning=True)
                        raise
                    time.sleep(1)

        return solns

    def fetch_functionals(self, params, branchid):
        funcs = []
        with HDF5File(self.pcomm, self.solution_filename(branchid), "r") as f:
            for param in params:
                key = paramstokey(param)

                attrs = f.attributes(key)

                thesefuncs = []
                for func in self.functionals:
                    thesefuncs.append(attrs[func[1]])
                funcs.append(thesefuncs)

        return funcs

    def known_parameters(self, fixed, branchid):
        out = []

        fixedindices = []
        fixedvalues  = []
        n = len(fixed)

        for (i, param) in enumerate(self.parameters):
            if param[1] in fixed:
                fixedindices.append(i)
                fixedvalues.append(fixed[param[1]])

        with h5py.File(self.solution_filename(branchid), "r") as f:
            for key in f['/']:
                params = keytoparams(key)
                add = True

                for i in range(n):
                    if params[fixedindices[i]] != fixedvalues[i]:
                        add = False
                        break
                if add:
                    out.append(params)

        return out

    def max_branch(self):
        filenames = glob.glob(os.path.join(self.directory, 'solution*.h5'))
        maxbranch = max(int(filename.split('-')[-1][:-3]) for filename in filenames)
        return maxbranch

    def pickle(self, obj):
        return pickle.dumps(obj, protocol=0)

    def unpickle(self, s):
        return pickle.loads(s)

    def save_stability(self, stable, eigenvalues, eigenfunctions, params, branchid):
        assert len(eigenvalues) == len(eigenfunctions)

        key = paramstokey(params)
        fname = self.stability_filename(branchid)
        if os.path.exists(fname):
            mode = "a"
            exists = True
        else:
            mode = "w"
            exists = False

        with HDF5File(self.pcomm, fname, mode) as f:
            if self.pcomm.size > 1:
                f.set_mpi_atomicity(True)

            # dummy dataset so that we can actually store the attribute
            f.write(array([0.0]), key + "/stability")
            attrs = f.attributes(key + "/stability")
            attrs["stable"] = self.pickle(stable)

            if len(eigenvalues) > 0:
                for (j, (eigval, eigfun)) in enumerate(zip(eigenvalues, eigenfunctions)):
                    ekey = key + "/stability/eigenfunction-%d" % j
                    f.write(eigfun, ekey)
                    f.attributes(ekey)["eigenvalue"] = eigval
        self.pcomm.barrier()

    def fetch_stability(self, params, branchids):
        stables = []
        key = paramstokey(params)

        for branchid in branchids:
            fname = self.stability_filename(branchid)
            with HDF5File(self.pcomm, fname, "r") as f:
                attrs = f.attributes(key + "/stability")
                if "stable" not in attrs:
                    msg = "Could not find stability information for %s in %s." % (params, fname)
                    raise KeyError(msg)
                stables.append(self.unpickle(attrs["stable"]))

        return stables
