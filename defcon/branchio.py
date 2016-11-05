from backend import HDF5File, Function
import h5py # FIXME: remove dependency on h5py, eventually
import cPickle as pickle
import os, os.path
import iomodule
import glob
from ast import literal_eval

# If you're storing one branch per hdf5 file, the
# hardest part is figuring out which parameters have which
# branches. I'm going to use a special HDF5 file for this
# purpose.

def paramstokey(params): return "(" + ", ".join("%.15e" % x for x in params) + ")"
def keytoparams(key): return literal_eval(key)

class ParameterMap(object):
    def __init__(self, directory, mode="a"):
        self.h5 = h5py.File(os.path.join(directory, "parameter_map.h5"), mode, driver="core")

    def __getitem__(self, params):
        key = paramstokey(params)
        out = list(self.h5.attrs.get(key, []))
        return out

    def __setitem__(self, params, value):
        key = paramstokey(params)
        self.h5.attrs[key] = value

class BranchIO(iomodule.SolutionIO):
    def __init__(self, directory):
        iomodule.SolutionIO.__init__(self, directory)
        self.pm = None

    def parameter_map(self):
        if self.pm is None:
            self.pm = ParameterMap(self.directory)
        return self.pm

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
        if os.path.exists(fname):
            mode = "a"
            exists = True
        else:
            mode = "w"
            exists = False

        with HDF5File(self.pcomm, fname, mode) as f:
            f.set_mpi_atomicity(True)
            f.write(solution, key + "/solution")

            attrs = f.attributes(key)
            for (func, value) in zip(self.functionals, funcs):
                attrs[func[1]] = value
        self.pcomm.Barrier()

    def fetch_solutions(self, params, branchids):
        key = paramstokey(params)
        solns = []
        for branchid in branchids:
            with HDF5File(self.pcomm, self.solution_filename(branchid), "r") as f:
                solution = Function(self.function_space)
                f.read(solution, key + "/solution")
                solns.append(solution)
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

        # FIXME: probably need atomic mode/an MPI barrier here?
        with HDF5File(self.pcomm, fname, mode) as f:
            attrs = f.attributes(key + "/stability")
            attrs["stable"] = self.pickle(stable)

            if len(eigenvalues) > 0:
                for (j, (eigval, eigfun)) in enumerate(zip(eigenvalues, eigenfunctions)):
                    ekey = key + "/stability/eigenfunction-%d" % j
                    f.write(eigfun, ekey)
                    f.attributes(ekey)["eigenvalue"] = eigval

    def fetch_stability(self, params, branchids):
        stables = []
        key = paramstokey(params)

        for branchid in branchids:
            fname = self.stability_filename(branchid)
            with HDF5File(self.pcomm, fname, "r") as f:
                attrs = f.attributes(key + "/stability")
                if "stable" not in attrs:
                    raise KeyError
                stables.append(self.unpickle(attrs["stable"]))

        return stables
