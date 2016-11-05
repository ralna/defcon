from backend import HDF5File, Function
import h5py # FIXME: remove dependency on h5py, eventually
import cPickle as pickle
import os, os.path
import iomodule

# If you're storing one branch per hdf5 file, the
# hardest part is figuring out which parameters have which
# branches. I'm going to use a special HDF5 file for this
# purpose.
class ParameterMap(object):
    def __init__(self, directory, mode="a"):
        os.system("ls -l output/parameter_map.h5")
        self.h5 = h5py.File(os.path.join(directory, "parameter_map.h5"), mode, driver="core")

    def __getitem__(self, params):
        key = "(" + ", ".join("%.15e" % x for x in params) + ")"
        out = list(self.h5.attrs.get(key, []))
        return out

    def __setitem__(self, params, value):
        key = "(" + ", ".join("%.15e" % x for x in params) + ")"
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

#class BranchIO(iomodule.IO):
#    """
#    Use a single HDF5 file per branch.
#    """
#
#    def parameter_map(self):
#
#    def pickle(self, obj):
#        return pickle.dumps(obj, protocol=0)
#
#    def unpickle(self, s):
#        return pickle.loads(s)
#
#    def save_solution(self, solution, funcs, params, branchid):
#    def fetch_solutions(self, params, branchids):
#    def fetch_functionals(self, params, branchids):
#    def known_branches(self, params):
#    def known_parameters(self, fixed, branchid):
#        raise NotImplementedError
#
#    def max_branch(self):
#        raise NotImplementedError
#
#    def save_arclength(self, params, branchid, ds, data):
#        raise NotImplementedError
#
#    def save_stability(self, stable, eigenvalues, eigenfunctions, params, branchid):
#        raise NotImplementedError
#
#    def fetch_stability(self, params, branchids):
#        raise NotImplementedError

