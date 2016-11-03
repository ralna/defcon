from backend import HDF5File, Function
import h5py # FIXME: remove dependency on h5py
import cPickle as pickle
import os
import iomodule

class H5IO(iomodule.IO):
    """
    Use a single HDF5 file to store all solutions.
    """

    def __init__(self, directory):
        iomodule.IO.__init__(self, directory)
        self.db = self.directory + os.path.sep + "db.h5"

    def construct(self, mcomm):
        # Create the HDF5 file collectively, if it
        # doesn't yet exist

        if mcomm.rank == 0:
            try:
                os.makedirs(self.directory)
            except OSError:
                import traceback; traceback.print_exc()
                pass

        # FIXME: dolfin HDF5File 'a' semantics are different to h5py 'a' semantics
        # (which is what we want, pg 17 of O'Reilly book)
        h5 = h5py.File(self.db, "a", driver="mpio", comm=mcomm)
        del h5

    def groupname(self, values, branchid=None):
        s = ""
        for (param, value) in zip(self.parameters, values):
            s += "/%s=%.15e" % (param[1], value)

        if branchid is not None:
            s += "/branchid-%d" % branchid

        return s

    def pickle(self, obj):
        return pickle.dumps(obj, protocol=0)

    def unpickle(self, s):
        return pickle.loads(s)

    def save_solution(self, solution, funcs, params, branchid):
        with HDF5File(self.pcomm, self.db, "a") as h5:
            grp = self.groupname(params, branchid)

            # Save the solution
            h5.write(solution, grp + "/data")

            # Write the functional values as attributes
            for (func, value) in zip(self.functionals, funcs):
                h5.attributes(grp)[func[1]] = value

            # Set the default value for stability
            h5.attributes(grp)['stability'] = self.pickle(None)

    def fetch_solutions(self, params, branchids):
        out = []
        with HDF5File(self.pcomm, self.db, "r") as h5:
            for branchid in branchids:
                z = Function(self.function_space)
                grp = self.groupname(params, branchid)
                h5.read(z, grp + "/data")
                out.append(z)

        return out

    def fetch_functionals(self, params, branchids):
        funcs = []
        with HDF5File(self.pcomm, self.db, "r") as h5:
            # Loop over branchids
            for branchid in branchids:
                grp = self.groupname(params, branchid)
                these_values = []

                # Loop over functionals
                for func in self.functionals:
                    func_value = h5.attributes(grp)[func[1]]
                    these_values.append(func_value)
                funcs.append(these_values)
        return these_values

    def known_branches(self, params):
        # FIXME: I don't think we can do this with the limited API
        # exposed by DOLFIN

        with h5py.File(self.db, "r", driver="mpio", comm=self.mcomm) as h5:
            grp = self.groupname(params)

            try:
                all_branches = h5[grp].items()
                out = map(lambda x: int(x[0][9:]), all_branches)
            except KeyError:
                out = []
        return set(out)

    def known_parameters(self, fixed, branchid):
        # I think we don't need this in the API anymore
        raise NotImplementedError

    def max_branch(self):
        # Again, is this *really* necessary?
        raise NotImplementedError
