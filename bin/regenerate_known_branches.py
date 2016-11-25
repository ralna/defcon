# A script that regenerates the known_parameters map in case it is deleted/unavailable.
# Use like
# python regenerate_known_branches.py /path/to/my/problem.py /path/to/output/directory

import sys
import scriptcommon
from ast import literal_eval

probpath = sys.argv[1]
outputdir = sys.argv[2]

problem = scriptcommon.fetch_bifurcation_problem(probpath)
io = problem.io(outputdir)

from defcon import *

if not isinstance(io, BranchIO):
    print "Only relevant for BranchIO."
    sys.exit(1)

import glob
import backend
import defcon.branchio
import os
import h5py
from mpi4py import MPI

mesh = problem.mesh(backend.comm_world)
Z = problem.function_space(mesh)
functionals = problem.functionals()
params = problem.parameters()
io.setup(params, functionals, Z)
pm = io.parameter_map()

h5s = glob.glob(os.path.join(outputdir, "*.h5"))

for h5 in h5s:
    basename = os.path.basename(h5)
    branchid = int(basename[9:-3])
    with h5py.File(h5, "r", driver="mpio", comm=backend.comm_world.tompi4py()) as f:
        for key in f.keys():
            params = defcon.branchio.keytoparams(key)
            old_pm = pm[params]
            pm[params] = [branchid] + old_pm

io.close_parameter_map()
