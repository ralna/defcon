#!/usr/bin/env python

# If the user is using the XMLIO class, this converts
# the saved solutions in the directory to PVD.

# A script that converts saved data to PVD.
# Use like
# python makepvd.py /path/to/my/problem.py /path/to/output/directory

import sys
import scriptcommon

probpath = sys.argv[1]
outputdir = sys.argv[2]

problem = scriptcommon.fetch_bifurcation_problem(probpath)

import glob
import backend

mesh = problem.mesh(backend.comm_world)
Z = problem.function_space(mesh)
pvd = backend.File(outputdir + "/roots.pvd")
f = backend.Function(Z, name="Solution")

for root in sorted(glob.glob(outputdir + "/solution-*.h5")):
    with backend.HDF5File(backend.comm_world, root, 'r') as g:
        g.read(f, "/solution")
    problem.save_pvd(f, pvd)

print "Wrote to " + outputdir + "/roots.pvd"
