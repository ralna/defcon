#!/usr/bin/env python

# A script that computes the stability of all solutions.

# Use like

# python recompute_stability.py /path/to/my/problem.py /path/to/output

import sys
import scriptcommon

probpath = sys.argv[1]
outputdir = sys.argv[2]
if outputdir.endswith("/"): outputdir = outputdir[:-1]

problem = scriptcommon.fetch_bifurcation_problem(probpath)
io = problem.io(outputdir)

import glob
import backend
import defcon.parametertools
import defcon.newton
import gc
import os

mesh = problem.mesh(backend.comm_world)
Z = problem.function_space(mesh)
functionals = problem.functionals()
params = problem.parameters()
consts = [x[0] for x in params]

z = backend.Function(Z)

io.setup(params, functionals, Z)
params = consts

# First get the directories to loop over
dirnames = [x.replace(outputdir + "/", "") for x in glob.glob(outputdir + "/*=*")]

for dirname in dirnames:
    # Load the appropriate parameter values
    defcon.parametertools.parametersfromstring(params, dirname)
    floats = map(float, params)

    solutionid = 0
    knownroots = []

    solutions = glob.glob(outputdir + "/" + dirname + "/*h5")
    for solution in solutions:
        gc.collect()

        # Compute branchid
        filename = os.path.basename(solution)
        filename = filename.replace("solution-", "")
        filename = filename.replace(".h5", "")
        branchid = int(filename)

        # Load from disk
        with backend.HDF5File(backend.comm_world, solution, "r") as h5:
            h5.read(z, "/solution")

        # Compute stability
        d = problem.compute_stability(consts, branchid, z)

        # Save stability
        io.save_stability(d["stable"], d.get("eigenvalues", []), d.get("eigenfunctions", []), floats, branchid)

        print "Solution %s: stability: %s" % (solution, d["stable"])
