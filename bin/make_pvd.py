#!/usr/bin/env python

# A script that converts saved data to PVD.
# Use like
# python make_pvd.py /path/to/my/problem.py /path/to/output/directory values
# e.g.
# python make_pvd.py /path/to/my/problem.py /path/to/output/directory "(0, 0)"

import sys
import scriptcommon
from ast import literal_eval

probpath = sys.argv[1]
outputdir = sys.argv[2]
if outputdir.endswith("/"): outputdir = outputdir[:-1]
values = literal_eval(sys.argv[3])
if isinstance(values, float): values = (values,)

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

io.setup(params, functionals, Z)
params = consts

filename = os.path.join(outputdir, "viz", "values-%s.pvd" % (sys.argv[3],))
pvd = backend.File(filename)
branches = io.known_branches(values)
solutions = io.fetch_solutions(values, branches)
for solution in solutions:
    solution.rename("Solution", "Solution")
    problem.save_pvd(solution, pvd)

print "Wrote to " + filename
