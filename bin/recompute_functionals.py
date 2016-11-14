#!/usr/bin/env python

# A script that computes the functionals of all solutions.

# Use like

# python recompute_functionals.py /path/to/my/problem.py /path/to/output

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

io.setup(params, functionals, Z)
params = consts

for branchid in range(io.max_branch()):
    for values in io.known_parameters(fixed={}, branchid=branchid):
        # Read solution
        solution = io.fetch_solutions(values, [branchid])[0]

        # Assign constants
        for (const, val) in zip(consts, values):
            const.assign(val)

        # Compute functionals
        funcs = []
        for functional in functionals:
            func = functional[0]
            j = func(solution, consts)
            assert isinstance(j, float)
            funcs.append(j)

        # Save to disk again
        print "Saving values: %s, branchid: %d" % (values, branchid)
        io.save_solution(solution, funcs, values, branchid)