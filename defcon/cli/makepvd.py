from __future__ import absolute_import

import sys
import os
from ast import literal_eval

from defcon import backend
from defcon.cli.common import fetch_bifurcation_problem


def usage(executable):
    sys.exit("""A script that converts saved data to PVD.
Use like
%s /path/to/my/problem.py /path/to/output/directory values
e.g.
%s /path/to/my/problem.py /path/to/output/directory "(0, 0)"
""" % (executable, executable))


def main(args):
    if len(args) != 4:
        usage(args[0] if len(args) > 0 else "defcon make-pvd")

    probpath = args[1]
    outputdir = args[2]
    if outputdir.endswith("/"): outputdir = outputdir[:-1]
    values = literal_eval(args[3])
    if isinstance(values, float): values = (values,)

    problem = fetch_bifurcation_problem(probpath)

    io = problem.io(outputdir)

    mesh = problem.mesh(backend.comm_world)
    Z = problem.function_space(mesh)
    functionals = problem.functionals()
    params = problem.parameters()
    consts = [x[0] for x in params]

    io.setup(params, functionals, Z)
    params = consts

    filename = os.path.join(outputdir, "viz", "values-%s.pvd" % (args[3],))
    pvd = backend.File(filename)
    branches = io.known_branches(values)
    print "Known branches at %s: %s" % (values, branches)
    solutions = io.fetch_solutions(values, branches)
    for solution in solutions:
        solution.rename("Solution", "Solution")
        problem.save_pvd(solution, pvd)

    print "Wrote to " + filename
