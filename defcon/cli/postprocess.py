from __future__ import absolute_import, print_function

import sys

from defcon import backend
from defcon.cli.common import fetch_bifurcation_problem
import six
import gc

def usage(executable):
    sys.exit("""A script that applies the postprocess routine to all solutions.

Use like

%s /path/to/my/problem.py /path/to/output
""" % (executable,))


def main(args):
    if len(args) != 3:
        usage(args[0] if len(args) > 0 else "defcon postprocess")

    probpath = args[1]
    outputdir = args[2]
    if outputdir.endswith("/"): outputdir = outputdir[:-1]

    problem = fetch_bifurcation_problem(probpath)
    if problem is None:
        usage(args[0])

    io = problem.io(outputdir)

    mesh = problem.mesh(backend.comm_world)
    Z = problem.function_space(mesh)
    functionals = problem.functionals()
    params = problem.parameters()
    consts = [x[0] for x in params]

    io.setup(params, functionals, Z)
    params = consts

    for branchid in six.moves.xrange(io.max_branch()+1):
        for values in io.known_parameters(fixed={}, branchid=branchid):
            # Read solution
            solution = io.fetch_solutions(values, [branchid])[0]

            # Assign constants
            for (const, val) in zip(consts, values):
                const.assign(val)

            problem.postprocess(solution, values, branchid, None)
            gc.collect()
