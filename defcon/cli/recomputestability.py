from __future__ import absolute_import

import sys
import os
import glob
import gc

from defcon import backend
from defcon.cli.common import fetch_bifurcation_problem
import defcon.parametertools


def usage(executable):
    sys.exit("""A script that computes the stability of all solutions.

Use like

%s /path/to/my/problem.py /path/to/output
""" % (executable,))

def main(args):
    if len(args) != 3:
        usage(args[0] if len(args) > 0 else "defcon recompute-stability")

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
