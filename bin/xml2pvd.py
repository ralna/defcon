#!/usr/bin/env python

# If the user is using the XMLIO class, this converts
# the saved solutions in the directory to PVD.
# Use like
# python xml2pvd /path/to/my/problem.py /path/to/output/directory

import sys
import os
import os.path
import imp

probpath = sys.argv[1]
outputdir = sys.argv[2]

probdir = os.path.dirname(probpath)
os.chdir(probdir)

prob = imp.load_source("prob", probpath)

globals().update(vars(prob))
# Run through each class we've imported and figure out which one inherits from BifurcationProblem.
classes = [key for key in globals().keys()]
for c in classes:
    try:
        globals()["bfprob"] = getattr(prob, c)
        assert(issubclass(bfprob, BifurcationProblem)) # check whether the class is a subclass of BifurcationProblem, which would mean it's the class we want. 
        problem = bfprob() # initialise the class.
        break
    except Exception: pass

import glob
from dolfin import *

mesh = problem.mesh(mpi_comm_world())
Z = problem.function_space(mesh)
pvd = File(outputdir + "/roots.pvd")
f = Function(Z, name="Solution")

for root in sorted(glob.glob(outputdir + "/solution-*.xml.gz")):
    f.assign(Function(Z, root))
    pvd << f

print "Wrote to " + outputdir + "/roots.pvd"
