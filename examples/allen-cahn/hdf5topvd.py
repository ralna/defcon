# This postprocessing script takes in an HDF5 file and saves the
# solutions to PVD. Example:

# python makepvd.py delta=4.00000000000000008326672684688674053177237510681e-02.hdf5
# paraview output/solutions.pvd

import sys
import os
import h5py as h5

# I want to call my file allen-cahn.py, damnit!
allencahn = __import__("allen-cahn")
globals().update(vars(allencahn))

problem = AllenCahnProblem()
mesh = problem.mesh(mpi_comm_world())
V    = problem.function_space(mesh)

dir = "output" + os.path.sep 
filename = dir + sys.argv[1] 
pvd = File(dir + "solutions.pvd")

# Find the keys
f = h5.File(filename, 'r')
solns = sorted(f.keys())
f.close()

with HDF5File(mesh.mpi_comm(), filename, 'r') as f:
    for soln in solns:
        print soln
        y = Function(V)
        f.read(y, str(soln))
        print y
        f.flush()
        pvd << y

print "Created ", (dir + "/solutions.pvd")
