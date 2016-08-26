import sys
import os
from parametertools import parameterstostring

def makepvd(branches, problem_type, working_dir, output_dir="output", solutions_dir=None):
    # Switch to the working directory and import the problem.
    os.chdir(working_dir)
    problem_name = __import__(problem_type)
    globals().update(vars(problem_name))

    # Run through each class we've imported and figure out which one inherits from BifurcationProblem.
    classes = []
    for key in globals().keys():
        if key is not 'BifurcationProblem': classes.append(key) # remove this to make sure we don't fetch the wrong class.
    for c in classes:
        try:
            globals()["bfprob"] = getattr(problem_name, c)
            assert(issubclass(bfprob, BifurcationProblem)) # check whether the class is a subclass of BifurcationProblem, which would mean it's the class we want. 
            problem = bfprob() # initialise the class.
            break
        except Exception: pass

    # Get the mesh.
    mesh = problem.mesh(mpi_comm_world())

    # If the mesh is 1D, we don't want to use paraview. 
    if mesh.geometry().dim() < 2: plot_with_mpl = True 

    # Get the function space and set up the I/O module for fetching solutions. 
    V = problem.function_space(mesh)
    problem_parameters = problem.parameters()
    io = FileIO(output_dir)
    io.setup(problem_parameters, None, V)

    # Directory for storing solutions in.
    if solutions_dir is None: solutions_dir = output_dir + os.path.sep + "solutions" + os.path.sep

    try: os.mkdir(solutions_dir)
    except OSError: pass

    # Create a pvd file for each solution we have been asked for.
    for branchid in branches.keys():
        for param in branches[branchid]:
            # Load the solution
            y = io.fetch_solutions(param, [branchid])[0]

            # Create the file. 
            pvd_filename = solutions_dir + "params=%s@branchid=%d.pvd" % (parameterstostring(problem_parameters, param), branchid)
            pvd = File(pvd_filename)

            # Write the solution.
            pvd << y
            pvd




