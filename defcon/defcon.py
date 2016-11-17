from parametertools import Parameters
from thread import DefconThread
from worker import DefconWorker
from master import DefconMaster

from mpi4py   import MPI

import sys

class DeflatedContinuation(object):
    """
    This class is the main driver. It passes most of the work off
    to the DefconWorker and DefconMaster classes.
    """
    def __init__(self, problem, **kwargs):
        """
        Constructor.

        *Arguments*
          problem (:py:class:`defcon.BifurcationProblem`)
            A class representing the bifurcation problem to be solved.
          deflation (:py:class:`defcon.DeflationOperator`)
            A class defining a deflation operator.
          teamsize (:py:class:`int`)
            How many processors should coordinate to solve any individual PDE.
          verbose (:py:class:`bool`)
            Activate verbose output.
          debug (:py:class:`bool`)
            Activate debugging output.
          logfiles (:py:class:`bool`)
            Whether defcon should remap stdout/stderr to logfiles (useful for many processes).
          continue_backwards (+1 or -1)
            Whether defcon should also continue backwards when it finds a new branch with deflation.
          clear_output (:py:class:`bool`)
            Whether defcon should first clear any old output.
          gc_frequency (:py:class:`int`)
            How many solves should pass before we call the garbage collector.
            Set to a small value (e.g. 1) for very large problems, and a large value (e.g. 100)
            for very small problems. We try to do something sensible by default.
          comm (MPI.Comm)
            The communicator that gathers all processes involved in this computation
        """

        worldcomm = kwargs.get("comm", MPI.COMM_WORLD).Dup()
        kwargs["comm"] = worldcomm

        self.problem = problem

        # Set up I/O
        io = problem.io()
        clear_output = kwargs.get("clear_output", False)
        if worldcomm.rank == 0 and clear_output:
            io.clear()

        io.construct(worldcomm)

        if worldcomm.rank == 0:
            self.thread = DefconMaster(problem, **kwargs)
        else:
            self.thread = DefconWorker(problem, **kwargs)

    def run(self, values, freeparam=None):
        """
        The main execution routine.

        *Arguments*
          values (:py:class:`dict`)
            A dictionary mapping ASCII name of parameter to list of parameter values.
            Use a list with one element for parameters you wish to fix.
          freeparam (:py:class:`str`)
            The ASCII name of the parameter on which to start the continuation.
            Additional continuation runs can be executed by overloading the `branch_found`
            routine.
        """

        # First, check we're parallel enough.
        if self.thread.worldcomm.size < 2:
            msg = """
Defcon started with only 1 process.
At least 2 processes are required (one master, one worker).

Launch with mpiexec: mpiexec -n <number of processes> python %s
""" % sys.argv[0]
            self.thread.log(msg, warning=True)
            sys.exit(1)

        # Next, check parameters.

        problem_parameters = self.problem.parameters()
        assert len(problem_parameters) == len(values)

        if freeparam is None:
            assert len(values) == 1
            freeparam = values.keys()[0]

        # Apply list to concretely instantiate the values
        for param in values:
            values[param] = list(values[param])

        parameters = Parameters(problem_parameters, values)

        # If we only have one value for each parameter, don't bother continuing backwards
        should_continue_backwards = True
        for key in values:
            if len(values[param]) == 1:
                should_continue_backwards = False
        if not should_continue_backwards:
            self.thread.continue_backwards = False

        # Aaaand .. run.

        self.thread.run(parameters, freeparam)

    def bifurcation_diagram(self, functional, fixed={}, style="ok", **kwargs):
        if self.thread.rank != 0:
            return

        parameters = self.problem.parameters()
        functionals = self.problem.functionals()
        io = self.problem.io()
        io.setup(parameters, functionals, None)

        import matplotlib.pyplot as plt
        if "linewidth" not in kwargs: kwargs["linewidth"] = 2
        if "markersize" not in kwargs: kwargs["linewidth"] = 1

        # Find the functional index.
        funcindex = None
        for (i, functionaldata) in enumerate(functionals):
            if functionaldata[1] == functional:
                funcindex = i
                break
        assert funcindex is not None

        # And find the free variable index -- the one that doesn't show up in fixed.
        freeindices = range(len(parameters))
        for (i, param) in enumerate(parameters):
            if param[1] in fixed:
                freeindices.remove(i)
        assert len(freeindices) == 1
        freeindex = freeindices[0]

        for branchid in range(io.max_branch() + 1):
            xs = []
            ys = []
            params = io.known_parameters(fixed, branchid)
            funcs = io.fetch_functionals(params, branchid)
            for i in xrange(0, len(params)):
                param = params[i]
                func = funcs[i]
                xs.append(param[freeindex])
                ys.append(func[funcindex])
            plt.plot(xs, ys, style, **kwargs)

        plt.grid()
        plt.xlabel(parameters[freeindex][2])
        plt.ylabel(functionals[funcindex][2])

