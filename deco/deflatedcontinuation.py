from operatordeflation import ShiftedDeflation
from parallellayout import ranktoteamno, teamnotoranks
from parametertools import parameterstofloats, parameterstoconstants
from tasks import QuitTask, ContinuationTask, DeflationTask

from mpi4py import MPI
from petsc4py import PETSc

import math
import threading


class DeflatedContinuation(object):
    """
    This class is the main driver that implements deflated continuation.
    """

    def __init__(self, problem, io, deflation=None, teamsize=1):
        """
        Constructor.

        *Arguments*
          problem (:py:class:`deco.BifurcationProblem`)
            A class representing the bifurcation problem to be solved.
          io (:py:class:`deco.IO`)
            A class describing how to store the solutions on disk.
          deflation (:py:class:`deco.DeflationOperator`)
            A class defining a deflation operator.
          teamsize (:py:class:`int`)
            How many processors should coordinate to solve any individual PDE.
        """
        self.problem = problem
        self.io = io

        if deflation is None:
            deflation = ShiftedDeflation(problem, power=2, shift=1)
        self.deflation = deflation

        self.teamsize = teamsize

        # Create a unique context, so as not to confuse my messages with other
        # libraries
        self.worldcomm = MPI.COMM_WORLD.Dup()
        self.rank = self.worldcomm.rank

        # Assert even divisibility of team sizes
        assert (self.worldcomm.size - 1) % teamsize == 0
        self.nteams = (self.worldcomm.size - 1) / self.teamsize

        # Create local communicator for the team I will join
        self.teamno = ranktoteamno(self.rank, self.teamsize)
        self.teamcomm = self.worldcomm.Split(self.teamno, key=0)

        # We also need to create a communicator for rank 0 to talk to each
        # team (except for team 0, which it already has, as it is a member)
        if self.rank == 0:
            self.teamcomms = []
            for teamno in range(0, self.nteams):
                teamcommpluszero = self.worldcomm.Split(teamno, key=0)
                self.teamcomms.append(teamcommpluszero)
        else:
            for teamno in range(0, self.nteams):
                if teamno == self.teamno:
                    self.mastercomm = self.worldcomm.Split(self.teamno, key=0)
                else:
                    self.worldcomm.Split(MPI.UNDEFINED, key=0)

        # Some MPI tags
        self.mastertoworker = 121
        self.workertomaster = 144

        # Take some data from the problem
        self.mesh = problem.mesh(PETSc.Comm(self.teamcomm))
        self.function_space = problem.function_space(self.mesh)
        self.parameters = problem.parameters()

    def run(self, free, fixed={}):
        """
        The main execution routine.

        *Arguments*
          free (:py:class:`dict`)
            A dictionary mapping ASCII name of parameter to list of parameter values.
          fixed (:py:class:`dict`)
            A dictionary mapping ASCII name of parameter to fixed value.
        """
        # First, check parameters.

        assert len(self.parameters) == len(fixed) + len(free)
        assert len(free) == 1

        # Fix the fixed parameters and identify the free parameter.

        freeparam = None
        freeindex = None
        for (index, param) in enumerate(self.parameters):
            if param[1] in fixed:
                param[0].assign(fixed[param[1]])

            if param[1] in free:
                freeparam = param
                freeindex = index
        assert freeparam is not None

        values = free[freeparam[1]]
        args = (freeindex, values)

        if self.rank == 0:
            # fork the master coordinating thread
            self.master(*args)
        else:
            # join a worker team
            self.worker()

    def master(self, freeindex, values):
        """
        The master coordinating routine.

        *Arguments*
          freeindex (:py:class:`tuple`)
            An index into self.parameters that states which parameter is free
          values (:py:class:`list`)
            A list of continuation values for the parameter
        """

        # Initialise data structures.

        # First, set the list of idle teams to all of them.
        idleteams = range(self.nteams)
        stat = MPI.Status()

        # Task id counter
        taskid = 0

        # Next, seed the list of tasks to perform with the initial search
        newtasks = []  # tasks yet to be sent out
        waittasks = [] # tasks sent out, waiting to hear back about

        initialparams = parameterstofloats(self.parameters, freeindex, values[0])
        guesses = self.problem.guesses(self.function_space, None, None, initialparams)
        for guess in guesses:
            newtasks.append(DeflationTask(taskid=taskid, oldparams=None, branchid=taskid,
                                       newparams=initialparams, knownbranches=[]))
            taskid += 1

        # Here comes the main master loop.
        while len(newtasks) + len(waittasks) > 0:

            # If there are any tasks to send out, send them.
            while len(newtasks) > 0:
                idleteam = idleteams.pop(0)
                task     = newtasks.pop(0)
                self.teamcomms[idleteam].bcast(task)
                # append to waittasks

            # ... wait for responses and deal with consequences

        # All continuation tasks have been finished. Tell the workers to quit.
        quit = QuitTask()
        for teamno in range(self.nteams):
            self.teamcomms[teamno].bcast(quit)

    def worker(self):
        """
        The main worker routine.

        Fetches its tasks from the master and executes them.
        """

        while True:
            task = self.mastercomm.bcast()

            if isinstance(task, QuitTask):
                return
            else:
                print "(%s, %s): task: %s" % (self.rank, self.teamno, task)
                import time; time.sleep(1)

