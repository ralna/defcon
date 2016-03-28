from operatordeflation import ShiftedDeflation
from parallellayout import ranktoteamno, teamnotoranks
from parametertools import parameterstofloats, parameterstoconstants
from tasks import QuitTask, ContinuationTask, DeflationTask, Response

from mpi4py import MPI
from petsc4py import PETSc

import math
import threading
import time

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
        assert self.worldcomm.size % teamsize == 0
        self.nteams = self.worldcomm.size / self.teamsize

        # Create local communicator for the team I will join
        self.teamno = ranktoteamno(self.rank, self.teamsize)
        self.teamcomm = self.worldcomm.Split(self.teamno, key=0)
        self.teamrank = self.teamcomm.rank

        # An MPI tag to indicate response messages
        self.responsetag = 121

        # We also need to create a communicator for rank 0 to talk to each
        # team (except for team 0, which it already has, as it is a member)
        if self.rank == 0:
            self.teamcomms = [self.teamcomm]
            for teamno in range(1, self.nteams):
                teamcommpluszero = self.worldcomm.Split(teamno, key=0)
                self.teamcomms.append(teamcommpluszero)
        else:
            if self.teamno == 0:
                self.mastercomm = self.teamcomm

            for teamno in range(1, self.nteams):
                if teamno == self.teamno:
                    self.mastercomm = self.worldcomm.Split(self.teamno, key=0)
                else:
                    self.worldcomm.Split(MPI.UNDEFINED, key=0)

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

        # Fetch the initial guesses to start us off
        if self.teamno == 0:
            initialparams = parameterstofloats(self.parameters, freeindex, values[0])
            initialguesses = self.problem.guesses(self.function_space, None, None, initialparams)

        if self.rank == 0:
            # Argh. MPI is so ugly. I can't have one thread on rank 0 bcasting
            # to rank 0's team (in the master), and have another thread on rank 0 bcasting
            # to receive it (in the worker). So I have to have a completely different
            # message passing mechanism for master to rank 0, compared to everyone else.
            self.zerotask = None

            # fork the master coordinating thread
            args = (freeindex, values, initialguesses)
            thread = threading.Thread(target=self.master, args=args)
            thread.start()

            # and get to work yourself
            self.worker()
            thread.join()
        else:
            # join a worker team
            self.worker()

    def send_task(self, task, idleteam):
        # Special case for rank 0 communicating with itself
        if idleteam == 0:
            self.zerotask = task

        teamcomm = self.teamcomms[idleteam].bcast(task)

    def fetch_task(self):
        # Special case for rank 0 communicating with itself
        if self.rank == 0:
            while self.zerotask is None:
                time.sleep(0.01)
            task = self.zerotask
            self.zerotask = None
            return task

        else:
            task = self.mastercomm.bcast()
            return task

    def master(self, freeindex, values, initialguesses):
        """
        The master coordinating routine.

        *Arguments*
          freeindex (:py:class:`tuple`)
            An index into self.parameters that states which parameter is free
          values (:py:class:`list`)
            A list of continuation values for the parameter
        """

        # Initialise data structures.
        stat = MPI.Status()

        # First, set the list of idle teams to all of them.
        idleteams = range(self.nteams)

        # Task id counter
        taskid = 0

        # Next, seed the list of tasks to perform with the initial search
        newtasks = []  # tasks yet to be sent out
        waittasks = {} # tasks sent out, waiting to hear back about

        initialparams = parameterstofloats(self.parameters, freeindex, values[0])
        for guess in initialguesses:
            newtasks.append(DeflationTask(taskid=taskid, oldparams=None, branchid=taskid,
                                       newparams=initialparams, knownbranches=[]))
            taskid += 1

        # Here comes the main master loop.
        while len(newtasks) + len(waittasks) > 0:

            # If there are any tasks to send out, send them.
            while len(newtasks) > 0 and len(idleteams) > 0:
                idleteam = idleteams.pop(0)
                task     = newtasks.pop(0)
                self.send_task(task, idleteam)
                waittasks[task.taskid] = task

            # We can't send out any more tasks, either because we have no
            # tasks to send out or we have no free processors.
            # If we aren't waiting for anything to finish, we'll exit the loop
            # here. otherwise, we wait for responses and deal with consequences.
            if len(waittasks) > 0:
                response = self.worldcomm.recv(status=stat, source=MPI.ANY_SOURCE, tag=self.responsetag)

                del waittasks[response.taskid]
                idleteams.append(stat.source)

        # All continuation tasks have been finished. Tell the workers to quit.
        quit = QuitTask()
        for teamno in range(self.nteams):
            self.send_task(quit, teamno)

    def worker(self):
        """
        The main worker routine.

        Fetches its tasks from the master and executes them.
        """

        while True:
            task = self.fetch_task()

            if isinstance(task, QuitTask):
                return
            else:
                print "(%s, %s): task: %s" % (self.rank, self.teamno, task)
                time.sleep(1)
                response = Response(task.taskid, success=True)
                if self.teamrank == 0:
                    self.worldcomm.send(response, dest=0, tag=self.responsetag)

