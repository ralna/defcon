from operatordeflation import ShiftedDeflation
from parallellayout import ranktoteamno, teamnotoranks
from parametertools import parameterstofloats, parameterstoconstants, nextparameters
from newton import newton
from tasks import QuitTask, ContinuationTask, DeflationTask, Response

import dolfin
from deflation import ForwardProblem
from petscsnessolver import PetscSnesSolver

from mpi4py import MPI
from petsc4py import PETSc

import math
import threading
import time
from Queue import PriorityQueue

class DeflatedContinuation(object):
    """
    This class is the main driver that implements deflated continuation.
    """

    def __init__(self, problem, io, deflation=None, teamsize=1, verbose=False):
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
          verbose (:py:class:`bool`)
            Activate verbose output.
        """
        self.problem = problem

        if deflation is None:
            deflation = ShiftedDeflation(problem, power=2, shift=1)
        self.deflation = deflation

        self.teamsize = teamsize
        self.verbose = verbose

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
        self.functionals = problem.functionals()
        self.state = dolfin.Function(self.function_space)
        self.residual = problem.residual(self.state, parameterstoconstants(self.parameters), dolfin.TestFunction(self.function_space))

        io.setup(self.parameters, self.functionals, self.function_space)
        self.io = io

        # We keep track of what solution we actually have in memory in self.state
        # for efficiency
        self.state_id = (None, None)

    def log(self, msg):
        if self.verbose:
            print "(%s, %s): %s" % (self.rank, self.teamno, msg)

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

        values = list(free[freeparam[1]])

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
            self.zerobranchid = None

            # fork the master coordinating thread
            args = (freeindex, values, initialguesses)
            thread = threading.Thread(target=self.master, args=args)
            thread.start()

            # and get to work yourself
            self.worker(freeindex, values)
            thread.join()
        else:
            # join a worker team
            self.worker(freeindex, values)

    def send_task(self, task, team):
        self.log("Sending task %s to team %s" % (task, team))

        # Special case for rank 0 communicating with itself
        if team == 0:
            self.zerotask = task

        self.teamcomms[team].bcast(task)

    def fetch_task(self):
        self.log("Fetching task")

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

    def send_branchid(self, branchid, team):
        self.log("Sending branchid %s to team %s" % (branchid, team))

        if team == 0:
            self.zerobranchid = branchid

        self.teamcomms[team].bcast(branchid)

    def fetch_branchid(self):
        self.log("Fetching branchid")

        # Special case for rank 0 communicating with itself
        if self.rank == 0:
            while self.zerobranchid is None:
                time.sleep(0.01)
            branchid = self.zerobranchid
            self.zerobranchid = None
            return branchid

        else:
            branchid = self.mastercomm.bcast()
            return branchid

    def compute_functionals(self, solution, params):
        funcs = []
        for functional in self.functionals:
            func = functional[0]
            j = func(solution, params)
            assert isinstance(j, float)
            funcs.append(j)
        return funcs

    def load_solution(self, oldparams, branchid, newparams):
        if (oldparams, branchid) == self.state_id:
            # We already have it in memory
            return

        if oldparams is None:
            # We're dealing with an initial guess
            guesses = self.problem.guesses(self.function_space, None, None, newparams)
            self.state.assign(guesses[branchid])
            self.state_id = (oldparams, branchid)
            return

        # We need to load from disk.
        fetched = self.io.fetch_solutions(oldparams, [branchid])
        self.state.assign(fetched[0])
        self.state_id = (oldparams, branchid)
        return

    def load_parameters(self, params):
        for (param, value) in zip(self.parameters, params):
            param[0].assign(value)

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

        # Branch id counter
        branchid = 0

        # Next, seed the list of tasks to perform with the initial search
        newtasks = PriorityQueue()  # tasks yet to be sent out
        waittasks = {} # tasks sent out, waiting to hear back about

        initialparams = parameterstofloats(self.parameters, freeindex, values[0])
        for guess in initialguesses:
            newtasks.put((-1, DeflationTask(taskid=taskid,
                                          oldparams=None,
                                          branchid=taskid,
                                          newparams=initialparams)))
            taskid += 1

        # Here comes the main master loop.
        while newtasks.qsize() + len(waittasks) > 0:
            # If there are any tasks to send out, send them.
            while newtasks.qsize() > 0 and len(idleteams) > 0:
                (priority, task) = newtasks.get()

                # Before we send it out, let's check we really want to.
                if isinstance(task, DeflationTask):
                    if len(self.io.known_branches(task.newparams)) >= self.problem.number_solutions(task.newparams):
                    # We've found all the branches the user's asked us for, let's roll
                        self.log("Master not dispatching %s because we have enough solutions" % task)
                        continue

                idleteam = idleteams.pop(0)
                self.send_task(task, idleteam)
                waittasks[task.taskid] = (task, idleteam)

            # We can't send out any more tasks, either because we have no
            # tasks to send out or we have no free processors.
            # If we aren't waiting for anything to finish, we'll exit the loop
            # here. otherwise, we wait for responses and deal with consequences.
            if len(waittasks) > 0:
                self.log("Cannot dispatch any tasks, waiting for response")
                response = self.worldcomm.recv(status=stat, source=MPI.ANY_SOURCE, tag=self.responsetag)

                (task, team) = waittasks[response.taskid]
                self.log("Received response %s from team %s" % (response, team))
                del waittasks[response.taskid]

                # Here comes the core logic of what happens for success or failure for the two
                # kinds of tasks.
                if isinstance(task, ContinuationTask):
                    if response.success:

                        # The worker will keep continuing, record that fact
                        newparams = nextparameters(values, freeindex, task.newparams)
                        if newparams is not None:
                            conttask = ContinuationTask(taskid=task.taskid,
                                                        oldparams=task.newparams,
                                                        branchid=task.branchid,
                                                        newparams=newparams)
                            waittasks[task.taskid] = ((conttask, team))
                        else:
                            idleteams.append(team)

                        # Either way, we want to add a deflation task to the
                        # queue. FIXME: we should really wait until *all*
                        # continuation tasks have reached this point or died,
                        # and then insert *all* deflation tasks at once. But
                        # that's a refinement.
                        newtask = DeflationTask(taskid=taskid,
                                                oldparams=task.oldparams,
                                                branchid=task.branchid,
                                                newparams=task.newparams)
                        taskid += 1
                        newtasks.put((newtask.newparams[freeindex], newtask))
                    else:
                        # We tried to continue a branch, but the continuation died. Oh well.
                        # The team is now idle.
                        idleteams.append(team)

                elif isinstance(task, DeflationTask):
                    if response.success:
                        # In this case, we want the master to
                        # 1. Allocate a new branch id for the discovered branch.
                        # FIXME: We might want to make this more sophisticated
                        # to catch duplicates --- in that event, send None. But
                        # for now we'll just accept it.
                        self.send_branchid(branchid, team)

                        # 2. If it wasn't an initial guess, insert a new
                        # deflation task, to seek again with the same settings.
                        if task.oldparams is not None:
                            newtask = DeflationTask(taskid=taskid,
                                                    oldparams=task.oldparams,
                                                    branchid=task.branchid,
                                                    newparams=task.newparams)
                            newtasks.put((newtask.newparams[freeindex], newtask))
                            taskid += 1

                        # 3. Record that the worker team is now continuing that branch,
                        # if there's continuation to be done.
                        newparams = nextparameters(values, freeindex, task.newparams)
                        if newparams is not None:
                            conttask = ContinuationTask(taskid=task.taskid,
                                                        oldparams=task.newparams,
                                                        branchid=branchid,
                                                        newparams=newparams)
                            waittasks[task.taskid] = ((conttask, team))
                        else:
                            # It's at the end of the continuation, there's no more continuation
                            # to do. Mark the team as idle.
                            idleteams.append(team)

                        branchid += 1
                    else:
                        # As expected, deflation found nothing interesting. The team is now idle.
                        idleteams.append(team)

        # All continuation tasks have been finished. Tell the workers to quit.
        quit = QuitTask()
        for teamno in range(self.nteams):
            self.send_task(quit, teamno)

    def worker(self, freeindex, values):
        """
        The main worker routine.

        Fetches its tasks from the master and executes them.
        """

        task = self.fetch_task()
        while True:
            if isinstance(task, QuitTask):
                return
            elif isinstance(task, DeflationTask):
                self.log("Executing task %s" % task)

                # Set up the problem
                self.load_solution(task.oldparams, task.branchid, task.newparams)
                self.load_parameters(task.newparams)
                other_solutions = self.io.fetch_solutions(task.newparams, self.io.known_branches(task.newparams))
                bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

                p = ForwardProblem(self.residual, self.function_space, self.state, bcs, power=2, shift=1)
                for o in other_solutions:
                    p.deflate(o)

                try:
                    solver = PetscSnesSolver()
                    solver.solve(p, self.state.vector())
                    success = True
                except:
                    import traceback; traceback.print_exc()
                    success = False

                self.state_id = (None, None) # not sure if it is a solution we care about yet

                response = Response(task.taskid, success=success)
                if self.teamrank == 0:
                    self.log("Sending response %s to master" % response)
                    self.worldcomm.send(response, dest=0, tag=self.responsetag)

                if success:
                    branchid = self.fetch_branchid()
                    if branchid is not None:
                        # We do care about this solution, so record the fact we have it in memory
                        self.state_id = (task.newparams, branchid)

                        # Save it to disk with the I/O module
                        functionals = self.compute_functionals(self.state, task.newparams)
                        self.io.save_solution(self.state, task.newparams, branchid)
                        self.io.save_functionals(functionals, task.newparams, branchid)

                        # Automatically start onto the continuation
                        newparams = nextparameters(values, freeindex, task.newparams)
                        if newparams is not None:
                            task = ContinuationTask(taskid=task.taskid,
                                                    oldparams=task.newparams,
                                                    branchid=branchid,
                                                    newparams=newparams)
                        else:
                            # Reached the end of the continuation, don't want to continue, move on
                            task = self.fetch_task()
                    else:
                        # Branch id is None, ignore the solution and move on
                        task = self.fetch_task()
                else:
                    # Deflation failed, move on
                    task = self.fetch_task()

            elif isinstance(task, ContinuationTask):
                self.log("Executing task %s" % task)

                # Set up the problem
                self.load_solution(task.oldparams, task.branchid, task.newparams)
                self.load_parameters(task.newparams)
                bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

                # Try to solve it
                p = ForwardProblem(self.residual, self.function_space, self.state, bcs, power=2, shift=1)

                try:
                    solver = PetscSnesSolver()
                    solver.solve(p, self.state.vector())
                    success = True
                except:
                    import traceback; traceback.print_exc()
                    success = False

                if success:
                    self.state_id = (task.newparams, task.branchid)

                    # Save it to disk with the I/O module
                    functionals = self.compute_functionals(self.state, task.newparams)
                    self.io.save_solution(self.state, task.newparams, branchid)
                    self.io.save_functionals(functionals, task.newparams, branchid)
                else:
                    self.state_id = (None, None)

                response = Response(task.taskid, success=success)
                if self.teamrank == 0:
                    self.log("Sending response %s to master" % response)
                    self.worldcomm.send(response, dest=0, tag=self.responsetag)

                newparams = nextparameters(values, freeindex, task.newparams)
                if success and newparams is not None:
                    task = ContinuationTask(taskid=task.taskid,
                                            oldparams=task.newparams,
                                            branchid=task.branchid,
                                            newparams=newparams)
                else:
                    task = self.fetch_task()
