# -*- coding: utf-8 -*-

from operatordeflation import ShiftedDeflation
from parallellayout import ranktoteamno, teamnotoranks
from parametertools import parameterstofloats, parameterstoconstants, nextparameters, prevparameters
from newton import newton
from tasks import QuitTask, ContinuationTask, DeflationTask, Response
from journal import Journal, FileJournal

import backend

from mpi4py import MPI
from petsc4py import PETSc
from numpy import isinf

import math
import time
import sys
import signal

try:
    import ipdb as pdb
except ImportError:
    import pdb

from heapq import heappush, heappop

class DeflatedContinuation(object):
    """
    This class is the main driver that implements deflated continuation.
    """

    def __init__(self, problem, deflation=None, teamsize=1, verbose=False, logfiles=False, strict=False):
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
          logfiles (:py:class:`bool`)
            Whether defcon should remap stdout/stderr to logfiles (useful for many processes).
          strict (:py:class:`bool`)
            Whether defcon should only run deflation processes when there's no change of finding 
            the same solution as a continuation task. 
        """
        self.problem = problem

        self.strict = strict

        self.teamsize = teamsize
        self.verbose = verbose

        # Create a unique context, so as not to confuse my messages with other
        # libraries
        self.worldcomm = MPI.COMM_WORLD.Dup()
        self.rank = self.worldcomm.rank

        # Assert even divisibility of team sizes
        assert (self.worldcomm.size-1) % teamsize == 0
        self.nteams = (self.worldcomm.size-1) / self.teamsize

        # Create local communicator for the team I will join
        self.teamno = ranktoteamno(self.rank, self.teamsize)
        self.teamcomm = self.worldcomm.Split(self.teamno, key=0)
        self.teamrank = self.teamcomm.rank

        # An MPI tag to indicate response messages
        self.responsetag = 121

        # We also need to create a communicator for rank 0 to talk to each
        # team (except for team 0, which it already has, as it is a member)
        if self.rank == 0:
            self.mastercomm = self.teamcomm
            self.teamcomms = []
            for teamno in range(0, self.nteams):

                teamcommpluszero = self.worldcomm.Split(teamno, key=0)
                self.teamcomms.append(teamcommpluszero)
        else:
            if self.teamno == 0:
                self.mastercomm = self.teamcomm

            for teamno in range(0, self.nteams):
                if teamno == self.teamno:
                    self.mastercomm = self.worldcomm.Split(self.teamno, key=0)
                else:
                    self.worldcomm.Split(MPI.UNDEFINED, key=0)

        # Take some data from the problem
        self.mesh = problem.mesh(PETSc.Comm(self.teamcomm))
        self.function_space = problem.function_space(self.mesh)
        self.parameters = problem.parameters()
        self.functionals = problem.functionals()
        self.state = backend.Function(self.function_space)
        self.residual = problem.residual(self.state, parameterstoconstants(self.parameters), backend.TestFunction(self.function_space))
        self.trivial_solutions = None # computed by the worker on initialisation later

        io = self.problem.io()
        io.setup(self.parameters, self.functionals, self.function_space)
        self.io = io

        # We keep track of what solution we actually have in memory in self.state
        # for efficiency
        self.state_id = (None, None)

        # If instructed, create logfiles for each team
        if logfiles:
            if self.teamrank == 0:
                sys.stdout = open("defcon.log.%d" % self.teamno, "w")
                sys.stderr = open("defcon.err.%d" % self.teamno, "w")
            else:
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")

        if deflation is None:
            deflation = ShiftedDeflation(problem, power=2, shift=1)
        params = [x[0] for x in self.parameters]
        deflation.set_parameters(params)
        self.deflation = deflation

    def log(self, msg, master=False, warning=False):
        if not self.verbose: return
        if self.teamrank != 0: return

        if master:
            if not warning:
                fmt = BLUE = "\033[1;37;34m%s\033[0m"
            else:
                fmt = RED = "\033[1;37;31m%s\033[0m"
        else:
            fmt = GREEN = "\033[1;37;32m%s\033[0m"

        if master:
            header = "MASTER:   "
        else:
            header = "TEAM %3d: " % self.teamno

        timestamp = "[%s] " % time.strftime("%H:%M:%S")

        print fmt % (timestamp + header + msg)
        sys.stdout.flush()

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
        self.fixed = fixed

        # Fix the fixed parameters and identify the free parameter.

        freeparam = None
        freeindex = None
        for (index, param) in enumerate(self.parameters):
            if param[1] in fixed:
                param[0].assign(fixed[param[1]])

            if param[1] in free:
                freeparam = param
                freeindex = index

        if freeparam is None:
            backend.info_red("Cannot find %s in parameters %s." % (free.keys()[0], [param[1] for param in self.parameters]))
            assert freeparam is not None

        values = list(free[freeparam[1]])

        if self.rank == 0:
            self.zerotask = []
            self.zerobranchid = []
            self.master(freeindex, values)
        else:
            # join a worker team
            self.worker(freeindex, values)

    def send_task(self, task, team):
        self.log("Sending task %s to team %s" % (task, team), master=True)

        # Special case for rank 0 communicating with itself
        if team == 0:
            self.zerotask.append(task)

        self.teamcomms[team].bcast(task)

    def fetch_task(self):
        self.log("Fetching task")

        # Special case for rank 0 communicating with itself
        if self.rank == 0:
            while len(self.zerotask) == 0:
                time.sleep(0.01)
            task = self.zerotask.pop(0)
            return task

        else:
            task = self.mastercomm.bcast(None)
            return task

    def send_branchid(self, branchid, team):
        self.log("Sending branchid %s to team %s" % (branchid, team), master=True)

        if team == 0:
            self.zerobranchid.append(branchid)

        self.teamcomms[team].bcast(branchid)

    def fetch_branchid(self):
        self.log("Fetching branchid")

        # Special case for rank 0 communicating with itself
        if self.rank == 0:
            while len(self.zerobranchid) == 0:
                time.sleep(0.01)
            branchid = self.zerobranchid.pop(0)
            self.log("Got branchid %d" % branchid)
            return branchid

        else:
            branchid = self.mastercomm.bcast(None)
            self.log("Got branchid %d" % branchid)
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
            guess = self.problem.initial_guess(self.function_space, newparams, branchid)
            self.state.assign(guess)
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

    def master(self, freeindex, values):
        """
        The master coordinating routine.

        *Arguments*
          freeindex (:py:class:`tuple`)
            An index into self.parameters that states which parameter is free
          values (:py:class:`list`)
            A list of continuation values for the parameter
        """

        # Escape hatches for debugging intricate deadlocks
        signal.signal(signal.SIGUSR1, lambda sig, frame: pdb.set_trace())

        # Initialise data structures.
        stat = MPI.Status()

        # First, set the list of idle teams to all of them.
        idleteams = range(self.nteams)

        # Task id counter
        taskid_counter = 0

        # Branch id counter
        branchid_counter = 0

        # Next, seed the list of tasks to perform with the initial search
        newtasks = []  # tasks yet to be sent out
        deferredtasks = [] # tasks that we've been forced to defer as we don't have enough information to ensure they're necessary; only used in strict mode
        waittasks = {} # tasks sent out, waiting to hear back about

        # A dictionary of parameters -> branches to ensure they exist,
        # to avoid race conditions
        ensure_branches = dict()

        # If we're going downwards in continuation parameter, we need to change
        # signs in a few places
        if values[0] < values[-1]:
            sign = +1
            minvals = min
        else:
            sign = -1
            minvals = max

        # If there's only one process, show a warning. FIXME: do something more advanced so we can run anyway. 
        if self.worldcomm.size < 2:
            self.log("Defcon started with only 1 process. At least 2 processes are required (one master, one worker).\n\nLaunch with mpiexec: mpiexec -n <number of processes> python <path to file>", master=True, warning=True)
            import sys; sys.exit(1)

        # Create a journal object.
        journal = FileJournal(self.io.directory, self.parameters, self.functionals, freeindex, sign)
        try:
            # First check to see if the journal exists.
            assert(journal.exists())

            # The journal file already exists. Let's find out what we already know so we can resume our computation where we left off.
            previous_sweep, branches, oldfreeindex = journal.resume()

            # Check that we are continuing from the same free parameter. If not, we want to start again.
            assert(oldfreeindex==freeindex)

            # Everything checks out, so lets schedule the appropriate tasks. 
            branchid_counter = len(branches)

            # Set all teams to idle.
            for teamno in range(self.nteams):
                journal.team_job(teamno, "i")

            # Schedule continuation tasks for any branches that aren't done yet.
            for branchid in branches.keys():
                oldparams = branches[branchid]
                newparams = nextparameters(values, freeindex, oldparams)
                if newparams is not None:
                    task = ContinuationTask(taskid=taskid_counter,
                                            oldparams=oldparams,
                                            branchid=int(branchid),
                                            newparams=newparams)
                    self.log("Scheduling task: %s" % task, master=True)
                    heappush(newtasks, (-1, task))
                    taskid_counter += 1


            # We need to schedule deflation tasks at every point from where we'd completed our sweep up to previously 
            # to the furthest we've got in continuation, on every branch.
            for branchid in branches.keys():
                # Get the fixed parameters
                knownparams = [x[freeindex] for x in self.io.known_parameters(self.fixed, branchid)]
                oldparams = list(parameterstofloats(self.parameters, freeindex, values[0]))
                oldparams[freeindex] = previous_sweep
                newparams = nextparameters(values, freeindex, tuple(oldparams))
                while newparams is not None and sign*newparams[freeindex] <= sign*branches[branchid][freeindex]: 
                    # As long as we're not at the end of the parameter range and we haven't exceeded the extent
                    # of this branch, schedule a deflation. 

                    if oldparams[freeindex] in knownparams:
                        task = DeflationTask(taskid=taskid_counter,
                                             oldparams=oldparams,
                                             branchid=int(branchid),
                                             newparams=newparams)
                        self.log("Scheduling task: %s" % task, master=True)
                        taskid_counter += 1
                        heappush(newtasks, (sign*task.newparams[freeindex], task))

                    oldparams = newparams
                    newparams = nextparameters(values, freeindex, newparams)

        except Exception:
            # Either the journal file does not exist, or something else bad happened. 
            # Oh well, start from scratch.
            journal.setup(self.nteams, min(values), max(values))
            initialparams = parameterstofloats(self.parameters, freeindex, values[0])
            previous_sweep = initialparams[freeindex]

            # Send off initial tasks
            knownbranches = self.io.known_branches(initialparams)
            branchid_counter = len(knownbranches)
            if len(knownbranches) > 0:
                nguesses = len(knownbranches)
                self.log("Using %d known solutions at %s" % (nguesses, initialparams,), master=True)
                oldparams = initialparams
                initialparams = nextparameters(values, freeindex, initialparams)

                for guess in range(nguesses):
                    task = ContinuationTask(taskid=taskid_counter,
                                            oldparams=oldparams,
                                            branchid=taskid_counter,
                                            newparams=initialparams)
                    heappush(newtasks, (-1, task))
                    taskid_counter += 1
            else:
                self.log("Using user-supplied initial guesses at %s" % (initialparams,), master=True)
                oldparams = None
                nguesses = self.problem.number_initial_guesses(initialparams)
                for guess in range(nguesses):
                    task = DeflationTask(taskid=taskid_counter,
                                         oldparams=oldparams,
                                         branchid=taskid_counter,
                                         newparams=initialparams)
                    heappush(newtasks, (-1, task))
                    taskid_counter += 1

        # Here comes the main master loop.
        while len(newtasks) + len(waittasks) + len(deferredtasks) > 0:
            # If there are any tasks to send out, send them.
            while len(newtasks) > 0 and len(idleteams) > 0:
                (priority, task) = heappop(newtasks)

                # Let's check if we have found enough solutions already.
                send = True
                if isinstance(task, DeflationTask):
                    knownbranches = self.io.known_branches(task.newparams)
                    if task.newparams in ensure_branches:
                        knownbranches = knownbranches.union(ensure_branches[task.newparams])
                    if len(knownbranches) >= self.problem.number_solutions(task.newparams):
                    # We've found all the branches the user's asked us for, let's roll.
                        self.log("Master not dispatching %s because we have enough solutions" % task, master=True)
                        continue

                    # Strict mode: If either there's still a task looking for solutions on earlier
                    # parameters, we want to not send this task out now and look at it again later.
                    # This is because the currently running task might find a branch that we will
                    # to deflate here.
                    if self.strict:
                        for (t, r) in waittasks.values():
                            if (sign*t.newparams[freeindex]<=sign*task.newparams[freeindex]):
                                send = False
                                break

                if send:
                    # OK, we're happy to send it out. Let's tell it any new information
                    # we've found out since we scheduled it.
                    if task.newparams in ensure_branches:
                        task.ensure(ensure_branches[task.newparams])
                    idleteam = idleteams.pop(0)
                    self.send_task(task, idleteam)
                    waittasks[task.taskid] = (task, idleteam)

                    # Write to the journal, saying that this team is now performing deflation. 
                    journal.team_job(idleteam, "d", task.newparams, task.branchid)
                else: 
                    # Best reschedule for later, as there is still pertinent information yet to come in. 
                    self.log("Deferring task %s." % task, master=True)
                    heappush(deferredtasks, (priority, task))


            # We can't send out any more tasks, either because we have no
            # tasks to send out or we have no free processors.
            # If we aren't waiting for anything to finish, we'll exit the loop
            # here. otherwise, we wait for responses and deal with consequences.
            if len(waittasks) > 0:
                self.log("Cannot dispatch any tasks, waiting for response.", master=True)

                waiting_values = [wtask[0].oldparams for wtask in waittasks.values() if wtask[0].oldparams is not None]
                newtask_values = [ntask[1].oldparams for ntask in newtasks if ntask[1].oldparams is not None]
                if len(waiting_values + newtask_values) > 0:
                    minparams = minvals(waiting_values + newtask_values, key = lambda x: x[freeindex])
                    prevparams = prevparameters(values, freeindex, minparams)
                    if prevparams is not None:
                        minwait = prevparams[freeindex]

                        tot_solutions = self.problem.number_solutions(minparams)
                        if isinf(tot_solutions): tot_solutions = '?'
                        num_solutions = len(self.io.known_branches(minparams))
                        self.log("Sweep completed <= %14.12e (%s/%s solutions)." % (minwait, num_solutions, tot_solutions), master=True)

                        # Write to the journal saying where we've completed our sweep up to.
                        journal.sweep(minwait)

                response = self.worldcomm.recv(status=stat, source=MPI.ANY_SOURCE, tag=self.responsetag)

                (task, team) = waittasks[response.taskid]
                self.log("Received response %s about task %s from team %s" % (response, task, team), master=True)
                del waittasks[response.taskid]

                # Here comes the core logic of what happens for success or failure for the two
                # kinds of tasks.
                if isinstance(task, ContinuationTask):
                    if response.success:

                        # Record this entry in the journal. 
                        journal.entry(team, task.oldparams, task.branchid, task.newparams, response.functionals, True)

                        # The worker will keep continuing, record that fact
                        newparams = nextparameters(values, freeindex, task.newparams)
                        if newparams is not None:
                            conttask = ContinuationTask(taskid=task.taskid,
                                                        oldparams=task.newparams,
                                                        branchid=task.branchid,
                                                        newparams=newparams)
                            waittasks[task.taskid] = ((conttask, team))
                            self.log("Waiting on response for %s" % conttask, master=True)
                            journal.team_job(team, "c", task.newparams, task.branchid)
                        else:
                            idleteams.append(team)
                            journal.team_job(team, "i")

                        newtask = DeflationTask(taskid=taskid_counter,
                                                oldparams=task.oldparams,
                                                branchid=task.branchid,
                                                newparams=task.newparams)
                        taskid_counter += 1
                        heappush(newtasks, (sign*newtask.newparams[freeindex], newtask))

                    else:
                        # We tried to continue a branch, but the continuation died. Oh well.
                        # The team is now idle.
                        self.log("Continuation task of team %d on branch %d failed at parameters %s." % (team, task.branchid, task.newparams), master=True, warning=True)
                        idleteams.append(team)
                        journal.team_job(team, "i")

                elif isinstance(task, DeflationTask):
                    if response.success:
                        # In this case, we want the master to
                        # 1. Allocate a new branch id for the discovered branch.
                        # FIXME: We might want to make this more sophisticated
                        # to catch duplicates --- in that event, send None. But
                        # for now we'll just accept it.
                        self.send_branchid(branchid_counter, team)

                        # Record this new solution in the journal
                        journal.entry(team, task.oldparams, branchid_counter, task.newparams, response.functionals, False)

                        # 2. Insert a new deflation task, to seek again with the same settings.
                        newtask = DeflationTask(taskid=taskid_counter,
                                                oldparams=task.oldparams,
                                                branchid=task.branchid,
                                                newparams=task.newparams)
                        if task.oldparams is not None:
                            newpriority = sign*newtask.newparams[freeindex]
                        else:
                            newpriority = -1

                        heappush(newtasks, (newpriority, newtask))
                        taskid_counter += 1

                        # 3. Record that the worker team is now continuing that branch,
                        # if there's continuation to be done.
                        newparams = nextparameters(values, freeindex, task.newparams)
                        if newparams is not None:
                            conttask = ContinuationTask(taskid=task.taskid,
                                                        oldparams=task.newparams,
                                                        branchid=branchid_counter,
                                                        newparams=newparams)
                            waittasks[task.taskid] = ((conttask, team))
                            self.log("Waiting on response for %s" % conttask, master=True)

                            # Write to the journal, saying that this team is now doing continuation.
                            journal.team_job(team, "c", task.newparams, task.branchid)
                        else:
                            # It's at the end of the continuation, there's no more continuation
                            # to do. Mark the team as idle.
                            idleteams.append(team)
                            journal.team_job(team, "i")

                        # We'll also make sure that any other DeflationTasks in the queue
                        # that have these parameters know about the existence of this branch.
                        if task.newparams not in ensure_branches:
                            ensure_branches[task.newparams] = set()
                        ensure_branches[task.newparams].add(branchid_counter)

                        branchid_counter += 1

                    else:
                        # As expected, deflation found nothing interesting. The team is now idle.
                        idleteams.append(team)
                        journal.team_job(team, "i")

            # Maybe we deferred some deflation tasks because we didn't have enough information to judge if they were worthwhile. Now we must reschedule.
            if len(deferredtasks) > 0:
                # Take as many as there are idle teams. This makes things run much smoother than taking them all. 
                for i in range(len(idleteams)):
                    try:
                        (priority, task) = heappop(deferredtasks)
                        heappush(newtasks, (priority, task))
                        self.log("Rescheduling the previously deferred task %s" % task, master=True)
                    except IndexError: break

        # All continuation tasks have been finished. Move sweepline to the end and tell the workers to quit.
        journal.sweep(values[-1])
        quit = QuitTask()
        for teamno in range(self.nteams):
            self.send_task(quit, teamno)
            journal.team_job(teamno, "q")


    def worker(self, freeindex, values):
        """
        The main worker routine.

        Fetches its tasks from the master and executes them.
        """

        # Escape hatch for debugging
        if self.rank != 0:
            signal.signal(signal.SIGUSR1, lambda sig, frame: pdb.set_trace())

        task = self.fetch_task()
        while True:
            if isinstance(task, QuitTask):
                return
            elif isinstance(task, DeflationTask):
                self.log("Executing task %s" % task)

                # Check for trivial solutions
                if self.trivial_solutions is None:
                    self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, freeindex)

                # Set up the problem
                self.load_solution(task.oldparams, task.branchid, task.newparams)
                self.load_parameters(task.newparams)
                knownbranches = self.io.known_branches(task.newparams)

                # If there are branches that must be there, spin until they are there
                if len(task.ensure_branches) > 0:
                    while True:
                        if task.ensure_branches.issubset(knownbranches):
                            break
                        self.log("Waiting until branches %s are available for %s. Known branches: %s" % (task.ensure_branches, task.newparams, knownbranches))
                        time.sleep(1)
                        knownbranches = self.io.known_branches(task.newparams)

                other_solutions = self.io.fetch_solutions(task.newparams, knownbranches)
                self.log("Deflating other branches %s" % knownbranches)
                bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

                self.deflation.deflate(other_solutions + self.trivial_solutions)
                success = newton(self.residual, self.state, bcs, self.teamno, self.deflation, snes_setup=self.problem.configure_snes)

                self.state_id = (None, None) # not sure if it is a solution we care about yet

                # Get the functionals now, so we can send them to the master.
                if success: functionals = self.compute_functionals(self.state, task.newparams)
                else: functionals = None

                response = Response(task.taskid, success=success, functionals=functionals)
                if self.teamrank == 0:
                    self.log("Sending response %s to master" % response)
                    self.worldcomm.send(response, dest=0, tag=self.responsetag)

                if success:
                    branchid = self.fetch_branchid()
                    if branchid is not None:
                        # We do care about this solution, so record the fact we have it in memory
                        self.state_id = (task.newparams, branchid)
                        # Save it to disk with the I/O module
                        self.log("Found new solution at parameters %s (branchid=%s) with functionals %s" % (task.newparams, branchid, functionals))
                        self.problem.monitor(task.newparams, branchid, self.state, functionals)
                        self.io.save_solution(self.state, functionals, task.newparams, branchid)
                        self.log("Saved solution to %s to disk" % task)

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

                # Check for trivial solutions
                if self.trivial_solutions is None:
                    self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, freeindex)

                # Set up the problem
                self.load_solution(task.oldparams, task.branchid, task.newparams)
                self.load_parameters(task.newparams)
                knownbranches = self.io.known_branches(task.newparams)
                other_solutions = self.io.fetch_solutions(task.newparams, knownbranches)
                bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

                # Try to solve it
                self.deflation.deflate(other_solutions + self.trivial_solutions)
                success = newton(self.residual, self.state, bcs, self.teamno, self.deflation, snes_setup=self.problem.configure_snes)

                if success:
                    self.state_id = (task.newparams, task.branchid)

                    # Save it to disk with the I/O module
                    functionals = self.compute_functionals(self.state, task.newparams)
                    self.problem.monitor(task.newparams, task.branchid, self.state, functionals)
                    self.io.save_solution(self.state, functionals, task.newparams, task.branchid)

                else:
                    functionals = None
                    self.state_id = (None, None)

                response = Response(task.taskid, success=success, functionals=functionals)
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

    def bifurcation_diagram(self, functional, fixed={}):
        if self.rank != 0:
            return

        import matplotlib.pyplot as plt


        # Find the functional index.
        funcindex = None
        for (i, functionaldata) in enumerate(self.functionals):
            if functionaldata[1] == functional:
                funcindex = i
                break
        assert funcindex is not None

        # And find the free variable index -- the one that doesn't show up in fixed.
        freeindices = range(len(self.parameters))
        for (i, param) in enumerate(self.parameters):
            if param[1] in fixed:
                freeindices.remove(i)
        assert len(freeindices) == 1
        freeindex = freeindices[0]

        for branchid in range(self.io.max_branch() + 1):
            xs = []
            ys = []
            params = self.io.known_parameters(fixed, branchid)
            funcs = self.io.fetch_functionals(params, branchid)
            for i in xrange(0, len(params)):
                param = params[i]
                func = funcs[i]
                xs.append(param[freeindex])
                ys.append(func[funcindex])
            plt.plot(xs, ys, 'ok', label="Branch %d" % branchid, linewidth=2, linestyle='-', markersize=1)

        plt.grid()
        plt.xlabel(self.parameters[freeindex][2])
        plt.ylabel(self.functionals[funcindex][2])
