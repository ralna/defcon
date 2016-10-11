# -*- coding: utf-8 -*-

from operatordeflation import ShiftedDeflation
from parallellayout import ranktoteamno, teamnotoranks
from parametertools import parameterstofloats, parameterstoconstants, nextparameters, prevparameters
from newton import newton
from tasks import QuitTask, ContinuationTask, DeflationTask, StabilityTask, Response
from journal import Journal, FileJournal
from iomodule import remap_c_streams

import backend

from mpi4py import MPI
from petsc4py import PETSc
from numpy import isinf

import math
import time
import sys
import signal
import gc
import os

try:
    import ipdb as pdb
except ImportError:
    import pdb

from heapq import heappush, heappop

class DeflatedContinuation(object):
    """
    This class is the main driver that implements deflated continuation.
    """

    def __init__(self, problem, deflation=None, teamsize=1, **kwargs):
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
          comm (MPI.Comm)
            The communicator that gathers all processes involved in this computation
        """

        self.problem = problem
        self.deflation = deflation
        self.teamsize = teamsize

        self.verbose  = kwargs.get("verbose", True)
        self.debug    = kwargs.get("debug", False)
        self.logfiles = kwargs.get("logfiles", False)
        self.continue_backwards = kwargs.get("continue_backwards", True)
        self.worldcomm = kwargs.get("comm", MPI.COMM_WORLD).Dup()

        self.configure_comms()
        self.fetch_data()
        self.configure_logs()
        self.construct_deflation()
        self.configure_io()

        # We keep track of what solution we actually have in memory in self.state
        # for efficiency
        self.state_id = (None, None)

    def configure_comms(self):
        # Create a unique context, so as not to confuse my messages with other
        # libraries
        self.rank = self.worldcomm.rank

        # Assert even divisibility of team sizes
        assert (self.worldcomm.size-1) % self.teamsize == 0
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

    def fetch_data(self):
        # Take some data from the problem
        problem = self.problem
        self.mesh = problem.mesh(PETSc.Comm(self.teamcomm))
        self.function_space = problem.function_space(self.mesh)
        self.parameters = problem.parameters()
        self.functionals = problem.functionals()
        self.state = backend.Function(self.function_space)
        self.residual = problem.residual(self.state, parameterstoconstants(self.parameters), backend.TestFunction(self.function_space))
        self.trivial_solutions = None # computed by the worker on initialisation later

    def configure_io(self):
        io = self.problem.io()
        io.setup(self.parameters, self.functionals, self.function_space)
        self.io = io

    def configure_logs(self):
        # If instructed, create logfiles for each team
        if self.logfiles:
            if self.rank == 0:
                stdout_filename = "defcon.log.master"
                stderr_filename = "defcon.err.master"
            else:
                if self.teamrank == 0:
                    stdout_filename = "defcon.log.%d" % self.teamno
                    stderr_filename = "defcon.err.%d" % self.teamno
                else:
                    stdout_filename = os.devnull
                    stderr_filename = os.devnull

            remap_c_streams(stdout_filename, stderr_filename)

    def construct_deflation(self):
        if self.deflation is None:
            self.deflation = ShiftedDeflation(self.problem, power=2, shift=1)
        params = [x[0] for x in self.parameters]
        self.deflation.set_parameters(params)

    def log(self, msg, master=False, warning=False):
        if not self.verbose: return
        if self.teamrank != 0: return

        if warning:
            fmt = RED = "\033[1;37;31m%s\033[0m"
        else:
            if master:
                fmt = BLUE = "\033[1;37;34m%s\033[0m"
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
        self.freeindex = None
        for (index, param) in enumerate(self.parameters):
            if param[1] in fixed:
                param[0].assign(fixed[param[1]])

            if param[1] in free:
                freeparam = param
                self.freeindex = index

        if freeparam is None:
            backend.info_red("Cannot find %s in parameters %s." % (free.keys()[0], [param[1] for param in self.parameters]))
            assert freeparam is not None

        values = list(free[freeparam[1]])
        self.freeindex = self.freeindex

        if self.rank == 0:
            self.master(values)
        else:
            # join a worker team
            self.worker(values)

    def send_task(self, task, team):
        self.log("Sending task %s to team %s" % (task, team), master=True)
        self.teamcomms[team].bcast(task)

    def fetch_task(self):
        self.log("Fetching task")
        task = self.mastercomm.bcast(None)
        return task

    def send_branchid(self, branchid, team):
        self.log("Sending branchid %s to team %s" % (branchid, team), master=True)
        self.teamcomms[team].bcast(branchid)

    def fetch_branchid(self):
        self.log("Fetching branchid")
        branchid = self.mastercomm.bcast(None)
        self.log("Got branchid %s" % branchid)
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

    def insert_continuation_task(self, values, oldparams, branchid, priority):
        newparams = nextparameters(values, self.freeindex, oldparams)
        if newparams is not None:
            task = ContinuationTask(taskid=self.taskid_counter,
                                    oldparams=oldparams,
                                    branchid=int(branchid),
                                    newparams=newparams,
                                    direction=+1)
            heappush(self.newtasks, (priority, task))
            self.taskid_counter += 1

            if self.compute_stability:
                stabtask = StabilityTask(taskid=self.taskid_counter,
                                         oldparams=oldparams,
                                         branchid=int(branchid),
                                         direction=+1,
                                         hint=None)
                newpriority = self.sign*stabtask.oldparams[self.freeindex]

                heappush(self.stabilitytasks, (newpriority, stabtask))
                self.taskid_counter += 1

            if self.continue_backwards:
                newparams = prevparameters(values, self.freeindex, oldparams)
                if newparams is not None:
                    task = ContinuationTask(taskid=self.taskid_counter,
                                            oldparams=oldparams,
                                            branchid=int(branchid),
                                            newparams=newparams,
                                            direction=-1)
                    self.log("Scheduling task: %s" % task, master=True)
                    heappush(self.newtasks, (priority, task))
                    self.taskid_counter += 1

                    if self.compute_stability:
                        stabtask = StabilityTask(taskid=self.taskid_counter,
                                                 oldparams=oldparams,
                                                 branchid=int(branchid),
                                                 direction=-1,
                                                 hint=None)
                        newpriority = self.sign*stabtask.oldparams[self.freeindex]

                        heappush(self.stabilitytasks, (newpriority, stabtask))
                        self.taskid_counter += 1

    def master(self, values):
        """
        The master coordinating routine.

        *Arguments*
          self.freeindex (:py:class:`tuple`)
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
        self.taskid_counter = 0

        # Branch id counter
        branchid_counter = 0

        # Next, seed the list of tasks to perform with the initial search
        self.newtasks = []  # tasks yet to be sent out
        deferredtasks = [] # tasks that we've been forced to defer as we don't have enough information to ensure they're necessary
        waittasks = {} # tasks sent out, waiting to hear back about

        # A heap of stability tasks. We keep these separately because we want
        # to process them with a lower priority than deflation and continuation
        # tasks.
        self.stabilitytasks = []
        # Decide if the user has overridden the compute_stability method or not
        self.compute_stability = "compute_stability" in self.problem.__class__.__dict__

        # A dictionary of parameters -> branches to ensure they exist,
        # to avoid race conditions
        ensure_branches = dict()

        # A set of tasks that have been invalidated by previous discoveries.
        invalidated_tasks = set()

        # If we're going downwards in continuation parameter, we need to change
        # signs in a few places
        if values[0] < values[-1]:
            self.sign = +1
            minvals = min
        else:
            self.sign = -1
            minvals = max

        # If there's only one process, show a warning. FIXME: do something more advanced so we can run anyway. 
        if self.worldcomm.size < 2:
            self.log("Defcon started with only 1 process. At least 2 processes are required (one master, one worker).\n\nLaunch with mpiexec: mpiexec -n <number of processes> python <path to file>", master=True, warning=True)
            import sys; sys.exit(1)

        # Create a journal object.
        journal = FileJournal(self.io.directory, self.parameters, self.functionals, self.freeindex, self.sign)
        try:
            # First check to see if the journal exists.
            assert(journal.exists())

            # The journal file already exists. Let's find out what we already know so we can resume our computation where we left off.
            (previous_sweep, branches, oldfreeindex, oldothers) = journal.resume()
            if previous_sweep is None: previous_sweep = values[0]

            # Check that we are continuing from the same free parameter. If not, we want to start again.
            assert(oldfreeindex==self.freeindex)

            # Check the values of the other parameters
            others = tuple(float(val[0]) for (i, val) in enumerate(self.parameters) if i != self.freeindex)
            assert(oldothers==others)

            # Everything checks out, so lets schedule the appropriate tasks. 
            branchid_counter = len(branches)

            # Set all teams to idle.
            for teamno in range(self.nteams):
                journal.team_job(teamno, "i")

            # Schedule continuation tasks for any branches that aren't done yet.
            for branchid in branches.keys():
                oldparams = branches[branchid]
                self.insert_continuation_task(values, oldparams, branchid, priority=float("-inf"))

            # We need to schedule deflation tasks at every point from where we'd completed our sweep up to previously 
            # to the furthest we've got in continuation, on every branch.
            for branchid in branches.keys():
                # Get the fixed parameters
                knownparams = [x[self.freeindex] for x in self.io.known_parameters(self.fixed, branchid)]
                oldparams = list(parameterstofloats(self.parameters, self.freeindex, values[0]))
                oldparams[self.freeindex] = previous_sweep
                newparams = nextparameters(values, self.freeindex, tuple(oldparams))
                while newparams is not None and self.sign*newparams[self.freeindex] <= self.sign*branches[branchid][self.freeindex]: 
                    # As long as we're not at the end of the parameter range and we haven't exceeded the extent
                    # of this branch, schedule a deflation. 

                    if oldparams[self.freeindex] in knownparams:
                        task = DeflationTask(taskid=self.taskid_counter,
                                             oldparams=oldparams,
                                             branchid=int(branchid),
                                             newparams=newparams)
                        self.log("Scheduling task: %s" % task, master=True)
                        self.taskid_counter += 1
                        heappush(self.newtasks, (self.sign*task.newparams[self.freeindex], task))

                    oldparams = newparams
                    newparams = nextparameters(values, self.freeindex, newparams)

        except Exception:
            # Either the journal file does not exist, or something else bad happened. 
            # Oh well, start from scratch.
            journal.setup(self.nteams, min(values), max(values))
            initialparams = parameterstofloats(self.parameters, self.freeindex, values[0])
            previous_sweep = initialparams[self.freeindex]

            # Send off initial tasks
            knownbranches = self.io.known_branches(initialparams)
            branchid_counter = len(knownbranches)
            if len(knownbranches) > 0:
                nguesses = len(knownbranches)
                self.log("Using %d known solutions at %s" % (nguesses, initialparams,), master=True)
                oldparams = initialparams
                initialparams = nextparameters(values, self.freeindex, initialparams)

                for guess in range(nguesses):
                    self.insert_continuation_task(values, oldparams, guess, priority=float("-inf"))
            else:
                self.log("Using user-supplied initial guesses at %s" % (initialparams,), master=True)
                oldparams = None
                nguesses = self.problem.number_initial_guesses(initialparams)
                for guess in range(nguesses):
                    task = DeflationTask(taskid=self.taskid_counter,
                                         oldparams=oldparams,
                                         branchid=self.taskid_counter,
                                         newparams=initialparams)
                    heappush(self.newtasks, (float("-inf"), task))
                    self.taskid_counter += 1

        # Here comes the main master loop.
        while len(self.newtasks) + len(waittasks) + len(deferredtasks) + len(self.stabilitytasks) > 0:

            if self.debug:
                self.log("DEBUG: newtasks = %s" % [(priority, str(x)) for (priority, x) in self.newtasks], master=True)
                self.log("DEBUG: waittasks = %s" % [(key, str(waittasks[key][0]), waittasks[key][1]) for key in waittasks], master=True)
                self.log("DEBUG: deferredtasks = %s" % [(priority, str(x)) for (priority, x) in deferredtasks], master=True)
                self.log("DEBUG: stabilitytasks = %s" % [(priority, str(x)) for (priority, x) in self.stabilitytasks], master=True)
                self.log("DEBUG: idleteams = %s" % idleteams, master=True)

            # Sanity check
            if len(set(idleteams).intersection(set([waittasks[key][1] for key in waittasks]))):
                self.log("ALERT: intersection of idleteams and waittasks: \n%s\n%s" % (idleteams, [(key, str(waittasks[key][0])) for key in waittasks]), master=True, warning=True)
            if set(idleteams).union(set([waittasks[key][1] for key in waittasks])) != set(range(self.nteams)):
                self.log("ALERT: team lost! idleteams and waitasks: \n%s\n%s" % (idleteams, [(key, str(waittasks[key][0])) for key in waittasks]), master=True, warning=True)

            # If there are any tasks to send out, send them.
            while len(self.newtasks) > 0 and len(idleteams) > 0:
                (priority, task) = heappop(self.newtasks)

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

                    # If there's a continuation task that hasn't reached us,
                    # we want to not send this task out now and look at it again later.
                    # This is because the currently running task might find a branch that we will need
                    # to deflate here.
                    for (t, r) in waittasks.values():
                        if isinstance(t, ContinuationTask) and self.sign*t.newparams[self.freeindex]<=self.sign*task.newparams[self.freeindex]:
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

            # And the same thing for stability tasks.
            while len(self.stabilitytasks) > 0 and len(idleteams) > 0:
                (priority, task) = heappop(self.stabilitytasks)
                idleteam = idleteams.pop(0)
                self.send_task(task, idleteam)
                waittasks[task.taskid] = (task, idleteam)

                # Write to the journal, saying that this team is now performing stability analysis.
                journal.team_job(idleteam, "s", task.oldparams, task.branchid)

            # We can't send out any more tasks, either because we have no
            # tasks to send out or we have no free processors.
            # If we aren't waiting for anything to finish, we'll exit the loop
            # here. otherwise, we wait for responses and deal with consequences.
            if len(waittasks) > 0:
                self.log("Cannot dispatch any tasks, waiting for response.", master=True)

                waiting_values = [wtask[0].oldparams for wtask in waittasks.values() if wtask[0].oldparams is not None]
                newtask_values = [ntask[1].oldparams for ntask in self.newtasks if ntask[1].oldparams is not None]
                deferred_values = [dtask[1].oldparams for dtask in deferredtasks if dtask[1].oldparams is not None]
                if len(waiting_values + newtask_values + deferred_values) > 0:
                    minparams = self.sign*minvals(waiting_values + newtask_values + deferred_values, key = lambda x: self.sign*x[self.freeindex])
                    prevparams = prevparameters(values, self.freeindex, minparams)
                    if prevparams is not None:
                        minwait = prevparams[self.freeindex]

                        tot_solutions = self.problem.number_solutions(minparams)
                        if isinf(tot_solutions): tot_solutions = '?'
                        num_solutions = len(self.io.known_branches(minparams))
                        self.log("Sweep completed <= %14.12e (%s/%s solutions)." % (minwait, num_solutions, tot_solutions), master=True)

                        # Write to the journal saying where we've completed our sweep up to.
                        journal.sweep(minwait)

                # Take this opportunity to call the garbage collector.
                gc.collect()

                response = self.worldcomm.recv(status=stat, source=MPI.ANY_SOURCE, tag=self.responsetag)

                (task, team) = waittasks[response.taskid]
                self.log("Received response %s about task %s from team %s" % (response, task, team), master=True)
                del waittasks[response.taskid]

                # Here comes the core logic of what happens for success or failure for the two
                # kinds of tasks.
                if isinstance(task, ContinuationTask):
                    if response.success:

                        # Record this entry in the journal.
                        journal.entry(team, task.oldparams, task.branchid, task.newparams, response.data['functionals'], True)

                        # The worker will keep continuing, record that fact
                        if task.direction > 0:
                            newparams = nextparameters(values, self.freeindex, task.newparams)
                        else:
                            newparams = prevparameters(values, self.freeindex, task.newparams)

                        if newparams is not None:
                            conttask = ContinuationTask(taskid=task.taskid,
                                                        oldparams=task.newparams,
                                                        branchid=task.branchid,
                                                        newparams=newparams,
                                                        direction=task.direction)
                            waittasks[task.taskid] = ((conttask, team))
                            self.log("Waiting on response for %s" % conttask, master=True)
                            journal.team_job(team, "c", task.newparams, task.branchid)
                        else:
                            idleteams.append(team)
                            journal.team_job(team, "i")

                        newtask = DeflationTask(taskid=self.taskid_counter,
                                                oldparams=task.oldparams,
                                                branchid=task.branchid,
                                                newparams=task.newparams)
                        self.taskid_counter += 1
                        heappush(self.newtasks, (self.sign*newtask.newparams[self.freeindex], newtask))

                    else:
                        # We tried to continue a branch, but the continuation died. Oh well.
                        # The team is now idle.
                        self.log("Continuation task of team %d on branch %d failed at parameters %s." % (team, task.branchid, task.newparams), master=True, warning=True)
                        idleteams.append(team)
                        journal.team_job(team, "i")

                elif isinstance(task, DeflationTask):
                    if response.success:

                        # Before processing the success, we want to make sure that we really
                        # want to keep this solution. After all, we might have been running
                        # five deflations in parallel; if they discover the same branch,
                        # we don't want them all to track it and continue it.
                        # So we check to see if this task has been invalidated
                        # by an earlier discovery.

                        if task in invalidated_tasks:
                            # * Send the worker the bad news.
                            self.send_branchid(None, team)

                            # * Remove the task from the invalidated list.
                            invalidated_tasks.remove(task)

                            # * Insert a new task --- this *might* be a dupe, or it might not
                            #   be! We need to try it again to make sure. If it is a dupe, it
                            #   won't discover anything; if it isn't, hopefully it will discover
                            #   the same (distinct) solution again.
                            if task.oldparams is not None:
                                priority = self.sign*task.newparams[self.freeindex]
                            else:
                                priority = float("-inf")
                            heappush(self.newtasks, (priority, task))

                            # The worker is now idle.
                            idleteams.append(team)
                            journal.team_job(team, "i")
                        else:
                            # OK, we're good! The search succeeded and nothing has invalidated it.
                            # In this case, we want the master to
                            # * Record any currently ongoing searches that this discovery
                            #   invalidates.
                            for (othertask, _) in waittasks.values():
                                if isinstance(othertask, DeflationTask):
                                    invalidated_tasks.add(othertask)

                            # * Allocate a new branch id for the discovered branch.
                            self.send_branchid(branchid_counter, team)

                            # * Record this new solution in the journal
                            journal.entry(team, task.oldparams, branchid_counter, task.newparams, response.data['functionals'], False)

                            # * Insert a new deflation task, to seek again with the same settings.
                            newtask = DeflationTask(taskid=self.taskid_counter,
                                                    oldparams=task.oldparams,
                                                    branchid=task.branchid,
                                                    newparams=task.newparams)
                            if task.oldparams is not None:
                                newpriority = self.sign*newtask.newparams[self.freeindex]
                            else:
                                newpriority = float("-inf")

                            heappush(self.newtasks, (newpriority, newtask))
                            self.taskid_counter += 1

                            # * Record that the worker team is now continuing that branch,
                            # if there's continuation to be done.
                            newparams = nextparameters(values, self.freeindex, task.newparams)
                            if newparams is not None:
                                conttask = ContinuationTask(taskid=task.taskid,
                                                            oldparams=task.newparams,
                                                            branchid=branchid_counter,
                                                            newparams=newparams,
                                                            direction=+1)
                                waittasks[task.taskid] = ((conttask, team))
                                self.log("Waiting on response for %s" % conttask, master=True)

                                # Write to the journal, saying that this team is now doing continuation.
                                journal.team_job(team, "c", task.newparams, task.branchid)
                            else:
                                # It's at the end of the continuation, there's no more continuation
                                # to do. Mark the team as idle.
                                idleteams.append(team)
                                journal.team_job(team, "i")

                            # * If we want to continue backwards, well, let's add that task too
                            if self.continue_backwards:
                                newparams = prevparameters(values, self.freeindex, task.newparams)
                                if newparams is not None:
                                    bconttask = ContinuationTask(taskid=self.taskid_counter,
                                                                oldparams=task.newparams,
                                                                branchid=branchid_counter,
                                                                newparams=newparams,
                                                                direction=-1)
                                    newpriority = self.sign*bconttask.newparams[self.freeindex]
                                    heappush(self.newtasks, (newpriority, bconttask))
                                    self.taskid_counter += 1

                            # We'll also make sure that any other DeflationTasks in the queue
                            # that have these parameters know about the existence of this branch.
                            if task.newparams not in ensure_branches:
                                ensure_branches[task.newparams] = set()
                            ensure_branches[task.newparams].add(branchid_counter)

                            # If the user wants us to compute stabilities, then let's
                            # do that.
                            if self.compute_stability:
                                stabtask = StabilityTask(taskid=self.taskid_counter,
                                                         oldparams=task.newparams,
                                                         branchid=branchid_counter,
                                                         direction=+1,
                                                         hint=None)
                                newpriority = self.sign*stabtask.oldparams[self.freeindex]

                                heappush(self.stabilitytasks, (newpriority, stabtask))
                                self.taskid_counter += 1

                                if self.continue_backwards:
                                    stabtask = StabilityTask(taskid=self.taskid_counter,
                                                             oldparams=task.newparams,
                                                             branchid=branchid_counter,
                                                             direction=-1,
                                                             hint=None)
                                    newpriority = self.sign*stabtask.oldparams[self.freeindex]

                                    heappush(self.stabilitytasks, (newpriority, stabtask))
                                    self.taskid_counter += 1

                            # Lastly, increment the branch counter.
                            branchid_counter += 1

                    else:
                        # As expected, deflation found nothing interesting. The team is now idle.
                        idleteams.append(team)
                        journal.team_job(team, "i")

                        # One more check. If this was an initial guess, and it failed, it might be
                        # because the user doesn't know when a problem begins to have a nontrivial
                        # branch. In this case keep trying.
                        if task.oldparams is None and branchid_counter == 0:
                            newparams = nextparameters(values, self.freeindex, task.newparams)
                            if newparams is not None:
                                newtask = DeflationTask(taskid=self.taskid_counter,
                                                        oldparams=task.oldparams,
                                                        branchid=task.branchid,
                                                        newparams=newparams)
                                newpriority = float("-inf")
                                heappush(self.newtasks, (newpriority, newtask))
                                self.taskid_counter += 1

                elif isinstance(task, StabilityTask):
                    if response.success:

                        # Record this in the journal: TODO

                        # The worker will keep continuing, record that fact
                        if task.direction > 0:
                            newparams = nextparameters(values, self.freeindex, task.oldparams)
                        else:
                            newparams = prevparameters(values, self.freeindex, task.oldparams)

                        if newparams is not None:
                            nexttask = StabilityTask(taskid=task.taskid,
                                                     branchid=task.branchid,
                                                     oldparams=newparams,
                                                     direction=task.direction,
                                                     hint=None)
                            waittasks[task.taskid] = ((nexttask, team))
                            self.log("Waiting on response for %s" % nexttask, master=True)
                            journal.team_job(team, "s", nexttask.oldparams, task.branchid)
                        else:
                            idleteams.append(team)
                            journal.team_job(team, "i")

                    else:
                        idleteams.append(team)
                        journal.team_job(team, "i")

            # Maybe we deferred some deflation tasks because we didn't have enough information to judge if they were worthwhile. Now we must reschedule.
            if len(deferredtasks) > 0:
                # Take as many as there are idle teams. This makes things run much smoother than taking them all. 
                for i in range(len(idleteams)):
                    try:
                        (priority, task) = heappop(deferredtasks)
                        heappush(self.newtasks, (priority, task))
                        self.log("Rescheduling the previously deferred task %s" % task, master=True)
                    except IndexError: break

        # All continuation tasks have been finished. Move sweepline to the end and tell the workers to quit.
        journal.sweep(values[-1])
        quit = QuitTask()
        for teamno in range(self.nteams):
            self.send_task(quit, teamno)
            journal.team_job(teamno, "q")


    def worker(self, values):
        """
        The main worker routine.

        Fetches its tasks from the master and executes them.
        """

        # Escape hatch for debugging
        if self.rank != 0:
            signal.signal(signal.SIGUSR1, lambda sig, frame: pdb.set_trace())

        task = self.fetch_task()
        while True:
            # If you add a new task, make sure to add a call to gc.collect()
            if isinstance(task, QuitTask):
                self.log("Quitting gracefully.")
                return
            elif isinstance(task, DeflationTask):
                self.log("Executing task %s" % task)

                # Check for trivial solutions
                if self.trivial_solutions is None:
                    self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, self.freeindex)

                # Set up the problem
                self.load_solution(task.oldparams, task.branchid, task.newparams)
                out = self.problem.transform_guess(task.oldparams, task.newparams, self.state); assert out is None

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
                if len(task.ensure_branches) > 0:
                    self.log("Found all necessary branches.")

                other_solutions = self.io.fetch_solutions(task.newparams, knownbranches)
                self.log("Deflating other branches %s" % knownbranches)
                bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

                self.deflation.deflate(other_solutions + self.trivial_solutions)
                (success, iters) = newton(self.residual, self.state, bcs,
                                 self.problem.nonlinear_problem,
                                 self.problem.solver,
                                 self.problem.solver_parameters(task.newparams),
                                 self.teamno, self.deflation)

                self.state_id = (None, None) # not sure if it is a solution we care about yet

                # Get the functionals now, so we can send them to the master.
                if success: functionals = self.compute_functionals(self.state, task.newparams)
                else: functionals = None

                response = Response(task.taskid, success=success, data={"functionals": functionals})
                if self.teamrank == 0:
                    self.log("Sending response %s to master" % response)
                    self.worldcomm.send(response, dest=0, tag=self.responsetag)

                # Take this opportunity to call the garbage collector.
                gc.collect()

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
                        newparams = nextparameters(values, self.freeindex, task.newparams)
                        if newparams is not None:
                            task = ContinuationTask(taskid=task.taskid,
                                                    oldparams=task.newparams,
                                                    branchid=branchid,
                                                    newparams=newparams,
                                                    direction=+1)
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
                    self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, self.freeindex)

                # Set up the problem
                self.load_solution(task.oldparams, task.branchid, task.newparams)
                self.load_parameters(task.newparams)
                knownbranches = self.io.known_branches(task.newparams)
                other_solutions = self.io.fetch_solutions(task.newparams, knownbranches)
                bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

                # Try to solve it
                self.deflation.deflate(other_solutions + self.trivial_solutions)
                (success, iters) = newton(self.residual, self.state, bcs,
                                 self.problem.nonlinear_problem,
                                 self.problem.solver,
                                 self.problem.solver_parameters(task.newparams),
                                 self.teamno, self.deflation)

                if success:
                    self.state_id = (task.newparams, task.branchid)

                    # Save it to disk with the I/O module
                    functionals = self.compute_functionals(self.state, task.newparams)
                    self.problem.monitor(task.newparams, task.branchid, self.state, functionals)
                    self.io.save_solution(self.state, functionals, task.newparams, task.branchid)

                else:
                    functionals = None
                    self.state_id = (None, None)

                response = Response(task.taskid, success=success, data={"functionals": functionals})
                if self.teamrank == 0:
                    self.log("Sending response %s to master" % response)
                    self.worldcomm.send(response, dest=0, tag=self.responsetag)

                # Take this opportunity to call the garbage collector.
                gc.collect()

                if task.direction > 0:
                    newparams = nextparameters(values, self.freeindex, task.newparams)
                else:
                    newparams = prevparameters(values, self.freeindex, task.newparams)

                if success and newparams is not None:
                    task = ContinuationTask(taskid=task.taskid,
                                            oldparams=task.newparams,
                                            branchid=task.branchid,
                                            newparams=newparams,
                                            direction=task.direction)
                else:
                    task = self.fetch_task()

            elif isinstance(task, StabilityTask):
                self.log("Executing task %s" % task)

                try:
                    self.load_solution(task.oldparams, task.branchid, -1)
                    self.load_parameters(task.oldparams)

                    d = self.problem.compute_stability(task.oldparams, task.branchid, self.state, hint=task.hint)
                    success = True
                    response = Response(task.taskid, success=success, data={"stable": d["stable"]})
                except:
                    import traceback; traceback.print_exc()
                    success = False
                    response = Response(task.taskid, success=success)

                if success:
                    # Save the data to disk with the I/O module
                    self.io.save_stability(d["stable"], d.get("eigenvalues", []), d.get("eigenfunctions", []), task.oldparams, task.branchid)

                if self.teamrank == 0:
                    self.log("Sending response %s to master" % response)
                    self.worldcomm.send(response, dest=0, tag=self.responsetag)

                # Take this opportunity to call the garbage collector.
                gc.collect()

                if task.direction > 0:
                    newparams = nextparameters(values, self.freeindex, task.oldparams)
                else:
                    newparams = prevparameters(values, self.freeindex, task.oldparams)

                if success and newparams is not None:
                    task = StabilityTask(taskid=task.taskid,
                                         oldparams=newparams,
                                         branchid=task.branchid,
                                         direction=task.direction,
                                         hint=d.get("hint", None))
                else:
                    task = self.fetch_task()

    def bifurcation_diagram(self, functional, fixed={}, style="ok", **kwargs):
        if self.rank != 0:
            return

        import matplotlib.pyplot as plt
        if "linewidth" not in kwargs: kwargs["linewidth"] = 2
        if "markersize" not in kwargs: kwargs["linewidth"] = 1

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
        self.freeindex = freeindices[0]

        for branchid in range(self.io.max_branch() + 1):
            xs = []
            ys = []
            params = self.io.known_parameters(fixed, branchid)
            funcs = self.io.fetch_functionals(params, branchid)
            for i in xrange(0, len(params)):
                param = params[i]
                func = funcs[i]
                xs.append(param[self.freeindex])
                ys.append(func[funcindex])
            plt.plot(xs, ys, style, **kwargs)

        plt.grid()
        plt.xlabel(self.parameters[self.freeindex][2])
        plt.ylabel(self.functionals[funcindex][2])
