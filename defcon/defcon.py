# -*- coding: utf-8 -*-

from operatordeflation import ShiftedDeflation
from parallellayout import ranktoteamno, teamnotoranks
from parametertools import Parameters
from newton import newton
from tasks import QuitTask, ContinuationTask, DeflationTask, StabilityTask, Response
from iomodule import remap_c_streams
from journal import FileJournal, task_to_code
from graph import DefconGraph

import backend

from mpi4py   import MPI
from petsc4py import PETSc
from numpy    import isinf

import time
import sys
import gc
import os
import traceback

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

class DefconThread(object):
    """
    The base class for DefconWorker/DefconMaster.
    """
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.functionals = problem.functionals()

        self.deflation = kwargs.get("deflation", None)
        self.teamsize  = kwargs.get("teamsize", 1)
        self.verbose   = kwargs.get("verbose", True)
        self.debug     = kwargs.get("debug", False)
        self.logfiles  = kwargs.get("logfiles", False)
        self.continue_backwards = kwargs.get("continue_backwards", True)
        self.worldcomm = kwargs["comm"]

        self.configure_comms()
        self.configure_logs()

    def configure_io(self, parameters):
        """
        parameters is a parametertools.Parameters object.
        """
        io = self.problem.io()
        # Don't construct the FunctionSpace on the master, it's a waste of memory
        if self.rank == 0:
            io.setup(parameters.parameters, self.functionals, None)
        else:
            io.setup(parameters.parameters, self.functionals, self.function_space)

        self.io = io

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


class DefconWorker(DefconThread):
    """
    This class handles the actual execution of the tasks necessary
    to do deflated continuation.
    """
    def __init__(self, problem, **kwargs):
        DefconThread.__init__(self, problem, **kwargs)

        # A map from the type of task we've received to the code that handles it.
        self.callbacks = {DeflationTask:    self.deflation_task,
                          StabilityTask:    self.stability_task,
                          ContinuationTask: self.continuation_task}

    def run(self, parameters, freeparam):

        self.parameters = parameters

        # Fetch data from the problem.
        self.mesh = self.problem.mesh(PETSc.Comm(self.teamcomm))
        self.function_space = self.problem.function_space(self.mesh)

        self.state = backend.Function(self.function_space)
        self.trivial_solutions = None
        self.residual = self.problem.residual(self.state, parameters.constants, backend.TestFunction(self.function_space))

        self.configure_io(parameters)
        self.construct_deflation(parameters)

        # We keep track of what solution we actually have in memory in self.state
        # for efficiency. FIXME: investigate if this actually saves us any
        # time; print out cache hits/misses in self.load_solution
        self.state_id = (None, None)

        task = None
        while True:
            gc.collect()

            if task is None:
                task = self.fetch_task()

            if isinstance(task, QuitTask):
                self.log("Quitting gracefully")
                return
            else:
                self.log("Executing task %s" % task)
                task = self.callbacks[task.__class__](task)
        return

    def construct_deflation(self, parameters):
        if self.deflation is None:
            self.deflation = ShiftedDeflation(self.problem, power=2, shift=1)
        self.deflation.set_parameters(parameters.constants)

    def log(self, msg, warning=False):
        DefconThread.log(self, msg, master=False, warning=warning)

    def fetch_task(self):
        self.log("Fetching task")
        task = self.mastercomm.bcast(None)
        return task

    def fetch_response(self):
        self.log("Fetching response")
        response = self.mastercomm.bcast(None)
        return response

    def send_response(self, response):
        if self.teamrank == 0:
            self.log("Sending response %s" % response)
            self.worldcomm.send(response, dest=0, tag=self.responsetag)

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
        self.parameters.update_constants(params)

    def compute_functionals(self, solution):
        funcs = []
        for functional in self.functionals:
            func = functional[0]
            j = func(solution, self.parameters.constants)
            assert isinstance(j, float)
            funcs.append(j)
        return funcs

    def deflation_task(self, task):
        # First, load trivial solutions
        if self.trivial_solutions is None:
            self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, task.freeindex)

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

        # Deflate and solve
        self.deflation.deflate(other_solutions + self.trivial_solutions)
        (success, iters) = newton(self.residual, self.state, bcs,
                         self.problem.nonlinear_problem,
                         self.problem.solver,
                         self.problem.solver_parameters(task.newparams, task.__class__),
                         self.teamno, self.deflation)

        self.state_id = (None, None) # not sure if it is a solution we care about yet

        # Get the functionals now, so we can send them to the master.
        if success: functionals = self.compute_functionals(self.state)
        else: functionals = None

        response = Response(task.taskid, success=success, data={"functionals": functionals})
        self.send_response(response)

        if not success:
            # that's it, move on
            return

        # Get a Response from the master telling us if we should go ahead or not
        response = self.fetch_response()
        if not response.success:
            # the master has instructed us not to bother with this solution.
            # move on.
            return
        branchid = response.data["branchid"]

        # We do care about this solution, so record the fact we have it in memory
        self.state_id = (task.newparams, branchid)
        # Save it to disk with the I/O module
        self.log("Found new solution at parameters %s (branchid=%s) with functionals %s" % (task.newparams, branchid, functionals))
        self.problem.monitor(task.newparams, branchid, self.state, functionals)
        self.io.save_solution(self.state, functionals, task.newparams, branchid)
        self.log("Saved solution to %s to disk" % task)

        # Automatically start onto the continuation
        newparams = self.parameters.next(task.newparams, task.freeindex)
        if newparams is not None:
            task = ContinuationTask(taskid=task.taskid,
                                    oldparams=task.newparams,
                                    freeindex=task.freeindex,
                                    branchid=branchid,
                                    newparams=newparams,
                                    direction=+1)
            return task
        else:
            # Reached the end of the continuation, don't want to continue, move on
            return

    def continuation_task(self, task):
        # Check for trivial solutions
        if self.trivial_solutions is None:
            self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, task.freeindex)

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
                         self.problem.solver_parameters(task.newparams, task.__class__),
                         self.teamno, self.deflation)

        if success:
            self.state_id = (task.newparams, task.branchid)

            # Save it to disk with the I/O module
            functionals = self.compute_functionals(self.state)
            self.problem.monitor(task.newparams, task.branchid, self.state, functionals)
            self.io.save_solution(self.state, functionals, task.newparams, task.branchid)

        else:
            functionals = None
            self.state_id = (None, None)

        response = Response(task.taskid, success=success, data={"functionals": functionals})
        self.send_response(response)

        if not success:
            # Continuation didn't work. Move on.
            # FIXME: we could make this adaptive; try halving the step and doing
            # two steps?
            return

        if task.direction > 0:
            newparams = self.parameters.next(task.newparams, task.freeindex)
        else:
            newparams = self.parameters.previous(task.newparams, task.freeindex)

        if newparams is None:
            # we have no more continuation to do, move on.
            return
        else:
            task = ContinuationTask(taskid=task.taskid,
                                    oldparams=task.newparams,
                                    freeindex=task.freeindex,
                                    branchid=task.branchid,
                                    newparams=newparams,
                                    direction=task.direction)
            return task

    def stability_task(self, task):
        options = self.problem.solver_parameters(task.oldparams, task.__class__)
        opts = PETSc.Options()
        for k in options:
            opts[k] = options[k]

        try:
            # Have we already computed the stability? For good reasons
            # we sometimes get the same task twice.
            stab = self.io.fetch_stability(task.oldparams, [task.branchid])
            success = True
            d = {"stable": stab[0]}
            response = Response(task.taskid, success=success, data={"stable": stab[0]})
        except IOError:
            # We don't know whether it's stable or not, compute.
            try:
                self.load_solution(task.oldparams, task.branchid, -1)
                self.load_parameters(task.oldparams)

                d = self.problem.compute_stability(task.oldparams, task.branchid, self.state, hint=task.hint)
                success = True
                response = Response(task.taskid, success=success, data={"stable": d["stable"]})
            except:
                self.log("Stability task %s failed; exception follows." % task, warning=True)
                traceback.print_exc()
                success = False
                response = Response(task.taskid, success=success)

        if success:
            # Save the data to disk with the I/O module
            self.io.save_stability(d["stable"], d.get("eigenvalues", []), d.get("eigenfunctions", []), task.oldparams, task.branchid)

        # Send the news to master.
        self.send_response(response)

        if not success:
            # Couldn't compute stability. Likely something is wrong. Abort and get
            # another task.
            return

        # If we're successful, we expect a command from master: should we go ahead, or not?
        response = self.fetch_response()

        if not response.success:
            # Master doesn't want us to continue. This is probably because the
            # ContinuationTask that needs to be finished before we can compute
            # its stability is still ongoing. We'll pick it up later.
            return

        if task.direction > 0:
            newparams = self.parameters.next(task.oldparams, task.freeindex)
        else:
            newparams = self.parameters.previous(task.oldparams, task.freeindex)

        if newparams is not None:
            task = StabilityTask(taskid=task.taskid,
                                 oldparams=newparams,
                                 freeindex=task.freeindex,
                                 branchid=task.branchid,
                                 direction=task.direction,
                                 hint=d.get("hint", None))
            return task
        else:
            # No more continuation to do, we're finished
            return

class DefconMaster(DefconThread):
    """
    This class implements the core logic of running deflated continuation
    in parallel.
    """
    def __init__(self, *args, **kwargs):
        DefconThread.__init__(self, *args, **kwargs)

        # A map from the type of task we're dealing with to the code that handles it.
        self.callbacks = {DeflationTask:    self.deflation_task,
                          StabilityTask:    self.stability_task,
                          ContinuationTask: self.continuation_task}

    def log(self, msg, warning=False):
        DefconThread.log(self, msg, master=True, warning=warning)

    def send_task(self, task, team):
        self.log("Sending task %s to team %s" % (task, team))
        self.teamcomms[team].bcast(task)

    def send_response(self, response, team):
        self.log("Sending response %s to team %s" % (response, team))
        self.teamcomms[team].bcast(response)

    def fetch_response(self):
        response = self.worldcomm.recv(source=MPI.ANY_SOURCE, tag=self.responsetag)
        return response

    def seed_initial_tasks(self, parameters, values, freeindex):
        # Queue initial tasks
        initialparams = parameters.floats(value=values[0], freeindex=freeindex)

        # Send off initial tasks
        knownbranches = self.io.known_branches(initialparams)
        self.branchid_counter = len(knownbranches)
        if len(knownbranches) > 0:
            nguesses = len(knownbranches)
            self.log("Using %d known solutions at %s" % (nguesses, initialparams,))

            for guess in range(nguesses):
                self.insert_continuation_task(initialparams, freeindex, guess, priority=float("-inf"))
        else:
            self.log("Using user-supplied initial guesses at %s" % (initialparams,))
            oldparams = None
            nguesses = self.problem.number_initial_guesses(initialparams)
            for guess in range(nguesses):
                task = DeflationTask(taskid=self.taskid_counter,
                                     oldparams=oldparams,
                                     freeindex=freeindex,
                                     branchid=self.taskid_counter,
                                     newparams=initialparams)
                self.graph.push(task, float("-inf"))
                self.taskid_counter += 1

    def finished(self):
        return self.graph.all_tasks() == 0

    def debug_print(self):
        if self.debug:
            self.graph.debug(self.log)
            self.log("DEBUG: idle_teams = %s" % self.idle_teams)

        # Also, a sanity check: idle_teams and busy_teams should be a disjoint partitioning of range(self.nteams)
        busy_teams = set([self.graph.wait_tasks[key][1] for key in self.graph.wait_tasks])
        if len(set(self.idle_teams).intersection(busy_teams)) > 0:
            self.log("ALERT: intersection of idle_teams and wait_tasks: \n%s\n%s" % (self.idle_teams, [(key, str(self.graph.wait_tasks[key][0])) for key in self.wait_tasks]), warning=True)
        if set(self.idle_teams).union(busy_teams) != set(range(self.nteams)):
            self.log("ALERT: team lost! idle_teams and wait_tasks: \n%s\n%s" % (self.idle_teams, [(key, str(self.graph.wait_tasks[key][0])) for key in self.graph.wait_tasks]), warning=True)

    def run(self, parameters, freeparam):
        self.parameters = parameters
        freeindex = self.parameters.labels.index(freeparam)

        self.configure_io(parameters)

        # List of idle teams
        self.idle_teams = range(self.nteams)

        # Task id counter
        self.taskid_counter = 0

        # Branch id counter
        self.branchid_counter = 0

        # Should we insert stability tasks? Let's see if the user
        # has overridden the compute_stability method or not
        self.compute_stability = "compute_stability" in self.problem.__class__.__dict__

        self.graph = DefconGraph()

        # We need to keep a map of parameters -> branches.
        # FIXME: make disk writes atomic and get rid of this.
        self.ensure_branches = {}

        # In parallel, we might make a discovery with deflation that invalidates
        # the results of other deflations ongoing. This set keeps track of the tasks
        # whose results we need to ignore.
        self.invalidated_tasks = set()

        # A map from (branchid, freeindex) -> (min, max) parameter value known.
        # This is needed to keep stability tasks from outrunning the
        # continuation on which they depend.
        self.branch_extent = {}

        # If we're going downwards in continuation parameter, we need to change
        # signs in a few places
        self.signs = []
        self.minvals = []
        for label in self.parameters.labels:
            values = self.parameters.values[label]
            if values[0] < values[-1]:
                self.signs.append(+1)
                self.minvals.append(min)
            else:
                self.signs.append(-1)
                self.minvals.append(max)

        # Initialise Journal
        self.journal = FileJournal(self.io.directory, self.parameters.parameters, self.functionals, freeindex, self.signs[freeindex])
        self.journal.setup(self.nteams, min(self.parameters.values[freeparam]), max(self.parameters.values[freeparam]))
        self.journal.sweep(self.parameters.values[freeparam][0])

        # Seed initial tasks
        self.seed_initial_tasks(parameters, parameters.values[freeparam], freeindex)

        # The main master loop.
        while not self.finished():
            self.debug_print()

            # Dispatch any tasks that can be dispatched
            while self.graph.executable_tasks() > 0 and len(self.idle_teams) > 0:
                self.dispatch_task()

            # We can't send out any more tasks, either because we have no
            # tasks to send out or we have no free processors.
            # If we aren't waiting for anything to finish, we'll exit the loop
            # here. otherwise, we wait for responses and deal with consequences.
            if len(self.graph.waiting()) > 0:
                self.compute_sweep(freeindex, freeparam)
                self.log("Cannot dispatch any tasks, waiting for response.")
                gc.collect()

                response = self.fetch_response()
                self.handle_response(response)

            self.graph.reschedule(len(self.idle_teams))

        # Finished the main loop, tell everyone to quit
        self.journal.sweep(values[-1])
        quit = QuitTask()
        for teamno in range(self.nteams):
            self.send_task(quit, teamno)
            self.journal.team_job(teamno, "q")

    def handle_response(self, response):
        (task, team) = self.graph.finish(response.taskid)
        self.log("Received response %s about task %s from team %s" % (response, task, team))
        self.callbacks[task.__class__](task, team, response)

    def dispatch_task(self):
        (task, priority) = self.graph.pop()

        send = True
        if isinstance(task, DeflationTask):
            knownbranches = self.io.known_branches(task.newparams)
            if task.newparams in self.ensure_branches:
                knownbranches = knownbranches.union(self.ensure_branches[task.newparams])

            if len(knownbranches) >= self.problem.number_solutions(task.newparams):
            # We've found all the branches the user's asked us for, let's roll.
                self.log("Master not dispatching %s because we have enough solutions" % task)
                return

            # If there's a continuation task that hasn't reached us,
            # we want to not send this task out now and look at it again later.
            # This is because the currently running task might find a branch that we will need
            # to deflate here.
            for t in self.graph.waiting(ContinuationTask):
                if task.freeindex == t.freeindex and self.signs[t.freeindex]*t.newparams[task.freeindex]<=self.signs[t.freeindex]*task.newparams[task.freeindex] and t.direction == +1:
                    send = False
                    break

        if send:
            # OK, we're happy to send it out. Let's tell it any new information
            # we've found out since we scheduled it.
            if hasattr(task, 'ensure'):
                if task.newparams in self.ensure_branches:
                    task.ensure(self.ensure_branches[task.newparams])
            idleteam = self.idle_teams.pop(0)
            self.send_task(task, idleteam)
            self.graph.wait(task.taskid, idleteam, task)

            if hasattr(task, 'newparams'):
                self.journal.team_job(idleteam, task_to_code(task), task.newparams, task.branchid)
            else:
                self.journal.team_job(idleteam, task_to_code(task), task.oldparams, task.branchid)
        else:
            # Best reschedule for later, as there is still pertinent information yet to come in. 
            self.log("Deferring task %s." % task)
            self.graph.defer(task, priority)

    def deflation_task(self, task, team, response):
        if not response.success:
            # As is typical, deflation found nothing interesting. The team
            # is now idle.
            self.idle_team(team)

            # One more check. If this was an initial guess, and it failed, it might be
            # because the user doesn't know when a problem begins to have a nontrivial
            # branch. In this case keep trying.
            if task.oldparams is None and self.branchid_counter == 0:
                newparams = self.parameters.next(task.newparams, task.freeindex)
                if newparams is not None:
                    newtask = DeflationTask(taskid=self.taskid_counter,
                                            oldparams=task.oldparams,
                                            freeindex=task.freeindex,
                                            branchid=task.branchid,
                                            newparams=newparams)
                    newpriority = float("-inf")
                    self.graph.push(newtask, newpriority)
                    self.taskid_counter += 1
            return

        # OK. So we were successful. But, Before processing the success, we want
        # to make sure that we really want to keep this solution. After all, we
        # might have been running five deflations in parallel; if they discover
        # the same branch, we don't want them all to track it and continue it.
        # So we check to see if this task has been invalidated by an earlier
        # discovery.

        if task in self.invalidated_tasks:
            # * Send the worker the bad news.
            self.log("Task %s has been invalidated" % task)
            responseback = Response(task.taskid, success=False)
            self.send_response(responseback, team)

            # * Remove the task from the invalidated list.
            self.invalidated_tasks.remove(task)

            # * Insert a new task --- this *might* be a dupe, or it might not
            #   be! We need to try it again to make sure. If it is a dupe, it
            #   won't discover anything; if it isn't, hopefully it will discover
            #   the same (distinct) solution again.
            if task.oldparams is not None:
                priority = self.signs[task.freeindex]*task.newparams[task.freeindex]
            else:
                priority = float("-inf")
            self.graph.push(task, priority)

            # The worker is now idle.
            self.idle_team(team)
            return

        # OK, we're good! The search succeeded and nothing has invalidated it.
        # In this case, we want the master to
        # * Record any currently ongoing searches that this discovery
        #   invalidates.
        for othertask in self.graph.waiting(DeflationTask):
            self.log("Invalidating %s" % othertask)
            self.invalidated_tasks.add(othertask)

        # * Allocate a new branch id for the discovered branch.
        branchid = self.branchid_counter
        self.branchid_counter += 1

        responseback = Response(task.taskid, success=True, data={"branchid": branchid})
        self.send_response(responseback, team)

        # * Record the branch extents
        for freeindex in range(len(self.parameters.labels)):
            self.branch_extent[(branchid, freeindex)] = [task.newparams[freeindex], task.newparams[freeindex]]

        # * Record this new solution in the journal
        self.journal.entry(team, task.oldparams, branchid, task.newparams, response.data["functionals"], False)

        # * Insert a new deflation task, to seek again with the same settings.
        newtask = DeflationTask(taskid=self.taskid_counter,
                                oldparams=task.oldparams,
                                freeindex=task.freeindex,
                                branchid=task.branchid,
                                newparams=task.newparams)
        if task.oldparams is not None:
            newpriority = self.signs[task.freeindex]*newtask.newparams[task.freeindex]
        else:
            newpriority = float("-inf")

        self.graph.push(newtask, newpriority)
        self.taskid_counter += 1

        # * Record that the worker team is now continuing that branch,
        # if there's continuation to be done.
        newparams = self.parameters.next(task.newparams, task.freeindex)
        if newparams is not None:
            conttask = ContinuationTask(taskid=task.taskid,
                                        oldparams=task.newparams,
                                        freeindex=task.freeindex,
                                        branchid=branchid,
                                        newparams=newparams,
                                        direction=+1)
            self.graph.wait(task.taskid, team, conttask)
            self.log("Waiting on response for %s" % conttask)
            # Write to the journal, saying that this team is now doing continuation.
            self.journal.team_job(team, "c", task.newparams, branchid)
        else:
            # It's at the end of the continuation, there's no more continuation
            # to do. Mark the team as idle.
            self.idle_team(team)

        # * Now let's ask the user if they want to do anything special,
        #   e.g. insert new tasks going in another direction.
        userin = ContinuationTask(taskid=self.taskid_counter,
                                  oldparams=task.newparams,
                                  freeindex=task.freeindex,
                                  branchid=branchid,
                                  newparams=newparams,
                                  direction=+1)
        self.taskid_counter += 1
        self.process_user_tasks(userin)

        # * If we want to continue backwards, well, let's add that task too
        if self.continue_backwards:
            newparams = self.parameters.previous(task.newparams, task.freeindex)
            if newparams is not None:
                bconttask = ContinuationTask(taskid=self.taskid_counter,
                                            oldparams=task.newparams,
                                            freeindex=task.freeindex,
                                            branchid=branchid,
                                            newparams=newparams,
                                            direction=-1)
                newpriority = self.signs[task.freeindex]*bconttask.newparams[task.freeindex]
                self.graph.push(bconttask, newpriority)
                self.taskid_counter += 1

        # We'll also make sure that any other DeflationTasks in the queue
        # that have these parameters know about the existence of this branch.
        if task.newparams not in self.ensure_branches:
            self.ensure_branches[task.newparams] = set()
        self.ensure_branches[task.newparams].add(branchid)

        # If the user wants us to compute stabilities, then let's
        # do that.
        if self.compute_stability:
            stabtask = StabilityTask(taskid=self.taskid_counter,
                                     oldparams=task.newparams,
                                     freeindex=task.freeindex,
                                     branchid=branchid,
                                     direction=+1,
                                     hint=None)
            newpriority = self.signs[task.freeindex]*stabtask.oldparams[task.freeindex]
            self.graph.push(stabtask, newpriority)
            self.taskid_counter += 1

            if self.continue_backwards and task.oldparams is not None:
                stabtask = StabilityTask(taskid=self.taskid_counter,
                                         oldparams=task.newparams,
                                         freeindex=task.freeindex,
                                         branchid=branchid,
                                         direction=-1,
                                         hint=None)
                newpriority = self.signs[task.freeindex]*stabtask.oldparams[task.freeindex]
                self.graph.push(stabtask, newpriority)
                self.taskid_counter += 1

        # Phew! What a lot of bookkeeping. That's it.
        return

    def continuation_task(self, task, team, response):
        if not response.success:
            # We tried to continue a branch, but the continuation died. Oh well.
            # The team is now idle.
            self.log("Continuation task of team %d on branch %d failed at parameters %s." % (team, task.branchid, task.newparams), warning=True)
            self.idle_team(team)
            return

        # Record success.
        self.journal.entry(team, task.oldparams, task.branchid, task.newparams, response.data["functionals"], True)

        # Update the branch extent.
        if (task.branchid, task.freeindex) not in self.branch_extent:
            self.branch_extent[(task.branchid, task.freeindex)] = [task.oldparams[task.freeindex], task.newparams[task.freeindex]]
        else:
            if task.direction > 0:
                self.branch_extent[(task.branchid, task.freeindex)][1] = task.newparams[task.freeindex]
            else:
                self.branch_extent[(task.branchid, task.freeindex)][0] = task.newparams[task.freeindex]

        # The worker will keep continuing, record that fact
        if task.direction > 0:
            newparams = self.parameters.next(task.newparams, task.freeindex)
        else:
            newparams = self.parameters.previous(task.newparams, task.freeindex)

        if newparams is None:
            # No more continuation to do, the team is now idle.
            self.idle_team(team)
        else:
            conttask = ContinuationTask(taskid=task.taskid,
                                        oldparams=task.newparams,
                                        freeindex=task.freeindex,
                                        branchid=task.branchid,
                                        newparams=newparams,
                                        direction=task.direction)
            self.graph.wait(task.taskid, team, conttask)
            self.log("Waiting on response for %s" % conttask)
            self.journal.team_job(team, task_to_code(conttask))

        # Now let's ask the user if they want to do anything special,
        # e.g. insert new tasks going in another direction.
        userin = ContinuationTask(taskid=self.taskid_counter,
                                  oldparams=task.newparams,
                                  freeindex=task.freeindex,
                                  branchid=task.branchid,
                                  newparams=newparams,
                                  direction=+1)
        self.taskid_counter += 1
        self.process_user_tasks(userin)

        # Whether there is another continuation task to insert or not,
        # we have a deflation task to insert.
        newtask = DeflationTask(taskid=self.taskid_counter,
                                oldparams=task.oldparams,
                                freeindex=task.freeindex,
                                branchid=task.branchid,
                                newparams=task.newparams)
        self.taskid_counter += 1
        newpriority = self.signs[task.freeindex]*newtask.newparams[task.freeindex]
        self.graph.push(newtask, newpriority)

    def stability_task(self, task, team, response):
        if not response.success:
            self.idle_team(team)
            return

        # Check if this is the end of the known data: if it is, don't
        # continue
        success = True
        if task.direction > 0:
            if self.signs[task.freeindex]*task.oldparams[task.freeindex] >= self.signs[task.freeindex]*self.branch_extent[(task.branchid, task.freeindex)][1]:
                success = False
        else:
            if self.signs[task.freeindex]*task.oldparams[task.freeindex] <= self.signs[task.freeindex]*self.branch_extent[(task.branchid, task.freeindex)][0]:
                success = False
        responseback = Response(task.taskid, success=success)
        self.send_response(responseback, team)

        if not success:
            # We've told the worker to stop. The team is now idle.
            self.idle_team(team)

            # If the continuation is still ongoing, we'll insert another stability
            # task into the queue.
            continuation_ongoing = False
            for currenttask in self.graph.waiting(ContinuationTask):
                if currenttask.branchid == task.branchid:
                    continuation_ongoing = True
                    break

            if not continuation_ongoing:
                for currenttask in self.graph.executable(ContinuationTask):
                    if currenttask.branchid == task.branchid:
                        continuation_ongoing = True
                        break

            if continuation_ongoing:
                # Insert another StabilityTask into the queue.
                if task.direction > 0:
                    newparams = self.parameters.next(task.oldparams, task.freeindex)
                else:
                    newparams = self.parameters.previous(task.oldparams, task.freeindex)

                if newparams is not None:
                    nexttask = StabilityTask(taskid=task.taskid,
                                             branchid=task.branchid,
                                             freeindex=task.freeindex,
                                             oldparams=newparams,
                                             direction=task.direction,
                                             hint=None)
                    newpriority = self.signs[task.freeindex]*nexttask.oldparams[task.freeindex]
                    self.graph.push(nexttask, newpriority)
            return

        # The worker will keep continuing, record that fact
        if task.direction > 0:
            newparams = self.parameters.next(task.oldparams, task.freeindex)
        else:
            newparams = self.parameters.previous(task.oldparams, task.freeindex)

        if newparams is not None:
            nexttask = StabilityTask(taskid=task.taskid,
                                     branchid=task.branchid,
                                     freeindex=task.freeindex,
                                     oldparams=newparams,
                                     direction=task.direction,
                                     hint=None)
            self.graph.wait(task.taskid, team, nexttask)
            self.log("Waiting on response for %s" % nexttask)
            self.journal.team_job(team, task_to_code(nexttask), nexttask.oldparams, task.branchid)
        else:
            self.idle_team(team)

    def insert_continuation_task(self, oldparams, freeindex, branchid, priority):
        newparams = self.parameters.next(oldparams, freeindex)
        branchid  = int(branchid)
        if newparams is not None:
            task = ContinuationTask(taskid=self.taskid_counter,
                                    oldparams=oldparams,
                                    freeindex=freeindex,
                                    branchid=branchid,
                                    newparams=newparams,
                                    direction=+1)
            self.graph.push(task, priority)
            self.taskid_counter += 1

            if self.compute_stability:
                stabtask = StabilityTask(taskid=self.taskid_counter,
                                         oldparams=oldparams,
                                         freeindex=freeindex,
                                         branchid=branchid,
                                         direction=+1,
                                         hint=None)
                newpriority = self.signs[freeindex]*stabtask.oldparams[freeindex]
                self.graph.push(stabtask, newpriority)
                self.taskid_counter += 1

            if self.continue_backwards:
                newparams = self.parameters.previous(oldparams, freeindex)
                if newparams is not None:
                    task = ContinuationTask(taskid=self.taskid_counter,
                                            oldparams=oldparams,
                                            freeindex=freeindex,
                                            branchid=branchid,
                                            newparams=newparams,
                                            direction=-1)
                    self.log("Scheduling task: %s" % task)
                    self.graph.push(task, priority)
                    self.taskid_counter += 1

                    if self.compute_stability:
                        stabtask = StabilityTask(taskid=self.taskid_counter,
                                                 oldparams=oldparams,
                                                 freeindex=freeindex,
                                                 branchid=branchid,
                                                 direction=-1,
                                                 hint=None)
                        newpriority = self.signs[freeindex]*stabtask.oldparams[freeindex]
                        self.graph.push(stabtask, newpriority)
                        self.taskid_counter += 1

    def idle_team(self, team):
        self.idle_teams.append(team)
        self.journal.team_job(team, "i")

    def compute_sweep(self, freeindex, freeparam):
        minparams = self.graph.sweep(self.minvals[freeindex], freeindex)
        if minparams is not None:
            prevparams = self.parameters.previous(minparams, freeindex)
            if prevparams is not None:
                minwait = prevparams[freeindex]

                tot_solutions = self.problem.number_solutions(minparams)
                if isinf(tot_solutions): tot_solutions = '?'
                num_solutions = len(self.io.known_branches(minparams))
                self.log("Deflation sweep completed <= %14.12e (%s/%s solutions)." % (minwait, num_solutions, tot_solutions))
                # Write to the journal saying where we've completed our sweep up to.
                self.journal.sweep(minwait)

        elif len(self.graph.executable(StabilityTask)) > 0:
            # We have only stability tasks to do.
            minwait = self.parameters.values[freeparam][-1]
            self.journal.sweep(minwait)

    def process_user_tasks(self, userin):
        user_tasks = self.problem.branch_found(userin)
        for (j, user_task) in enumerate(user_tasks):
            assert user_task.taskid == userin.taskid + j + 1

            if hasattr(user_task, 'newparams'):
                user_task.newparams = self.parameters.next(user_task.oldparams, user_task.freeindex)
                if user_task.newparams is None:
                    self.log("Warning: disregarding user-inserted task %s" % user_task, warning=True)
                    continue

            priority = user_task.oldparams[user_task.freeindex]
            self.log("Registering user-inserted task %s" % user_task)
            self.graph.push(user_task, priority)
            self.taskid_counter += 1

