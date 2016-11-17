from newton import newton
from tasks import QuitTask, ContinuationTask, DeflationTask, StabilityTask, Response
from mg import create_dm
from thread import DefconThread
from operatordeflation import ShiftedDeflation

import backend
from petsc4py import PETSc

import traceback

class DefconWorker(DefconThread):
    """
    This class handles the actual execution of the tasks necessary
    to do deflated continuation.
    """
    def __init__(self, problem, **kwargs):
        DefconThread.__init__(self, problem, **kwargs)

        # Record gc_frequency from kwargs
        self.gc_frequency = kwargs.get("gc_frequency")

        # A map from the type of task we've received to the code that handles it.
        self.callbacks = {DeflationTask:    self.deflation_task,
                          StabilityTask:    self.stability_task,
                          ContinuationTask: self.continuation_task}

    def run(self, parameters, freeparam):

        self.parameters = parameters

        # Fetch data from the problem.
        self.mesh = self.problem.mesh(PETSc.Comm(self.teamcomm))
        self.function_space = self.problem.function_space(self.mesh)
        self.dm = create_dm(self.function_space, self.problem)

        # Configure garbage collection frequency:
        if self.gc_frequency is None:
            dofs_per_core = self.function_space.dim() / self.teamcomm.size
            if dofs_per_core > 100000: self.gc_frequency = 1
            if dofs_per_core < 10000:  self.gc_frequency = 100
            else:                      self.gc_frequency = 10

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
            self.collect()

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

    def fetch_response(self, block=False):
        self.log("Fetching response")
        response = self.mastercomm.bcast(None)

        if block:
            self.mastercomm.barrier()

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
        other_solutions = self.io.fetch_solutions(task.newparams, task.ensure_branches)
        bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

        # Deflate and solve
        self.log("Deflating other branches %s" % task.ensure_branches)
        self.deflation.deflate(other_solutions + self.trivial_solutions)
        (success, iters) = newton(self.residual, self.state, bcs,
                         self.problem.nonlinear_problem,
                         self.problem.solver,
                         self.problem.solver_parameters(task.newparams, task.__class__),
                         self.teamno, self.deflation, self.dm)

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

        # We want to add one more synchronisation point with master, so that it doesn't go
        # haring off with this solution until it's written to disk (i.e. now)
        response = self.fetch_response(block=True)

        # Automatically start onto the continuation
        newparams = self.parameters.next(task.newparams, task.freeindex)
        if newparams is not None:
            task = ContinuationTask(taskid=task.taskid,
                                    oldparams=task.newparams,
                                    freeindex=task.freeindex,
                                    branchid=branchid,
                                    newparams=newparams,
                                    direction=+1)
            task.ensure(response.data["ensure_branches"])
            return task
        else:
            # Reached the end of the continuation, don't want to continue, move on
            return

    def continuation_task(self, task):
        # Check for trivial solutions
        if self.trivial_solutions is None:
            self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, task.freeindex)

        # Set up the problem

        # We want each (branch, freeindex) pairing to have a unique branch id,
        # to allow for one HDF5 file per branchid (otherwise two teams could
        # write to the same file, causing a race condition). This case occurs
        # when we do multi-parameter continuation, and need to continue a branch
        # in another direction -- we want to load the old branch, but store it
        # with a new branchid.
        if hasattr(task, 'source_branchid'):
            self.load_solution(task.oldparams, task.source_branchid, task.newparams)
        else:
            self.load_solution(task.oldparams, task.branchid, task.newparams)
        self.load_parameters(task.newparams)
        other_solutions = self.io.fetch_solutions(task.newparams, task.ensure_branches)
        bcs = self.problem.boundary_conditions(self.function_space, task.newparams)

        # Try to solve it
        self.log("Deflating other branches %s" % task.ensure_branches)
        self.deflation.deflate(other_solutions + self.trivial_solutions)
        (success, iters) = newton(self.residual, self.state, bcs,
                         self.problem.nonlinear_problem,
                         self.problem.solver,
                         self.problem.solver_parameters(task.newparams, task.__class__),
                         self.teamno, self.deflation, self.dm)

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
            response = self.fetch_response()
            task = ContinuationTask(taskid=task.taskid,
                                    oldparams=task.newparams,
                                    freeindex=task.freeindex,
                                    branchid=task.branchid,
                                    newparams=newparams,
                                    direction=task.direction)
            task.ensure(response.data["ensure_branches"])
            return task

    def stability_task(self, task):
        options = self.problem.solver_parameters(task.oldparams, task.__class__)
        opts = PETSc.Options()
        for k in options:
            opts[k] = options[k]

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

class ProfiledDefconWorker(DefconThread):
    """
    This class handles the actual execution of the tasks necessary
    to do deflated continuation.
    """
    def __init__(self, problem, **kwargs):
        DefconThread.__init__(self, problem, **kwargs)

        # Record gc_frequency from kwargs
        self.gc_frequency = kwargs.get("gc_frequency")

        # A map from the type of task we've received to the code that handles it.
        self.callbacks = {DeflationTask:    self.deflation_task,
                          StabilityTask:    self.stability_task,
                          ContinuationTask: self.continuation_task}

        PETSc.Log.begin()

    def collect(self):
        ev = PETSc.Log.Event("garbage")
        ev.begin()
        DefconThread.collect(self)
        ev.end()

    def run(self, parameters, freeparam):

        runev = PETSc.Log.Event("run")
        runev.begin()
        initev = PETSc.Log.Event("initialisation")
        initev.begin()

        self.parameters = parameters

        # Fetch data from the problem.
        self.mesh = self.problem.mesh(PETSc.Comm(self.teamcomm))
        self.function_space = self.problem.function_space(self.mesh)
        self.dm = create_dm(self.function_space, self.problem)

        # Configure garbage collection frequency:
        if self.gc_frequency is None:
            dofs_per_core = self.function_space.dim() / self.teamcomm.size
            if dofs_per_core > 100000: self.gc_frequency = 1
            if dofs_per_core < 10000:  self.gc_frequency = 100
            else:                      self.gc_frequency = 10

        self.state = backend.Function(self.function_space)
        self.trivial_solutions = None
        self.residual = self.problem.residual(self.state, parameters.constants, backend.TestFunction(self.function_space))

        self.configure_io(parameters)
        self.construct_deflation(parameters)

        # We keep track of what solution we actually have in memory in self.state
        # for efficiency. FIXME: investigate if this actually saves us any
        # time; print out cache hits/misses in self.load_solution
        self.state_id = (None, None)

        initev.end()

        task = None
        while True:
            self.collect()

            if task is None:
                task = self.fetch_task()

            if isinstance(task, QuitTask):
                runev.end()
                self.report_profile()
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
        ev = PETSc.Log.Event("fetching task")
        ev.begin()
        self.log("Fetching task")
        task = self.mastercomm.bcast(None)
        ev.end()
        return task

    def fetch_response(self, block=False):
        ev = PETSc.Log.Event("fetching response")
        ev.begin()
        self.log("Fetching response")
        response = self.mastercomm.bcast(None)

        if block:
            self.mastercomm.barrier()

        ev.end()
        return response

    def send_response(self, response):
        ev = PETSc.Log.Event("sending response")
        ev.begin()
        if self.teamrank == 0:
            self.log("Sending response %s" % response)
            self.worldcomm.send(response, dest=0, tag=self.responsetag)
        ev.end()

    def load_solution(self, oldparams, branchid, newparams):
        ev = PETSc.Log.Event("loading solution")
        ev.begin()

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

        ev.end()
        return

    def load_parameters(self, params):
        self.parameters.update_constants(params)

    def compute_functionals(self, solution):
        ev = PETSc.Log.Event("computing functionals")
        ev.begin()
        funcs = []
        for functional in self.functionals:
            func = functional[0]
            j = func(solution, self.parameters.constants)
            assert isinstance(j, float)
            funcs.append(j)
        ev.end()
        return funcs

    def deflation_task(self, task):
        deflev = PETSc.Log.Event("deflation task")
        deflev.begin()
        # First, load trivial solutions
        if self.trivial_solutions is None:
            self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, task.freeindex)

        # Set up the problem
        ioev = PETSc.Log.Event("deflation: loading")
        ioev.begin()
        self.load_solution(task.oldparams, task.branchid, task.newparams)
        out = self.problem.transform_guess(task.oldparams, task.newparams, self.state); assert out is None

        self.load_parameters(task.newparams)
        other_solutions = self.io.fetch_solutions(task.newparams, task.ensure_branches)
        ioev.end()
        bcev = PETSc.Log.Event("deflation: boundary conditions")
        bcev.begin()
        bcs = self.problem.boundary_conditions(self.function_space, task.newparams)
        bcev.end()

        # Deflate and solve
        solveev = PETSc.Log.Event("deflation: solve")
        solveev.begin()
        self.log("Deflating other branches %s" % task.ensure_branches)
        self.deflation.deflate(other_solutions + self.trivial_solutions)
        (success, iters) = newton(self.residual, self.state, bcs,
                         self.problem.nonlinear_problem,
                         self.problem.solver,
                         self.problem.solver_parameters(task.newparams, task.__class__),
                         self.teamno, self.deflation, self.dm)
        solveev.end()

        self.state_id = (None, None) # not sure if it is a solution we care about yet

        # Get the functionals now, so we can send them to the master.
        funcev = PETSc.Log.Event("deflation: functionals")
        funcev.begin()
        if success: functionals = self.compute_functionals(self.state)
        else: functionals = None
        funcev.end()

        sendev = PETSc.Log.Event("deflation: sending")
        sendev.begin()
        response = Response(task.taskid, success=success, data={"functionals": functionals})
        self.send_response(response)
        sendev.end()

        if not success:
            # that's it, move on
            deflev.end()
            return

        # Get a Response from the master telling us if we should go ahead or not
        fetchev = PETSc.Log.Event("deflation: receiving")
        fetchev.begin()
        response = self.fetch_response()
        fetchev.end()
        if not response.success:
            # the master has instructed us not to bother with this solution.
            # move on.
            deflev.end()
            return
        branchid = response.data["branchid"]

        # We do care about this solution, so record the fact we have it in memory
        self.state_id = (task.newparams, branchid)
        # Save it to disk with the I/O module
        saveev = PETSc.Log.Event("deflation: saving")
        saveev.begin()
        self.log("Found new solution at parameters %s (branchid=%s) with functionals %s" % (task.newparams, branchid, functionals))
        self.problem.monitor(task.newparams, branchid, self.state, functionals)
        self.io.save_solution(self.state, functionals, task.newparams, branchid)
        self.log("Saved solution to %s to disk" % task)
        saveev.end()

        # We want to add one more synchronisation point with master, so that it doesn't go
        # haring off with this solution until it's written to disk (i.e. now)
        fetchev.begin()
        response = self.fetch_response(block=True)
        fetchev.end()

        # Automatically start onto the continuation
        newparams = self.parameters.next(task.newparams, task.freeindex)
        if newparams is not None:
            task = ContinuationTask(taskid=task.taskid,
                                    oldparams=task.newparams,
                                    freeindex=task.freeindex,
                                    branchid=branchid,
                                    newparams=newparams,
                                    direction=+1)
            task.ensure(response.data["ensure_branches"])
            deflev.end()
            return task
        else:
            # Reached the end of the continuation, don't want to continue, move on
            deflev.end()
            return

    def continuation_task(self, task):
        contev = PETSc.Log.Event("continuation task")
        contev.begin()

        # Check for trivial solutions
        if self.trivial_solutions is None:
            self.trivial_solutions = self.problem.trivial_solutions(self.function_space, task.newparams, task.freeindex)

        # Set up the problem

        # We want each (branch, freeindex) pairing to have a unique branch id,
        # to allow for one HDF5 file per branchid (otherwise two teams could
        # write to the same file, causing a race condition). This case occurs
        # when we do multi-parameter continuation, and need to continue a branch
        # in another direction -- we want to load the old branch, but store it
        # with a new branchid.
        ioev = PETSc.Log.Event("continuation: loading")
        ioev.begin()
        if hasattr(task, 'source_branchid'):
            self.load_solution(task.oldparams, task.source_branchid, task.newparams)
        else:
            self.load_solution(task.oldparams, task.branchid, task.newparams)
        self.load_parameters(task.newparams)
        other_solutions = self.io.fetch_solutions(task.newparams, task.ensure_branches)
        ioev.end()
        bcev = PETSc.Log.Event("continuation: boundary conditions")
        bcev.begin()
        bcs = self.problem.boundary_conditions(self.function_space, task.newparams)
        bcev.end()

        # Try to solve it
        solveev = PETSc.Log.Event("continuation: solve")
        solveev.begin()
        self.log("Deflating other branches %s" % task.ensure_branches)
        self.deflation.deflate(other_solutions + self.trivial_solutions)
        (success, iters) = newton(self.residual, self.state, bcs,
                         self.problem.nonlinear_problem,
                         self.problem.solver,
                         self.problem.solver_parameters(task.newparams, task.__class__),
                         self.teamno, self.deflation, self.dm)
        solveev.end()

        if success:
            self.state_id = (task.newparams, task.branchid)

            # Save it to disk with the I/O module
            funcev = PETSc.Log.Event("continuation: functionals")
            funcev.begin()
            functionals = self.compute_functionals(self.state)
            self.problem.monitor(task.newparams, task.branchid, self.state, functionals)
            funcev.end()
            saveev = PETSc.Log.Event("continuation: saving")
            saveev.begin()
            self.io.save_solution(self.state, functionals, task.newparams, task.branchid)
            saveev.end()

        else:
            functionals = None
            self.state_id = (None, None)

        response = Response(task.taskid, success=success, data={"functionals": functionals})
        sendev = PETSc.Log.Event("continuation: sending")
        sendev.begin()
        self.send_response(response)
        sendev.end()

        if not success:
            # Continuation didn't work. Move on.
            # FIXME: we could make this adaptive; try halving the step and doing
            # two steps?
            contev.end()
            return

        if task.direction > 0:
            newparams = self.parameters.next(task.newparams, task.freeindex)
        else:
            newparams = self.parameters.previous(task.newparams, task.freeindex)

        if newparams is None:
            # we have no more continuation to do, move on.
            contev.end()
            return
        else:
            fetchev = PETSc.Log.Event("continuation: receiving")
            fetchev.begin()
            response = self.fetch_response()
            fetchev.end()
            task = ContinuationTask(taskid=task.taskid,
                                    oldparams=task.newparams,
                                    freeindex=task.freeindex,
                                    branchid=task.branchid,
                                    newparams=newparams,
                                    direction=task.direction)
            task.ensure(response.data["ensure_branches"])

            contev.end()
            return task

    def stability_task(self, task):
        stabev = PETSc.Log.Event("stability task")
        stabev.begin()

        options = self.problem.solver_parameters(task.oldparams, task.__class__)
        opts = PETSc.Options()
        for k in options:
            opts[k] = options[k]

        try:
            ioev = PETSc.Log.Event("stability: loading")
            ioev.begin()
            self.load_solution(task.oldparams, task.branchid, -1)
            ioev.end()

            self.load_parameters(task.oldparams)

            solveev = PETSc.Log.Event("stability: solve")
            solveev.begin()
            d = self.problem.compute_stability(task.oldparams, task.branchid, self.state, hint=task.hint)
            solveev.end()

            success = True
            response = Response(task.taskid, success=success, data={"stable": d["stable"]})
        except:
            self.log("Stability task %s failed; exception follows." % task, warning=True)
            traceback.print_exc()
            success = False
            response = Response(task.taskid, success=success)

        if success:
            # Save the data to disk with the I/O module
            saveev = PETSc.Log.Event("stability: saving")
            saveev.begin()
            self.io.save_stability(d["stable"], d.get("eigenvalues", []), d.get("eigenfunctions", []), task.oldparams, task.branchid)
            saveev.end()

        # Send the news to master.
        sendev = PETSc.Log.Event("stability: sending")
        sendev.begin()
        self.send_response(response)
        sendev.end()

        if not success:
            # Couldn't compute stability. Likely something is wrong. Abort and get
            # another task.
            stabev.end()
            return

        # If we're successful, we expect a command from master: should we go ahead, or not?
        recvev = PETSc.Log.Event("stability: receiving")
        recvev.begin()
        response = self.fetch_response()
        recvev.end()

        if not response.success:
            # Master doesn't want us to continue. This is probably because the
            # ContinuationTask that needs to be finished before we can compute
            # its stability is still ongoing. We'll pick it up later.
            stabev.end()
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
            stabev.end()
            return task
        else:
            # No more continuation to do, we're finished
            stabev.end()
            return

    def report_profile(self):
        print "-" * 80
        print "| Profiling statistics collected by team %3d" % self.teamno + " "*35 + "|"
        print "-" * 80

        print " " + "*"*22
        print " * Global statistics *"
        print " " + "*"*22

        total_time = PETSc.Log.Event("run").getPerfInfo()['time']
        init_time  = PETSc.Log.Event("initialisation").getPerfInfo()['time']
        fetch_time = PETSc.Log.Event("fetching task").getPerfInfo()['time']
        gc_time    = PETSc.Log.Event("garbage").getPerfInfo()['time']
        recv_time  = PETSc.Log.Event("fetching response").getPerfInfo()['time']
        send_time  = PETSc.Log.Event("sending response").getPerfInfo()['time']
        load_time  = PETSc.Log.Event("loading solution").getPerfInfo()['time']
        func_time  = PETSc.Log.Event("computing functionals").getPerfInfo()['time']
        defl_time  = PETSc.Log.Event("deflation task").getPerfInfo()['time']
        cont_time  = PETSc.Log.Event("continuation task").getPerfInfo()['time']
        stab_time  = PETSc.Log.Event("stability task").getPerfInfo()['time']
        print "     total time:            %12.4f s" % total_time
        print "     initialisation:        %12.4f s (%05.2f%%)" % (init_time,  100*init_time/total_time)
        print "     garbage collection:    %12.4f s (%05.2f%%)" % (gc_time,    100*gc_time/total_time)
        print "     fetching tasks:        %12.4f s (%05.2f%%)" % (fetch_time, 100*fetch_time/total_time)
        print "     receiving responses:   %12.4f s (%05.2f%%)" % (recv_time, 100*recv_time/total_time)
        print "     sending responses:     %12.4f s (%05.2f%%)" % (send_time, 100*send_time/total_time)
        print "     loading solutions:     %12.4f s (%05.2f%%)" % (load_time, 100*load_time/total_time)
        print "     computing functionals: %12.4f s (%05.2f%%)" % (func_time, 100*func_time/total_time)
        print "     deflation [all]:       %12.4f s (%05.2f%%)" % (defl_time, 100*defl_time/total_time)
        print "     continuation [all]:    %12.4f s (%05.2f%%)" % (cont_time, 100*cont_time/total_time)
        print "     stability [all]:       %12.4f s (%05.2f%%)" % (stab_time, 100*stab_time/total_time)

        if defl_time > 0:
            print

            print " " + "*"*23
            print " * Deflation breakdown *"
            print " " + "*"*23

            load_time  = PETSc.Log.Event("deflation: loading").getPerfInfo()['time']
            bc_time    = PETSc.Log.Event("deflation: boundary conditions").getPerfInfo()['time']
            solve_time = PETSc.Log.Event("deflation: solve").getPerfInfo()['time']
            recv_time  = PETSc.Log.Event("deflation: receiving").getPerfInfo()['time']
            send_time  = PETSc.Log.Event("deflation: sending").getPerfInfo()['time']
            func_time  = PETSc.Log.Event("deflation: functionals").getPerfInfo()['time']
            save_time  = PETSc.Log.Event("deflation: saving").getPerfInfo()['time']
            print "     total time:            %12.4f s" % defl_time
            print "     loading solutions:     %12.4f s (%05.2f%%)" % (load_time, 100*load_time/defl_time)
            print "     constructing BCs:      %12.4f s (%05.2f%%)" % (bc_time, 100*bc_time/defl_time)
            print "     solve:                 %12.4f s (%05.2f%%)" % (solve_time, 100*solve_time/defl_time)
            print "     receiving responses:   %12.4f s (%05.2f%%)" % (recv_time, 100*recv_time/defl_time)
            print "     sending responses:     %12.4f s (%05.2f%%)" % (send_time, 100*send_time/defl_time)
            print "     computing functionals: %12.4f s (%05.2f%%)" % (func_time, 100*func_time/defl_time)
            print "     saving to disk:        %12.4f s (%05.2f%%)" % (save_time, 100*save_time/defl_time)

        if cont_time > 0:
            print

            print " " + "*"*27
            print " * Continuation breakdown *"
            print " " + "*"*27

            load_time  = PETSc.Log.Event("continuation: loading").getPerfInfo()['time']
            bc_time    = PETSc.Log.Event("continuation: boundary conditions").getPerfInfo()['time']
            solve_time = PETSc.Log.Event("continuation: solve").getPerfInfo()['time']
            recv_time  = PETSc.Log.Event("continuation: receiving").getPerfInfo()['time']
            send_time  = PETSc.Log.Event("continuation: sending").getPerfInfo()['time']
            func_time  = PETSc.Log.Event("continuation: functionals").getPerfInfo()['time']
            save_time  = PETSc.Log.Event("continuation: saving").getPerfInfo()['time']
            print "     total time:            %12.4f s" % cont_time
            print "     loading solutions:     %12.4f s (%05.2f%%)" % (load_time, 100*load_time/cont_time)
            print "     constructing BCs:      %12.4f s (%05.2f%%)" % (bc_time, 100*bc_time/cont_time)
            print "     solve:                 %12.4f s (%05.2f%%)" % (solve_time, 100*solve_time/cont_time)
            print "     receiving responses:   %12.4f s (%05.2f%%)" % (recv_time, 100*recv_time/cont_time)
            print "     sending responses:     %12.4f s (%05.2f%%)" % (send_time, 100*send_time/cont_time)
            print "     computing functionals: %12.4f s (%05.2f%%)" % (func_time, 100*func_time/cont_time)
            print "     saving to disk:        %12.4f s (%05.2f%%)" % (save_time, 100*save_time/cont_time)

        if stab_time > 0:
            print

            print " " + "*"*22
            print " * Stability breakdown *"
            print " " + "*"*22

            load_time  = PETSc.Log.Event("stability: loading").getPerfInfo()['time']
            solve_time = PETSc.Log.Event("stability: solve").getPerfInfo()['time']
            recv_time  = PETSc.Log.Event("stability: receiving").getPerfInfo()['time']
            send_time  = PETSc.Log.Event("stability: sending").getPerfInfo()['time']
            save_time  = PETSc.Log.Event("stability: saving").getPerfInfo()['time']
            print "     total time:            %12.4f s" % stab_time
            print "     loading solutions:     %12.4f s (%05.2f%%)" % (load_time, 100*load_time/stab_time)
            print "     solve:                 %12.4f s (%05.2f%%)" % (solve_time, 100*solve_time/stab_time)
            print "     receiving responses:   %12.4f s (%05.2f%%)" % (recv_time, 100*recv_time/stab_time)
            print "     sending responses:     %12.4f s (%05.2f%%)" % (send_time, 100*send_time/stab_time)
            print "     saving to disk:        %12.4f s (%05.2f%%)" % (save_time, 100*save_time/stab_time)

        print
