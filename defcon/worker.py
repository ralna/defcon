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

