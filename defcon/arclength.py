
import gc
import json

import defcon
import newton
import backend

from   mpi4py   import MPI
from   petsc4py import PETSc
from   parametertools import parameterstoconstants
from   tasks import QuitTask, ArclengthTask, Response
from   math import copysign, sqrt
from   heapq import heappush, heappop

class ArclengthContinuation(defcon.DeflatedContinuation):
    """
    This class is the main driver for arclength continuation.
    """

    def __init__(self, problem, teamsize=1, verbose=False, logfiles=False, debug=False):
        """
        Constructor.

        *Arguments*
          problem (:py:class:`defcon.BifurcationProblem`)
            A class representing the bifurcation problem to be solved.
          teamsize (:py:class:`int`)
            How many processors should coordinate to solve any individual PDE.
          verbose (:py:class:`bool`)
            Activate verbose output.
          debug (:py:class:`bool`)
            Activate debugging output.
          logfiles (:py:class:`bool`)
            Whether defcon should remap stdout/stderr to logfiles (useful for many processes).
        """
        self.problem = problem

        self.teamsize = teamsize
        self.verbose  = verbose
        self.logfiles = logfiles
        self.debug    = debug

        self.configure_comms()
        self.configure_logs()

        self.parameters = problem.parameters()
        self.functionals = problem.functionals()
        self.mesh = problem.mesh(PETSc.Comm(self.teamcomm))
        self.function_space = problem.function_space(self.mesh)

        self.configure_io()

    def fetch_data(self):
        problem = self.problem

        self.R = backend.FunctionSpace(self.mesh, "R", 0)

        mixed_element = backend.MixedElement([self.function_space.ufl_element(), self.R.ufl_element()])
        self.mixed_space = backend.FunctionSpace(self.mesh, mixed_element)

        self.consts = list(parameterstoconstants(self.parameters))

        self.state = backend.Function(self.mixed_space)
        self.prev  = backend.Function(self.mixed_space)
        self.test  = backend.TestFunction(self.mixed_space)
        self.ds    = backend.Constant(0)

        (z, lmbda) = backend.split(self.state)
        (z_prev, lmbda_prev) = backend.split(self.prev)
        (w, mu)    = backend.split(self.test)

        # Override the constant with the value of the parameter we're solving for
        self.consts[self.freeindex] = lmbda

        self.state_residual = problem.residual(z, self.consts, w)
        self.residual = (
                         self.state_residual
                       + mu * (backend.inner(z - z_prev, z - z_prev)
                             + backend.inner(lmbda - lmbda_prev, lmbda - lmbda_prev)
                             - self.ds**2)*backend.dx  # arclength criterion
                        )

        # We pass in None here because for arclength we can't have the
        # boundary conditions depend on the parameter values (well, one
        # could, but it would be a lot of work)
        self.bcs = problem.boundary_conditions(self.mixed_space.sub(0), None)
        # Why do they break the interface at every opportunity?
        self.hbcs = problem.boundary_conditions(self.mixed_space.sub(0), None)
        [bc.homogenize() for bc in self.hbcs]

        self.tangent = backend.Function(self.mixed_space)
        self.tangent_prev = None

        self.state_residual_derivative = backend.derivative(self.state_residual, self.state, self.tangent)

    def run(self, params, free, ds, sign, bounds):
        """
        The main execution routine.

        *Arguments*
          params (:py:class:`tuple`)
            A tuple of parameter values to start from. All known solutions for these
            values will be used for the arclength continuation.
          free (:py:class:`str`)
            The name of the parameter that will be varied in the arclength continuation.
          ds (:py:class:`float`)
            The value of the step to take in arclength.
          sign (:py:class:`int`)
            The initial direction of travel for the parameter (must be +1 or -1)
          bounds (:py:class:`tuple`)
            The bounds of interest (param_min, param_max)
        """

        assert len(self.parameters) == len(params)
        assert sign in [+1, -1]
        assert ds > 0
        assert len(bounds) == 2
        assert bounds[0] < bounds[1]

        # Fix the fixed parameters and identify the free parameter.
        freeindex = None
        for (index, param) in enumerate(self.parameters):
            if param[1] == free:
                freeindex = index
                break

        if freeindex is None:
            backend.info_red("Cannot find %s in parameters %s." % (free, [param[1] for param in self.parameters]))
            assert freeindex is not None

        assert bounds[0] <= params[freeindex] <= bounds[1]

        self.freeindex = freeindex

        if self.rank == 0:
            self.master(params, ds, sign, bounds)
        else:
            # join a worker team
            self.worker()

    def master(self, params, ds, sign, bounds):
        # Initialise data structures.
        stat = MPI.Status()

        # First, set the list of idle teams to all of them.
        idleteams = range(self.nteams)

        # Task id counter
        taskid_counter = 0

        # The lists of tasks
        newtasks = []  # tasks yet to be sent out
        waittasks = {} # tasks sent out, waiting to hear back about

        if self.worldcomm.size < 2:
            self.log("Defcon started with only 1 process. At least 2 processes are required (one master, one worker).\n\nLaunch with mpiexec: mpiexec -n <number of processes> python <path to file>", master=True, warning=True)
            import sys; sys.exit(1)

        # Seed the list of tasks.
        for branchid in self.io.known_branches(params):
            task = ArclengthTask(taskid=taskid_counter,
                                 params=params,
                                 branchid=branchid,
                                 bounds=bounds,
                                 sign=sign,
                                 ds=ds)
            heappush(newtasks, (branchid, task))
            taskid_counter += 1

        # Here comes the main master loop.
        while len(newtasks) + len(waittasks) > 0:

            if self.debug:
                self.log("DEBUG: newtasks = %s" % [(priority, str(x)) for (priority, x) in newtasks], master=True)
                self.log("DEBUG: waittasks = %s" % [(key, str(waittasks[key][0]), waittasks[key][1]) for key in waittasks], master=True)
                self.log("DEBUG: idleteams = %s" % idleteams, master=True)

            # Sanity check
            if len(set(idleteams).intersection(set([waittasks[key][1] for key in waittasks]))):
                self.log("ALERT: intersection of idleteams and waittasks: \n%s\n%s" % (idleteams, [(key, str(waittasks[key][0])) for key in waittasks]), master=True, warning=True)
            if set(idleteams).union(set([waittasks[key][1] for key in waittasks])) != set(range(self.nteams)):
                self.log("ALERT: team lost! idleteams and waitasks: \n%s\n%s" % (idleteams, [(key, str(waittasks[key][0])) for key in waittasks]), master=True, warning=True)

            # If there are any tasks to send out, send them.
            while len(newtasks) > 0 and len(idleteams) > 0:
                (priority, task) = heappop(newtasks)
                idleteam = idleteams.pop(0)
                self.send_task(task, idleteam)
                waittasks[task.taskid] = (task, idleteam)

            # We can't send out any more tasks, either because we have no
            # tasks to send out or we have no free processors.
            # If we aren't waiting for anything to finish, we'll exit the loop
            # here. otherwise, we wait for responses and deal with consequences.
            if len(waittasks) > 0:
                self.log("Cannot dispatch any tasks, waiting for response.", master=True)

                # Take this opportunity to call the garbage collector.
                gc.collect()

                response = self.worldcomm.recv(status=stat, source=MPI.ANY_SOURCE, tag=self.responsetag)

                (task, team) = waittasks[response.taskid]
                self.log("Received response %s about task %s from team %s" % (response, task, team), master=True)
                del waittasks[response.taskid]
                idleteams.append(team)

        quit = QuitTask()
        for teamno in range(self.nteams):
            self.send_task(quit, teamno)

    def worker(self):
        self.fetch_data()
        task = self.fetch_task()
        while True:
            # If you add a new task, make sure to add a call to gc.collect()
            if isinstance(task, QuitTask):
                self.log("Quitting gracefully.")
                return
            elif isinstance(task, ArclengthTask):
                self.log("Executing task %s" % task)

                params    = task.params
                branchid  = task.branchid
                bounds    = task.bounds
                ds        = task.ds
                sign      = task.sign

                param = params[self.freeindex]
                self.ds.assign(ds)

                # Configure the parameters
                for (const, value) in zip(self.consts, params):
                    if isinstance(const, backend.Constant):
                        const.assign(value)

                # Load the solution into the previous value
                solution = self.io.fetch_solutions(params, [branchid])[0]

                if backend.__name__  == "dolfin":
                    backend.assign(self.state.sub(0), solution)
                    r = backend.Function(self.R)
                    r.assign(backend.Constant(param))
                    backend.assign(self.state.sub(1), r)
                    del r
                elif backend.__name__ == "firedrake":
                    raise NotImplementedError("Don't know how to assign to subfunctions in firedrake")

                # Data about functionals
                functionals = self.compute_functionals(solution, self.consts)
                data = [(param, functionals)]
                self.log("Initialising arclength at %s = %.15e with functionals %s" % (self.parameters[self.freeindex][1], param, functionals))

                # And begin the main loop
                while bounds[0] <= param <= bounds[1]:
                    gc.collect()

                    # Step 1. Compute the tangent linearisation at self.state
                    (z, lmbda) = backend.split(self.state)
                    (w, mu)    = backend.split(self.test)
                    (z_tlm, lmbda_tlm) = backend.split(self.tangent)

                    # Normalisation condition
                    if self.tangent_prev is not None:
                        # point in the same direction as before
                        normalisation_condition = backend.inner(self.tangent, self.tangent_prev) - backend.Constant(1.0)
                    else:
                        # start going in the direction of sign
                        normalisation_condition = lmbda_tlm - backend.Constant(copysign(1.0, sign))

                    F = self.state_residual_derivative + mu*normalisation_condition*backend.dx
                    success = newton.newton(F, self.tangent, self.hbcs,
                                            self.problem.assembler,
                                            self.problem.solver,
                                            self.teamno)
                    if not success:
                        self.log("Warning: failed to compute tangent", warning=True)
                        break

                    # Step 2. Update the state guess with the tangent
                    self.prev.assign(self.state)
                    nrm = sqrt(backend.assemble(self.problem.squared_norm(z_tlm, z_tlm, self.consts) + backend.inner(lmbda_tlm, lmbda_tlm)*backend.dx))
                    self.state.assign(self.state + (ds/nrm) * self.tangent)

                    # Step 3. Solve the arclength system
                    success = newton.newton(self.residual, self.state, self.bcs,
                                            self.problem.assembler,
                                            self.problem.solver,
                                            self.teamno)
                    if not success:
                        self.log("Warning: failed to solve arclength system", warning=True)
                        break

                    # Step 4. Compute functionals and save information
                    functionals = self.compute_functionals(z, self.consts)
                    if backend.__name__ == "dolfin":
                        param = self.state.split(deepcopy=True)[1].vector().array()[0]
                    else:
                        raise NotImplementedError("Don't know how to do this in firedrake")

                    data.append((param, functionals))
                    self.log("Continued arclength to %s = %.15e with functionals %s" % (self.parameters[self.freeindex][1], param, functionals))

                    # Step 5. Cycle the tangent linear variables
                    if self.tangent_prev is None:
                        self.tangent_prev = backend.Function(self.mixed_space)
                    self.tangent_prev.assign(self.tangent)

                response = Response(task.taskid, success=success)
                if self.teamrank == 0:
                    self.log("Sending response %s to master" % response)
                    self.worldcomm.send(response, dest=0, tag=self.responsetag)
                    self.io.save_arclength(params, self.freeindex, branchid, ds, data)

                task = self.fetch_task()

    def bifurcation_diagram(self, functional, parameter, branchids=None):
        if self.rank != 0:
            return

        if branchids is None:
            branchids = [""] # find all

        import matplotlib.pyplot as plt
        import glob

        # Find the functional index.
        funcindex = None
        for (i, functionaldata) in enumerate(self.functionals):
            if functionaldata[1] == functional:
                funcindex = i
                break
        assert funcindex is not None

        # And find the variable index.
        paramindex = None
        for (i, param) in enumerate(self.parameters):
            if param[1] == parameter:
                paramindex = i
                break

        for branchid in branchids:
            for jsonfile in glob.glob(self.io.directory + "/arclength/*freeindex-%s-branchid-%s*.json" % (paramindex, branchid)):
                data = json.load(open(jsonfile, "r"))
                x = [entry[0] for entry in data]
                y = [entry[1][funcindex] for entry in data]

                plt.plot(x, y, 'o-k', linewidth=2, markersize=1)

        plt.grid()
        plt.xlabel(self.parameters[paramindex][2])
        plt.ylabel(self.functionals[funcindex][2])