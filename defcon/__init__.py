import mpi4py.rc
mpi4py.rc.threaded = False
from mpi4py import MPI

try:
    import matplotlib
    matplotlib.use('PDF')
except ImportError:
    pass

import backends
backends.import_backend()
import backend

from numpy                import arange, linspace
from bifurcationproblem   import BifurcationProblem
from defcon               import DeflatedContinuation
from arclength            import ArclengthContinuation
from iomodule             import IO, SolutionIO
from h5io                 import H5IO
from tasks                import DeflationTask, ContinuationTask, StabilityTask, ArclengthTask
from operatordeflation    import ShiftedDeflation

if backend.__name__ == "dolfin":
    from nonlinearsolver import SNUFLSolver
    from Probe import Probe # borrowed from Mikael Mortensen's excellent fenicstools
    backend.comm_world = backend.mpi_comm_world()

    def vec(x):
        if isinstance(x, backend.Function):
            x = x.vector()
        return backend.as_backend_type(x).vec()

    def mat(x):
        return backend.as_backend_type(x).mat()

elif backend.__name__ == "firedrake":
    backend.comm_world = MPI.COMM_WORLD

from backend import HDF5File

# We have to disable the GC (this is a general thing with running DOLFIN in parallel).
# By default, each Python process decides completely by itself whether to do a
# garbage collection or not. Now suppose some object (e.g. an LU factorisation
# computed by MUMPS) is shared between two processes in a team. It can occur that
# one member of the team decides to clean up, and calls a collective operation,
# deadlocking the entire team. This is not good.
#
# Thus, we disable the GC here and put explicit calls to gc.collect in various places
# in defcon, for both the master and workers.
import gc
gc.disable()
