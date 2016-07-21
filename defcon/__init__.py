from mpi4py import rc
rc.initialize = False

from mpi4py import MPI
if not MPI.Is_initialized():
    ret = MPI.Init_thread(required=MPI.THREAD_MULTIPLE)
    if ret != MPI.THREAD_MULTIPLE and MPI.COMM_WORLD.size > 1:
        print "Error: defcon needs MPI_THREAD_MULTIPLE support. Update your version of MPI."
        assert ret == MPI.THREAD_MULTIPLE
else:
    print "Warning: defcon did not initialize MPI."
    print "Please make sure that whoever did initialize MPI initialized with"
    print "MPI_Init_thread(MPI_THREAD_MULTIPLE)"
    print "and made sure that MPI offered that level of support."

try:
    import matplotlib
    matplotlib.use('PDF')
except ImportError:
    pass

import dolfin
dolfin.set_log_level(dolfin.ERROR)

from numpy                import arange, linspace
from bifurcationproblem   import BifurcationProblem
from defcon               import DeflatedContinuation
from iomodule             import IO, FileIO
