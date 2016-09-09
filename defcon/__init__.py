from mpi4py import MPI

try:
    import matplotlib
    matplotlib.use('PDF')
except ImportError:
    pass

import sys
if "dolfin" in sys.modules and "firedrake" not in sys.modules:
    # dolfin imported, no firedrake
    import dolfin
    assert dolfin.has_petsc4py()

    # Check dolfin version
    if dolfin.__version__.startswith("1") or dolfin.__version__.startswith("2016.1.0"):
        raise ImportError("Your version of DOLFIN is too old. DEFCON needs the development version of DOLFIN, 2016.2.0+.")

    dolfin.set_log_level(dolfin.ERROR)
    sys.modules['backend'] = dolfin

elif "firedrake" in sys.modules and "dolfin" not in sys.modules:
    # firedrake imported, no dolfin
    import firedrake
    sys.modules['backend'] = firedrake

elif "firedrake" in sys.modules and "dolfin" in sys.modules:
    # both loaded, don't know what to do
    raise ImportError("Import exactly one of dolfin or firedrake before importing defcon.")

else: # nothing loaded, default to DOLFIN
    import dolfin
    assert dolfin.has_petsc4py()

    # Check dolfin version
    if dolfin.__version__.startswith("1") or dolfin.__version__.startswith("2016.1.0"):
        raise ImportError("Your version of DOLFIN is too old. DEFCON needs the development version of DOLFIN, 2016.2.0+.")

    dolfin.set_log_level(dolfin.ERROR)
    sys.modules['backend'] = dolfin

from numpy                import arange, linspace
from bifurcationproblem   import BifurcationProblem
from defcon               import DeflatedContinuation
from iomodule             import IO, SolutionIO, BranchIO
from operatordeflation    import ShiftedDeflation
