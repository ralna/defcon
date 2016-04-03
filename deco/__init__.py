import dolfin
dolfin.set_log_level(dolfin.ERROR)

from numpy                import arange, linspace
from bifurcationproblem   import BifurcationProblem
from deco                 import DeflatedContinuation
from iomodule             import IO, FileIO
