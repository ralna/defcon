import dolfin
dolfin.set_log_level(dolfin.ERROR)

from numpy                import arange, linspace
from bifurcationproblem   import BifurcationProblem
from deflatedcontinuation import DeflatedContinuation
from io                   import IO, FileIO
