## Synopsis

The defcon library implements the deflated continuation algorithm of
Farrell, Beentjes and Birkisson. Its objective is to compute the
solutions of

f(u, λ) = 0,

where u is the solution of a PDE and λ is a parameter on which the PDE
depends.

The algorithm that defcon implements has two main advantages over the
previous state of the art:

* Defcon can compute disconnected bifurcation diagrams as well as connected ones.
* Defcon scales to massive discretisations of PDEs if a scalable preconditioner is available.

For a full description of the algorithm, see

http://arxiv.org/abs/1603.00809

## Dependencies

The easiest way to run defcon is inside the docker images supplied by the FEniCS project
(http://fenicsproject.org/download); all dependencies are installed there.

If you're compiling things yourself, defcon depends on

* mpi4py (http://pythonhosted.org/mpi4py/)
* FEniCS, compiled with PETSc and petsc4py support (http://fenicsproject.org)

Defcon recommends (and some of the examples depend on)

* matplotlib (http://matplotlib.org)
* mshr (https://bitbucket.org/fenics-project/mshr)

## Current status

Defcon's serial capabilities are reasonably well tested. Its parallel features are
experimental.

## Code Examples

The easiest way to learn how to use it is to examine the examples
in examples/. Start with examples/elastica, and compare to the Euler
elastica section of the paper cited above.

## Installation

    export PYTHONPATH=/path/to/defcon:$PYTHONPATH

The contribution of a setup.py would be welcome.

## Troubleshooting

* Make sure all Expressions and CompiledSubDomains take in the mpi_comm argument (see e.g. examples/navier-stokes).

## Contributors

P. E. Farrell <patrick.farrell@maths.ox.ac.uk>
J. Pollard <j.pollard@protonmail.com>

## License

GNU LGPL, version 3.
