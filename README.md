## Synopsis

The deco library implements the deflated continuation algorithm of
Farrell, Beentjes and Birkisson. Its objective is to compute the
solutions of

f(u, λ) = 0,

where u is the solution of a PDE and λ is a parameter on which the PDE
depends.

The algorithm that deco implements has two main advantages over the
previous state of the art:

* Deco can compute disconnected bifurcation diagrams as well as connected ones.
* Deco scales to massive discretisations of PDEs if a scalable preconditioner is available.

For a full description of the algorithm, see

http://arxiv.org/abs/1603.00809

## Dependencies

Deco depends on

* an MPI library that supports MPI_THREAD_MULTIPLE. In practice this means MPICH and not OpenMPI.
* mpi4py (http://pythonhosted.org/mpi4py/)
* FEniCS, compiled with PETSc and petsc4py support (http://fenicsproject.org)

## Current status

Deco's serial capabilities are reasonably well tested. Its parallel features are
experimental.

## Code Examples

The easiest way to learn how to use it is to examine the examples
in examples/. Start with examples/elastica, and compare to the Euler
elastica section of the paper cited above.

## Installation

    export PYTHONPATH=/path/to/deco:$PYTHONPATH

The contribution of a setup.py would be welcome.

## Contributors

P. E. Farrell <patrick.farrell@maths.ox.ac.uk>

## License

GNU LGPL, version 3.
