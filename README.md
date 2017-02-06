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
* petsc4py (https://bitbucket.org/petsc/petsc4py)
* FEniCS, compiled with PETSc, petsc4py and HDF5 support (http://fenicsproject.org)

Defcon recommends (and some of the examples depend on)

* h5py, compiled against parallel HDF5 (http://www.h5py.org/)
* matplotlib (http://matplotlib.org)
* mshr (https://bitbucket.org/fenics-project/mshr)
* slepc4py (https://bitbucket.org/slepc/slepc4py)

## Current status

Defcon's serial capabilities are reasonably well tested. Its parallel features are
experimental.

## Code Examples

The easiest way to learn how to use it is to examine the examples
in `examples/`. Start with `examples/elastica`, and compare to the Euler
elastica section of the paper cited above.

## Installation

    pip install .

or

    pip install --user -e .

for *editable* installe into user directory (typically `~/.local`).

## Troubleshooting

* Make sure all `Expressions` and `CompiledSubDomains` take in the `mpi_comm` argument
  (see e.g. `examples/navier-stokes`).

## Contributors

* P. E. Farrell <patrick.farrell@maths.ox.ac.uk>
* J. Pollard <j.pollard@protonmail.com>
* Robert C. Kirby <robert_kirby@baylor.edu>
* Jan Blechta <blechta@karlin.mff.cuni.cz>
* Matteo Croci <croci@maths.ox.ac.uk>

## License

GNU LGPL, version 3.
