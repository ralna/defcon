import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy

reducedshell = __import__("reduced-shell")
from reducedshell import c0max, cImax, Nc0, NcI, c0loadings, cIloadings

import backend
problem = reducedshell.ReducedNaghdi()
io = problem.io()

parameters  = problem.parameters()
functionals = problem.functionals()
mesh = problem.mesh(backend.comm_world)
Z = problem.function_space(mesh)

io.setup(parameters, functionals, Z)

num_solutions = numpy.zeros((len(c0loadings), len(cIloadings)))
num_stables   = numpy.zeros((len(c0loadings), len(cIloadings)))

for (i, c0loading) in enumerate(c0loadings):
    for (j, cIloading) in enumerate(cIloadings):
        params = (c0loading, cIloading)
        known_branches      = io.known_branches(params)
        num_solutions[Ny - j - 1, i] = len(known_branches)
        num_stables[Ny - j - 1, i]   = sum(io.fetch_stability(params, known_branches))

plt.imshow(num_solutions, extent=(0, c0max, 0, cImax), interpolation='nearest', cmap=cm.gist_rainbow)
plt.xlabel(r'$c_0$')
plt.ylabel(r'$c_I$')
plt.title('Number of solutions')
plt.colorbar()
plt.savefig('num-solutions.pdf')
plt.clf()

plt.imshow(num_stables, extent=(0, 1, 0, 1), interpolation='nearest', cmap=cm.gist_rainbow)
plt.xlabel(r'$c_0$')
plt.ylabel(r'$c_I$')
plt.title('Number of stable solutions')
plt.colorbar()
plt.savefig('num-stables.pdf')
