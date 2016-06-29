"""
Some utility functions that map teams to ranks and ranks to teams.

The default strategy is to group ranks sequentially, i.e. if teamsize = 4
and the ranks are

[0, 1, 2, 3, 4, 5, 6, 7, 8, ...]

then the teams will be

[0, 0, 0, 0, 1, 1, 1, 1, 2, ...]

but this may not be optimal on your machine. If you want to implement a different
layout, just change these two functions.
"""

import math
from mpi4py import MPI

def ranktoteamno(rank, teamsize):
    if rank == 0:
        return MPI.COMM_WORLD.size
    return int(math.floor((rank-1)/float(teamsize)))

    #return int(math.floor((rank)/float(teamsize)))

def teamnotoranks(teamno, teamsize):
    return range(teamno*teamsize, (teamno+1)*teamsize)

if __name__ == "__main__":
    teamsize = 4
    print "rank 0 -> team ", ranktoteamno(0, teamsize)
    print "rank 1 -> team ", ranktoteamno(1, teamsize)
    print "rank 2 -> team ", ranktoteamno(2, teamsize)
    print "rank 3 -> team ", ranktoteamno(3, teamsize)
    print "rank 4 -> team ", ranktoteamno(4, teamsize)
    print "rank 5 -> team ", ranktoteamno(5, teamsize)
    print "team 0 -> ranks ", teamnotoranks(0, teamsize)
    print "team 1 -> ranks ", teamnotoranks(1, teamsize)
