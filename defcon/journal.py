from dolfin import *

import time
import os
from ast import literal_eval


class Journal(object):
    """
    Base class for Journal implementations.
    """
    def setup(self, nteams, minparam, maxparam):
        raise NotImplementedError

    def entry(self, oldparams, branchid, newparams, functionals, continuation):
        raise NotImplementedError

    def sweep(self, params):
        raise NotImplementedError

    def team_job(self, team, task):
        raise NotImplementedError

    def exists(self):
        raise NotImplementedError

    def resume(self):
        raise NotImplementedError


class FileJournal(Journal):
    """
    Class that implements journalling using a single file, where the first line tells us some one-off
    details about a problem, and the subsequent lines each contain one piece of information.
    """
    def __init__(self, directory, parameters, functionals, freeindex, sign):
        self.directory = directory + os.path.sep + "journal"
        self.parameters = parameters
        self.functionals = functionals
        self.freeindex = freeindex
        self.sign = sign
        self.sweep_params = None

    def setup(self, nteams, minparam, maxparam):       
        # Create the journal file and directory
        try: os.mkdir(self.directory)
        except OSError: pass

        # Write all the necessary information about the problem, the axis labels and extent, the number of teams, etc.
        xlabel = self.parameters[self.freeindex][2]
        ylabels = [func[2] for func in self.functionals]
        unicodeylabels = [func[1] for func in self.functionals]
        with file(self.directory + os.path.sep + "journal.txt", 'w') as f:
            f.write("%s;%s;%s;%s;%s;%s;%s\n" % (self.freeindex, xlabel, ylabels, unicodeylabels, nteams, minparam, maxparam))
            f.flush()
            f.close()

    def entry(self, teamid, oldparams, branchid, newparams, functionals, continuation):
        with file(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("%s;%s;%s;%s;%s;%s \n" % (teamid, oldparams, branchid, newparams, functionals, continuation))
            f.flush()
            f.close()

    def sweep(self, params):
        if (self.sweep_params is None) or sign*self.sweep_params < sign*params:
            with file(self.directory + os.path.sep + "journal.txt", 'a') as f:
                f.write("$%.20f\n" % params) # Need to make sure we get the decimal places correct here, else there will be bugs with checkpointing.
                f.flush()
                f.close()

    def team_job(self, team, task):
        with file(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("~%s;%s\n" % (team, task))
            f.flush()
            f.close()

    def exists(self):
        return os.path.exists(self.directory)

    def resume(self):
        # Read data from the file.
        pullData = open(self.directory + os.path.sep + "journal.txt", 'r').read().split('\n')

        branches = dict()
        sweep = None

        for eachLine in pullData[1:]:
            if len(eachLine) > 1:
                if eachLine[0] == '$':
                    # This line tells us how far the sweep has gone.
                    sweep = float(eachLine[1:])
                elif eachLine[0] == '~':
                    # This lines tells us something about what task one of the teams was doing, which is unimportant.
                    pass   
                else:
                    # This tells us about a point we discovered. 
                    teamno, oldparams, branchid, newparams, functionals, cont = eachLine.split(';')
                    params = literal_eval(newparams)
                    branches[branchid] = (tuple([float(param) for param in params]), cont)

        # Remove any branches that consist of only one point found by deflation, 
        # which avoids a bug whereby a point might be discovered by deflation 
        # and written to the journal, but not saved by the IO module. 
        for branchid in branches.keys():
            if not branches[branchid][1]: del branches[branchid]

        # Strip the 'cont' vaues out of the branches dictionary. 
        branches = dict([(key, branches[key][0]) for key in branches.keys()])
 
        # Get the extent of the shortest known branch.
        minparams = self.sign*min([self.sign*val[self.freeindex] for val in branches.values()])

        return sweep, minparams, branches
