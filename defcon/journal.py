from dolfin import *

import time
import os
from ast import literal_eval


class Journal(object):
    def setup(self, parameters, functionals, freeindex):
        raise NotImplementedError

    def entry(self, oldparams, branchid, newparams, functionals, continuation):
        raise NotImplementedError

    def sweep(self, params):
        raise NotImplementedError

    def team_job(self, team, task):
        raise NotImplementedError


class FileJournal(Journal):
    def __init__(self, directory, parameters, functionals, freeindex):
        self.directory = directory + os.path.sep + "journal"
        self.parameters = parameters
        self.functionals = functionals
        self.freeindex = freeindex

    def setup(self, nteams, minparam, maxparam):       
        # Create the journal file and directory
        try: os.mkdir(self.directory)
        except OSError: pass

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
        with file(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("$%.20f\n" % params) # Need to make sure we get the decimal palces correct here, else there will be bugs with checkpointing.
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
        pullData = open(self.directory + os.path.sep + "journal.txt", 'r').read().split('\n')

        branches = dict()
        sweep = None

        for eachLine in pullData[1:]:
            if len(eachLine) > 1:
                if eachLine[0] == '$':
                    sweep = float(eachLine[1:])
                elif eachLine[0] == '~':
                    pass   
                else:
                    teamno, oldparams, branchid, newparams, functionals, cont = eachLine.split(';')
                    params = literal_eval(newparams)
                    branches[branchid] = tuple([float(param) for param in params])

        return sweep, branches


