from dolfin import *

import time
import os


class Journal(object):
    def setup(self, parameters, functionals, freeindex):
        raise NotImplementedError

    def entry(self, oldparams, branchid, newparams, functionals, continuation):
        raise NotImplementedError

    def sweep(self, params):
        raise NotImplementedError

    def team_job(self, team, task):
        raise NotImplementedError


# TODO: The queue tries to implement parallel writes with locks. Does this work???
# No, it doesn't. 
class FileJournal(Journal):
    def __init__(self, directory):
        self.directory = directory + os.path.sep + "journal"

    def setup(self, parameters, functionals, freeindex, nteams, minparam, maxparam):
        self.parameters = parameters
        self.functionals = functionals
        self.freeindex = freeindex
       
        # Create the journal file and directory
        try: os.mkdir(self.directory)
        except OSError: pass

        xlabel = self.parameters[freeindex][2]
        ylabels = [func[2] for func in self.functionals]
        unicodeylabels = [func[1] for func in self.functionals]
        with file(self.directory + os.path.sep + "journal.txt", 'w') as f:
            f.write("%s;%s;%s;%s;%s;%s;%s\n" % (freeindex, xlabel, ylabels, unicodeylabels, nteams, minparam, maxparam))
            f.flush()
            f.close()

    def entry(self, teamid, oldparams, branchid, newparams, functionals, continuation):
        """ Enqueue the journal entry. """
        with file(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("%s;%s;%s;%s;%s;%s \n" % (teamid, oldparams, branchid, newparams, functionals, continuation))
            f.flush()
            f.close()

    def sweep(self, params):
        with file(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("$%s\n" % params)
            f.flush()
            f.close()

    def team_job(self, team, task):
        with file(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("~%s;%s\n" % (team, task))
            f.flush()
            f.close()

