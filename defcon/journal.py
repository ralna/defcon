from dolfin import *

from multiprocessing import Queue
from threading import Lock
import time
import os


class Journal(object):
    def setup(self, parameters, functionals, freeindex):
        raise NotImplementedError

    def entry(self, oldparams, branchid, newparams, functionals, continuation):
        raise NotImplementedError

    def write(self):
        raise NotImplementedError

# TODO: The queue tries to implement parallel writes with locks. Does this work???
class FileJournal(Journal):
    def __init__(self, directory):
        self.directory = directory + os.path.sep + "journal"

        self.lock = Lock()
    def setup(self, parameters, functionals, freeindex):
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
            f.write("%s;%s;%s;%s\n" % (freeindex, xlabel, ylabels, unicodeylabels))
            f.flush()
            f.close()

    def entry(self, teamid, oldparams, branchid, newparams, functionals, continuation):
        """ Enqueue the journal entry. """
        self.lock.acquire()
        with file(self.directory + os.path.sep + "journal.txt", 'a') as f:
            f.write("%s;%s;%s;%s;%s;%s \n" % (teamid, oldparams, branchid, newparams, functionals, continuation))
            f.flush()
        self.lock.release()

