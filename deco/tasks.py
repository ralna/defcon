class Task(object):
    """
    A base class for Tasks.
    """
    pass

class QuitTask(Task):
    """
    A task indicating the slave should quit.
    """
    pass

class ContinuationTask(Task):
    """
    A base task for continuing a known branch.

    *Arguments*
      taskid (int)
        Global identifier for this task
      oldparams (dict)
        Parameter values to continue from
      branchid (int)
        Which branch to continue (int)
      newparams (dict)
        Parameter values to continue to
    """
    def __init__(self, taskid, oldparams, branchid, newparams):
        self.taskid    = taskid
        self.oldparams = oldparams
        self.branchid  = branchid
        self.newparams = newparams

class DeflationTask(Task):
    """
    A task that seeks new, unknown solutions for a given parameter
    value.

    *Arguments*
      taskid (int)
        Global identifier for this task
      oldparams (dict)
        Parameter values to continue from. If None, this means use the initial guesses
      branchid (int)
        Which branch to continue (int). If oldparams is None, this is the number of the
        to use
      newparams (dict)
        Parameter values to continue to
      knownbranches (list)
        Branch ids that have known solutions for newparams
    """
    def __init__(self, taskid, oldparams, branchid, newparams, knownbranches):
        self.taskid    = taskid
        self.oldparams = oldparams
        self.branchid  = branchid
        self.newparams = newparams
        self.knownbranches = knownbranches
