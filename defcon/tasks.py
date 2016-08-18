class Task(object):
    """
    A base class for Tasks.
    """
    pass

class QuitTask(Task):
    """
    A task indicating the slave should quit.
    """
    def __str__(self):
        return "QuitTask"

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
    def __init__(self, taskid, oldparams, branchid, newparams, ensure_branches=None):
        self.taskid    = taskid
        self.oldparams = oldparams
        self.branchid  = branchid
        self.newparams = newparams

        if ensure_branches is None: ensure_branches = set()
        self.ensure_branches = ensure_branches

    def ensure(self, branches):
        self.ensure_branches.update(branches)

    def __str__(self):
        return "ContinuationTask(taskid=%s, oldparams=%s, branchid=%s, newparams=%s)" % (self.taskid, self.oldparams, self.branchid, self.newparams)

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
        Which branch to search from (int). If oldparams is None, this is the number of the initial guess
        to use
      newparams (dict)
        Parameter values to continue to
      ensure_branches (set):
        Branches that *must* be deflated; if they are not present yet, wait for them
    """
    def __init__(self, taskid, oldparams, branchid, newparams, ensure_branches=None):
        self.taskid    = taskid
        self.oldparams = oldparams
        self.branchid  = branchid
        self.newparams = newparams

        if ensure_branches is None: ensure_branches = set()
        self.ensure_branches = ensure_branches

    def ensure(self, branches):
        self.ensure_branches.update(branches)

    def __str__(self):
        return "DeflationTask(taskid=%s, oldparams=%s, branchid=%s, newparams=%s)" % (self.taskid, self.oldparams, self.branchid, self.newparams)

class Response(object):
    """
    A class that encapsulates whether a given task was successful or not."
    """
    def __init__(self, taskid, success, functionals=None):
        self.taskid = taskid
        self.success = success
        self.functionals = functionals

    def __str__(self):
        return "Response(taskid=%s, success=%s)" % (self.taskid, self.success)