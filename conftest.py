import os
import functools
from petsc4py import PETSc

def pytest_runtest_setup(item):
    """
    - Execute each test in the directory where the test file lives.
    - Clear the PETSc options
    """

    opts = PETSc.Options()
    # should just do
    # opts.clear()
    # but it doesn't work!
    for key in opts.getAll():
        opts.delValue(key)

    starting_directory = os.getcwd()
    test_directory = os.path.dirname(str(item.fspath))
    os.chdir(test_directory)

    teardown = functools.partial(os.chdir, starting_directory)
    # There's probably a cleaner way than accessing a private member.
    item.session._setupstate.addfinalizer(teardown, item)
