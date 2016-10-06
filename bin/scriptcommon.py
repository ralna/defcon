def fetch_bifurcation_problem(path):
    import os
    import os.path
    import imp

    cwd = os.getcwd()

    probdir = os.path.dirname(path)
    if len(probdir) > 0: os.chdir(probdir)

    # Check if the user's given us a directory or a .py file
    if path.endswith(".py"):
        probpath = path
    elif os.path.isdir(path):
        path = os.path.abspath(path)
        if path.endswith(os.path.sep):
            path = path[:-1]

        lastname = path.split(os.path.sep)[-1]
        probpath = path + os.path.sep + lastname + ".py"
    else:
        raise ValueError("Either specify the .py file or the directory containing it.")

    print probpath
    prob = imp.load_source("prob", probpath)

    globals().update(vars(prob))
    # Run through each class we've imported and figure out which one inherits from BifurcationProblem.
    classes = [key for key in globals().keys()]
    for c in classes:
        try:
            globals()["bfprob"] = getattr(prob, c)
            assert issubclass(bfprob, BifurcationProblem) and bfprob is not BifurcationProblem # check whether the class is a subclass of BifurcationProblem, which would mean it's the class we want. 
            problem = bfprob() # initialise the class.
            break
        except Exception: pass

    return problem
