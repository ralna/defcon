from __future__ import print_function

import sys


def usage():
    print("""
Usage: defcon <command> [options] [args]

Commands:
  gui
  make-pvd
  merge-outputs
  recompute-functionals
  recompute-known-branches
  recompute-stability

Help:
  defcon <command> -h
""")


def main(args=None):
    """This is the commandline tools of the defcon package.
    If you want to call this routine directly (e.g. for puspose of
    testing use args: `(<command>, <arg1>, <arg2>, ...)`, e.g.::

        main(args=("merge-outputs", "/path/to/my/problem.py",
                   "/path/to/output1", "/path/to/output2"))
    """
    if args is None:
        args = sys.argv[1:]

    # NOTE: We only import defcon.gui or defcon.cli if needed in the elif
    #       branches below before exiting; the code probably has side effects

    if len(args) == 0:
        print("No command specified.")
        usage()
        return 1
    elif args[0] == "gui":
        import defcon.gui
        return defcon.gui.main(['defcon gui']+sys.argv[2:])
    elif args[0] == "make-pvd":
        import defcon.cli.makepvd
        return defcon.cli.makepvd.main(['defcon make-pvd']+sys.argv[2:])
    elif args[0] == "merge-outputs":
        import defcon.cli.mergeoutputs
        return defcon.cli.mergeoutputs.main(['defcon merge-outputs']+sys.argv[2:])
    elif args[0] == "recompute-functionals":
        import defcon.cli.recomputefunctionals
        return defcon.cli.recomputefunctionals.main(['defcon recompute-functionals']+sys.argv[2:])
    elif args[0] == "recompute-known-branches":
        import defcon.cli.recomputeknownbranches
        return defcon.cli.recomputeknownbranches.main(['defcon recompute-known-branches']+sys.argv[2:])
    elif args[0] == "recompute-stability":
        import defcon.cli.recomputestability
        return defcon.cli.recomputestability.main(['defcon recompute-stability']+sys.argv[2:])
    else:
        print("Uknown command specified.")
        usage()
        return 1


if __name__ == "__main__":
    sys.exit(main())
