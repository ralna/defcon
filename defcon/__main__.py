from __future__ import print_function

import sys
import getopt

import defcon.gui


def usage():
    print("""Usage: defcon <command> [options]

defcon gui --help
""")


def main(args=None):
    """This is the commandline tools of the defcon package."""
    if args is None:
        args = sys.argv[1:]

    # Get command-line arguments
    try:
        opts, args = getopt.getopt(args, "h", ["help"])
    except getopt.GetoptError:
        print("Illegal command-line arguments.")
        usage()
        return 1

    # Check for --help
    if ("-h", "") in opts or ("--help", "") in opts:
        usage()
        return 0

    # Check command
    if len(args) >= 1 and args[0] == "gui":
        return defcon.gui.main(['defcon-gui']+sys.argv[2:])
    else:
        print("No command specified.")
        usage()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
