# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import os
import re
import glob


version = re.findall('__version__ = "(.*)"',
                     open('defcon/__init__.py', 'r').read())[0]

packages = [
    "defcon",
    "defcon.cli",
    "defcon.gui",
    ]

CLASSIFIERS = """
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Programming Language :: Python
Programming Language :: C++
Topic :: Scientific/Engineering :: Mathematics
"""
classifiers = CLASSIFIERS.split('\n')[1:-1]

# TODO: This is cumbersome and prone to omit something
demofiles = glob.glob(os.path.join("examples", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.xml*"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.geo"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.xml*"))

# Don't bother user with test files
[demofiles.remove(f) for f in demofiles if "test_" in f]

setup(name="defcon",
      version=version,
      author="Patrick Farrell",
      author_email="patrick.farrell@maths.ox.ac.uk",
      url="http://bitbucket.com/pefarrell/defcon",
      description="Deflated Continuation",
      long_description="Deflated Continuation algorithm of "
                       "Farrell, Beentjes and Birkisson",
      classifiers=classifiers,
      license="GNU LGPL v3 or later",
      packages=packages,
      package_dir={"defcon": "defcon"},
      package_data={"defcon": ["Probe/*.h", "Probe/*.cpp", "gui/resources/*.png"]},
      data_files=[(os.path.join("share", "defcon", os.path.dirname(f)), [f])
                  for f in demofiles],
      entry_points={'console_scripts': ['defcon = defcon.__main__:main']}
    )
