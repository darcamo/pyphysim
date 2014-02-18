#!/usr/bin/env python
# -*- coding: utf-8 -*-


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# - Run the command "python setup.py nosetests" to run the tests.
# - Run the command "python setup.py build_exe" to create the executables
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

from setuptools import find_packages
from setuptools import setup


import os

# xxxxx Cython extensions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

misc_c = Extension(name="misc_c", sources=["util/misc_c.pyx"],
                   include_dirs=[numpy.get_include()])
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx Get a listof the packages in the project xxxxxxxxxxxxxxxxxxxxx
# The find_packages method returns a list with all Python packages found
# within directory (except the excluded ones). Using find_packages instead
# of writing the name of the packages directly guarantees that we won't
# forget a package which is added in the future.
packages = find_packages(where='.', exclude=[])
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Utility function to read the README file. Used for the long_description.
# It's nice, because now 1) we have a top level README file and 2) it's
# easier to type in the README file than to put a raw string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Setup Configuration xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
setup(
    # xxxxxxxxxx Basic Package Information xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    name="PyPhySim",
    version="0.1",

    # Metadata for PyPI
    author="Darlan Cavalcante Moreira",
    author_email="darcamo@gmail.com",
    license="GNU General Public License (GPL)",
    url="http://code.google.com/p/py-physim/",
    download_url='fillmein',
    keywords='phy QAM PSK QPSK BPSK Modulation Monte Carlo',
    description=("Implementation of a digital communication (physical layer) in python."),
    long_description=read("README"),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Telecommunications Industry",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
    ],
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Scripts for the regular build
    scripts=["bin/run_python_coverage.sh",
             "bin/py-physim",
             "count_lines_of_code.sh"],

    #py_modules=['modulators', 'simulations'],
    packages=packages,
    package_data={'': ["README", "LICENSE.txt"],
                  'tests': ["README"]},
    #setup_requires=['nose>=1.0'],

    # xxxxx Cython Stuff xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ext_modules=[misc_c],
    cmdclass={'build_ext': build_ext}
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
)
