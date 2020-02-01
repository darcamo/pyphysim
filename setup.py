#!/usr/bin/env python
# -*- coding: utf-8 -*-

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# - Run the command "python setup.py nosetests" to run the tests.
# - Run the command "python setup.py build_exe" to create the executables
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

from setuptools import dist
# This will make sure that cython and numpy are installed before setup runs
dist.Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os

# xxxxx Cython extensions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# from distutils.extension import Extension
# from Cython.Distutils import build_ext, Extension
# from Cython.Build import cythonize
import numpy as np

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx Get the project version xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def get_version():
    """Get the project version.
    """
    import pyphysim
    return pyphysim.__version__


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Get a listof the packages in the project xxxxxxxxxxxxxxxxxxxxx
# The find_packages method returns a list with all Python packages found
# within directory (except the excluded ones). Using find_packages instead
# of writing the name of the packages directly guarantees that we won't
# forget a package which is added in the future.
# packages = find_packages(where='pyphysim', exclude=[])
packages = find_packages(where='.', exclude=['tests', 'apps'])

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Utility function to read the README file. Used for the long_description.
# It's nice, because now 1) we have a top level README file and 2) it's
# easier to type in the README file than to put a raw string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


doc_requires = ["sphinx", "sphinx_rtd_theme"]
test_requires = ["nose"]
extra_requires = ["pyzmq", "pandas", "ipython", "ipyparallel", "ipywidgets"]

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Setup Configuration xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
setup(
    # xxxxxxxxxx Basic Package Information xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    name="pyphysim",
    version=get_version(),

    # Metadata for PyPI
    author="Darlan Cavalcante Moreira",
    author_email="darcamo@gmail.com",
    license="GNU General Public License (GPL)",
    url="https://github.com/darcamo/pyphysim",
    keywords='phy QAM PSK QPSK BPSK Modulation Monte Carlo',
    description=("Implementation of a digital communication (physical layer) "
                 "in python."),
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Telecommunications Industry",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Scripts for the regular build
    scripts=[
        "bin/run_python_coverage.sh", "bin/py-physim",
        "bin/count_lines_of_code.sh"
    ],
    packages=packages,
    package_data={
        '': ["README.md", "LICENSE.md"],
        "pyphysim.util":
        ["misc_c.pyx"]  # Makes sure that the pyx is included in sdist
    },
    setup_requires=["cython", "numpy"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "cython",
        "configobj",
        "h5py",
        "IPython",
    ],
    # Use 'pip install pyphysim[dev]' or 'pip install -e . "[dev]"'
    extra_require={
        "docs": doc_requires,
        "tests": test_requires,
        "extra": extra_requires,
        "dev": doc_requires + test_requires + extra_requires
    },
    # xxxxx Cython Stuff xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ext_modules=[
        Extension(name="pyphysim.c_extensions.misc_c",
                  sources=["pyphysim/util/misc_c.pyx"],
                  include_dirs=[np.get_include()])
    ],
    cmdclass={'build_ext': build_ext},
    requires=[
        'numpy', 'scipy', 'configobj', 'validate', 'matplotlib', 'h5py',
        'IPython', 'Cython'
    ],
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
)
