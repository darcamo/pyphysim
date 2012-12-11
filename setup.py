#!/usr/bin/env python
# -*- coding: utf-8 -*-


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# - Run the command "python setup.py nosetests" to run the tests.
# - Run the command "python setup.py build_exe" to create the executables
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

from setuptools import find_packages
from cx_Freeze import setup, Executable


import os

# xxxxx Cython extensions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

bla = Extension("bla", ["lib/bla.pyx"],
                include_dirs=[numpy.get_include()])
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx Get a listof the packages in the project xxxxxxxxxxxxxxxxxxxxx
# The find_packages method returns a list with all Python packages found
# within directory (except the excluded ones). Using find_packages instead
# of writing the name of the packages directly guarantees that we won't
# forget a package which is added in the future.
packages = find_packages(where='.', exclude=[])
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx General Configuration for cx_freeze xxxxxxxxxxxxxxxxxxxxxxxxxx
# cx_freeze allow us to create executables (already containing the required
# libraries) that we can run on other machines.
#
# Extra files to be included in the same directory where the executable
# will be created
includefiles = ['README']
# Modules we don't want to include (note that cx_freeze has a bug and is
# not working with matplotlib right now)
excludes = ['matplotlib',
            'Tkinter',
            'wx',
            'traits',
            'traitsui',
            'kiva',
            'chaco',
            'PIL',
            'PyQt4',
            'enable',
            'apport',
            'email',
            'distutils']
# Path where cx_freeze will look for the modules
path = []  # Note that if you put something here then the default sys.path
           # won't be used. Therefore, uncomment the line below to also
           # include sys.path path.extend(sys.path)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx Target executables for cx_freeze xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# There will be one executable for each app in the project.
simulate_comp = Executable(
    #script="apps/hello.py",
    script="apps/simulate_comp.py",
    initScript="Console",
    #base=None,
    #targetDir="build/exe",
    #targetName="helloexecutable.exe",
    compress=True,
    copyDependentFiles=True,
    appendScriptToExe=False,
    appendScriptToLibrary=False,
    icon=None,
    includes="glib")
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

    #requires=["setuptools", "numpy", "nose"],

    # Scripts for the regular build
    scripts=["bin/run_python_coverage.sh",
             "bin/py-physim"],

    #py_modules=['modulators', 'simulations'],
    packages=packages,
    package_data={'': ["README", "Makefile", "LICENSE.txt"],
                  'tests': ["README"],
                  'util': ["README"]},
    #data_files=['py-physim.org'],

    #windows=[{"script": "gui.py"}],

    setup_requires=['nose>=1.0'],

    # xxxxxxxxxx CX_FREEZE stuff xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    options={
        'build_exe': {
            'excludes': excludes,
            #'packages': ['numpy.core._internal', 'numpy'],
            'packages': packages,
            'include_files': includefiles,
            'path': path,
        }},
    executables=[simulate_comp],
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Cython Stuff xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ext_modules=[bla],
    cmdclass={'build_ext': build_ext}
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
)







# Mensagem para escrever no Fórum

# Como construir um pacote Python para rodar em outra máquina?

# Preciso de uma maneira de rodar um programa que criei em Python (com alguns pacotes e módulos em cada pacote) em outra máquina (que possui apenas uma versão antiga do python instalada e não tem também as bibliotecas que preciso. Aparentemente existem várias opções para isso como o freeze, o pyinstaller, dentre outros.
