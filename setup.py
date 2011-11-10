#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
import os

# sys.path.append('googlemaps')
# import googlemaps


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name="Py-PhySim",
      version="0.1",
      author="Darlan Cavalcante Moreira",
      author_email="darcamo@gmail.com",
      description=("Implementation of a digital communication (physical layer)"
                   " in python."),
      long_description=read("README"),
      requires=["numpy"],
      url="http://code.google.com/p/py-physim/",
      license="GNU General Public License (GPL)",
      scripts=["bin"],
      py_modules=['modulators', 'simulations'],
      packages=['', 'util', 'tests'],
      package_data={'': ["LICENSE.txt"],
                    'tests': ["README"],
                    'util': ["README"]},
      #data_files=['pena_calculator.org'],
      #windows=[{"script": "gui.py"}],
      # options={"py2exe":{"includes":["sip"]}},
      #options={"py2exe": {"skip_archive": True, "includes": ["sip"]}}
      keywords='phy QAM PSK QPSK BPSK Modulation Monte Carlo',
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
)
