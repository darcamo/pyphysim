![Testing](https://travis-ci.org/darcamo/pyphysim.svg?branch=master) 
[![Coverage Status](https://coveralls.io/repos/github/darcamo/pyphysim/badge.svg?branch=master)](https://coveralls.io/github/darcamo/pyphysim?branch=master) 
[![Documentation Status](https://readthedocs.org/projects/pyphysim/badge/?version=latest)](http://pyphysim.readthedocs.io/en/latest/?badge=latest)

PyPhysim
========

Simulation of Digital Communication (physical layer) in Python.

This includes classes related to digital modulation, AWGN channels, MIMO,
OFDM, etc.. It also includes classes related to multiuser transmission such
as block diagonalization, interference alignment, etc.

Furthermore, a framework for implementing Monte Carlo simulations is also
implemented (see the pyphysim.simulations package).


Note
----

It is possible to run the files in the "apps" and "tests" folders without
installing PyPhysim. In that case, you should probably at least run the
setup script to compile any C extension with

`python setup.py build_ext`
