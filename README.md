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

Install [poetry](https://python-poetry.org/), clone this repository and then use
the command `poetry install` to install pyphysim.

You can also directly install it from pypi with `pip install pyphysim`.
