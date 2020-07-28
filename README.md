![Testing](https://travis-ci.org/darcamo/pyphysim.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/darcamo/pyphysim/badge.svg?branch=master)](https://coveralls.io/github/darcamo/pyphysim?branch=master)
[![Documentation Status](https://readthedocs.org/projects/pyphysim/badge/?version=latest)](http://pyphysim.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

PyPhysim
========

Simulation of Digital Communication (physical layer) in Python.

This includes classes related to digital modulation (M-QAM, M-PSK, etc), AWGN
channel, Rayleigh and tapped delay line channel models, channel estimation,
MIMO, OFDM, etc.. It also includes classes related to multiuser transmission,
such as block diagonalization, interference alignment, etc., as well as classes
representing access nodes and users for easily creating physical layer
simulations.

Furthermore, a framework for implementing Monte Carlo simulations is also
implemented (see the pyphysim.simulations package) to help with creating
simulators that run many independent realizations and average the results.


Installation
------------

Pyphysim is available in [Pypi](https://pypi.org/project/pyphysim/) and can be
installed with pip or similar tools. If you want to install from the git
repository, then install [poetry](https://python-poetry.org/) first, clone the
repository, and run the command `poetry install` from the cloned folder to
install pyphysim and its dependencies in a virtual environment (created by
poetry). After that, just use `poetry shell` to activate the environment and you
should be able to import pyphysim from python started in that shell.


Examples
========

There are a few notebooks in the
[notebooks](https://github.com/darcamo/pyphysim/tree/master/notebooks) folder in
the GitHub repository. A list of some of the notebooks with a small description
is shown below for quick access.

- [Transmission_with_AWGN_channel.ipynb](https://github.com/darcamo/pyphysim/blob/master/notebooks/Transmission_with_AWGN_channel.ipynb):
  Illustrates the use of digital modulation to transmit through an AWGN channel
  and the usage of classes that help creating Monte Carlo simulations
- [Transmission_with_Rayleigh_and_AWGN_channels.ipynb](https://github.com/darcamo/pyphysim/blob/master/notebooks/Transmission_with_Rayleigh_and_AWGN_channels.ipynb): Extends the previous notebook to transmission through a Rayleigh channel and discuss running simulation
- [TDL_and_OFDM.ipynb](https://github.com/darcamo/pyphysim/blob/master/notebooks/TDL_and_OFDM.ipynb): Simulation of transmission through a frequency selective channel that varies in time using OFDM
- [PathLoss classes.ipynb](https://github.com/darcamo/pyphysim/blob/master/notebooks/PathLoss_classes.ipynb): Illustrate some path loss classes
- [Alamouti.ipynb](https://github.com/darcamo/pyphysim/blob/master/notebooks/Alamouti.ipynb): Illustrates how to simulate using multiple antennas and how to use configuration files
