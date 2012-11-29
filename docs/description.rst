PyPhysim organization
=====================


The PyPhysim library is roughly organized in several packages with related
modules. Most of the packages define functions and classes used to write
physical layer simulators, but two packages are special: the `apps` package
and the `tests` package. The `apps` package contains actual simulators that
can be run, while the `tests` package, as the name suggests, have tests
for the several packages in PyPhysim.

At last, there is also a `bin` directory containing a few scripts. One
useful script to be run while developing PyPhysim is the
**run_python_corerage** script, which will run the python-coverage program
in all the test files in the tests folder. This will give a good estimate
of the test coverage in PyPhysim (see :doc:`writing_unittests` for details
about writting unittests for PyPhysim).


Modules in PyPhysim
-------------------

A summary of the available packages in PyPhysim is shown below.

.. toctree::
   :maxdepth: 1

   cell
   comm
   comp
   ia
   MATLAB
   subspace
   util

