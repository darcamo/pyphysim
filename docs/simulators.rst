Implementing Simulators with PyPhysim
=====================================

Several simulators are already implemented in the "`apps`" package, and can
be used as examples of how to implement simulators with the PyPhysim
library. The best complete example is probably the
":file:`apps/simulate_psk.py`" file.

In general, the :mod:`.simulations` module provides a basic framework
for implementing Monte Carlo Simulations and implementing a new simulator
with PyPhysim starts by subclassing
:class:`.SimulationRunner` and implementing the
:meth:`SimulationRunner._run_simulation` method with the code to simulate a
single iteration for that specific simulator.


A few other classes in the simulations module complete the framework by
handling simulation parameters and simulation results. The classes in the
framwork consist of

- :class:`.SimulationRunner`
- :class:`.SimulationParameters`
- :class:`.SimulationResults`
- :class:`.Result`

For a description of how to implement Monte Carlo simulations using the
clases defined in the :mod:`.simulations` module see
:ref:`implementing_monte_carlo_simulations`.


Getting simulation Parameters from a file
-----------------------------------------

Python has the `ConfigParser library
<http://docs.python.org/2/library/configparser.html#module-ConfigParser>`_
(renamed to configparser in Python 3) to parse configuration
parameters. You can use it in the :meth:`__init__` method of your simulator
class (the simulatro main class that inherits from
:class:`.SimulationRunner`) to read the simulation parameters from a file.

Another good alternative is using the `configobj library <http://www.voidspace.org.uk/python/configobj.html>`_, which provides an
arguable easier to use API then configparser. Another advantage of using
configobj is its ability to validate the configuration file.

