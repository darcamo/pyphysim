Implementing Simulators with PyPhysim
=====================================

Several simulators are already implemented in the :ref:`apps` package, and
can be used as examples of how to implement simulators with the PyPhysim
library. The best one to start is probably the `apps/simulate_psk.py` file.

In general, implementing a new simulator with PyPhysim starts by
subclassing :class:`util.simulations.SimulationRunner` and implementing the
:meth:`util.simulations.SimulationRunner._run_simulation` method with
the code to simulate a single iteration for that specific simulator.

For more information see the docummentation of the :mod:`util.simulations`
module.
