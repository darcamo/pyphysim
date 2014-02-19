.. _implementing_monte_carlo_simulations:

Implementing Monte Carlo simulations
------------------------------------

A Monte Carlo simulation involves performing the same simulation many times
with random samples to calculate the statistical properties of some
phenomenon.

The SimulationRunner class makes implementing Monte Carlo simulations
easier by implementing much of the necessary code while leaving the
specifics of each simulator to be implemented in a subclass.

In the simplest case, in order to implement a simulator one would subclass
SimulationRunner, set the simulation parameters in the __init__ method and
implement the :meth:`._run_simulation` method with the code to simulate a
single iteration for that specific simulator. The simulation can then be
performed by calling the :meth:`.simulate` method of an object of that
derived class. After the simulation is finished, the 'results' parameter of
that object will have the simulation results.

The process of implementing a simulator is described in more details in the
following.


Simulation Parameters
~~~~~~~~~~~~~~~~~~~~~

The simulation parameters can be set in any way as long as they can be
accessed in the :meth:`._run_simulation` method. For parameters that won't
be changed, a simple way that works is to store these parameters as
attributes in the __init__ method.

On the other hand, if you want to run multiple simulations, each one with
different values for some of the simulation parameters then store these
parameters in the `self.params` attribute and set them to be unpacked (See
docummentation of the SimulationParameters class for more details). The
:meth:`.simulate` method will automatically get all possible combinations
of parameters and perform a whole Monte Carlo simulation for each of
them. The :meth:`.simulate` method will pass the 'current_parameters' (a
SimulationParameters object) to :meth:`._run_simulation` from where
:meth:`._run_simulation` can get the current combination of parameters.

If you want/need to save the simulation parameters for future reference,
however, then you should store all the simulation parameters in the
`self.params` attribute. This will allow you to call the method
:meth:`.SimulationParameters.save_to_file` to save everything into a
file. The simulation parameters can be recovered latter from this file by
calling the static method :meth:`SimulationParameters.load_from_pickled_file`.


Simulation Results
~~~~~~~~~~~~~~~~~~

In the implementation of the :meth:`._run_simulation` method in a subclass
of SimulationRunner it is necessary to create an object of the
SimulationResults class, add each desided result to it (using the
:meth:`.add_result` method of the :class:`SimulationResults` class) and
then return this object at the end of :meth:`._run_simulation`. Note that
each result added to this :class:`SimulationResults` object must itself be
an object of the :class:`Result` class.

After each run of the :meth:`._run_simulation` method the returned
:class:`SimulationResults` object is merged with the `self.results`
attribute from where the simulation results can be retreived after the
simulation finishes. Note that the way the results from each
:meth:`._run_simulation` run are merged together depend on the
`update_type` attribute of the :class:`Result` object.

Since you will have the complete simulation results in the self.results
object you can easily save them to a file calling its
:class:`SimulationResults.save_to_file` method.

.. note::

   Call the :meth:`SimulationResults.set_parameters` method to set the
   simulation parameters in the self.results object before calling its
   save_to_file method. This way you will have information about which
   simulation parameters were used to generate the results.


Number of iterations the :meth:`._run_simulation` method is performed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of times the :meth:`._run_simulation` method is performed for a
given parameter combination depend on the `self.rep_max` attribute. It is
set by default to '1' and therefore you should set it to the desired value
in the __init__ method of the SimulationRunner subclass.


Optional methods
~~~~~~~~~~~~~~~~

A few methods can be implemented in the SimulationRunner subclass for extra
functionalities. The most useful one is probably the :meth:`._keep_going`
method, which can speed up the simulation by avoid running unecessary
iterations of the :meth:`._run_simulation` method.

Basically, after each iteration of the :meth:`._run_simulation` method the
:meth:`._keep_going` method is called. If it returns True then more
iterations of :meth:`._run_simulation` will be performed until
:meth:`._keep_going` returns False or rep_max iterations are
performed. When the :meth:`._keep_going` method is called it receives a
SimulationResults object with the cumulated results from all iterations so
far, which it can then use to decide it the iterations should continue or
not.

The other optional methods provide hooks to run code at specific points of
the :meth:`.simulate` method. They are described briefly below:

 - :meth:`SimulationRunner._on_simulate_start`:
         This method is called once at the beginning of the simulate
         method.
 - :meth:`SimulationRunner_on_simulate_finish`:
         This method is called once at the end of the simulate method.
 - :meth:`SimulationRunner_on_simulate_current_params_start`:
         This method is called once for each combination of simulation
         parameters before any iteration of _run_simulation is
         performed.
 - :meth:`SimulationRunner_on_simulate_current_params_finish`:
         This method is called once for each combination of simulation
         parameters after all iteration of _run_simulation are
         performed.

At last, for a working example of a simulator, see the
:file:`apps/simulate_psk.py` file.

Example of Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

See the documentation of the :class:`SimulationRunner` class for a pseudo
implementation of a subclass of the :class:`SimulationRunner`.


Running Simulations in Parallel
-------------------------------

If some parameter was marked to be unpacked and instead of calling the
:meth:`.simulate` method you call the :meth:`.simulate_in_parallel` method,
then the simulations for the different parameters will be performed in
parallel using the parallel capabilities of the IPython interpreter.

In order to call :meth:`.simulate_in_parallel` you need to first create a
Client (IPython.parallel.Client) and then get a "view" from it. This view
is a required argument to call :meth:`.simulate_in_parallel`.

The the IPython documentation to understand more.
