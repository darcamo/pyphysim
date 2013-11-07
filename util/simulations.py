#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module containing useful classes to implement Monte Carlo simulations.

The main class for Monte Carlo simulations is the :class:`SimulationRunner`
class, but a few other classes are also implemented to handle simulation
parameters and simulation results.

More specifically, the :mod:`simulations` module implements the classes:
 - :class:`SimulationRunner`
 - :class:`SimulationParameters`
 - :class:`SimulationResults`
 - :class:`Result`

For a description of how to implement Monte Carlo simulations using the
clases defined in the :mod:`simulations` module see the section below.

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
"""

__revision__ = "$Revision$"

import cPickle as pickle
from collections import OrderedDict, Iterable
import itertools
import copy
import numpy as np
from time import time

from util.misc import pretty_time, calc_confidence_interval, replace_dict_values
from util.progressbar import ProgressbarText2, ProgressbarText3, ProgressbarZMQServer2, ProgressbarZMQClient

__all__ = ['SimulationRunner', 'SimulationParameters', 'SimulationResults', 'Result']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Module functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def _parse_range_expr(value, converter=float):
    """
    Parse a string in the form of min:max or min:step:max and return a
    numpy array.
    """
    import validate
    try:
        limits = value.split(':')
        limits = [converter(i) for i in limits]
        if len(limits) == 2:
            value = np.arange(limits[0], limits[1])
        elif len(limits) == 3:
            value = np.arange(limits[0], limits[2], limits[1])
    except Exception:
        raise validate.VdtTypeError(value)

    return value


def _parse_float_range_expr(value):
    """
    Parse a string in the form of min:max or min:step:max and return a
    numpy array (of floats).
    """
    return _parse_range_expr(value, float)


def _parse_int_range_expr(value):
    """
    Parse a string in the form of min:max or min:step:max and return a
    numpy array (of integers).
    """
    return _parse_range_expr(value, int)


def _real_numpy_array_check(value, min=None, max=None):
    """
    Parse and validate `value` as a numpy array (of floats).

    Value can be either a single number, a range expression in the form of
    min:max or min:step:max, or even a list containing numbers and range
    expressions.

    Notes
    -----
    You can either separate the values with commas or spaces (any comma
    will have the same effect as a space). However, if you separate with
    spaces the values should be brackets, while if you separate with
    commands there should be no brackets.
    .. code::
        SNR = 0,5,10:20
        SNR = [0 5 10:20]
    """
    import validate
    if isinstance(value, str):
        # Remove '[' and ']' if they exist.
        if value[0] == '[' and value[-1] == ']':
            value = value[1:-1].strip()
            value = value.replace(',', ' ')  # Replace any commas by a space
            value = value.split()  # Split based on spaces

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Test if it is a list or not
    if isinstance(value, list):
        # If it is a list, each element can be either a number of a 'range
        # expression' that can be parsed with _parse_float_range_expr. We simple
        # apply _real_numpy_array_check on each element in the list to do
        # the work and stack horizontally all the results.
        value = [_real_numpy_array_check(a, min, max) for a in value]
        value = np.hstack(value)

    else:
        # It its not a list, it can be either a single number of a 'range
        # expression' that can be parsed with _parse_float_range_expr
        try:
            value = validate.is_float(value)
            value = np.array([value])
        except validate.VdtTypeError:
            value = _parse_float_range_expr(value)

    # xxxxxxxxxx Validate if minimum and maximum allowed values xxxxxxxxxxx
    if min is not None:
        # maybe "min" was passed as a string and thus we need to convert it
        # to a float
        min = float(min)
        if value.min() < min:
            raise validate.VdtValueTooSmallError(value.min())

    if max is not None:
        # maybe "min" was passed as a string and thus we need to convert it
        # to a float
        max = float(max)
        if value.max() > max:
            raise validate.VdtValueTooBigError(value.max())
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return value


def _integer_numpy_array_check(value, min=None, max=None):
    """
    Parse and validate `value` as a numpy array (of integers).

    Value can be either a single number, a range expression in the form of
    min:max or min:step:max, or even a list containing numbers and range
    expressions.

    Notes
    -----
    You can either separate the values with commas or spaces (any comma
    will have the same effect as a space). However, if you separate with
    spaces the values should be brackets, while if you separate with
    commands there should be no brackets.
    .. code::
        max_iter = 5,10:20
        max_iter = [0 5 10:20]
    """
    import validate
    if isinstance(value, str):
        # Remove '[' and ']' if they exist.
        if value[0] == '[' and value[-1] == ']':
            value = value[1:-1].strip()
            value = value.replace(',', ' ')  # Replace any commas by a space
            value = value.split()  # Split based on spaces

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Test if it is a list or not
    if isinstance(value, list):
        # If it is a list, each element can be either a number of a 'range
        # expression' that can be parsed with _parse_int_range_expr. We simple
        # apply _integer_numpy_array_check on each element in the list to do
        # the work and stack horizontally all the results.
        value = [_integer_numpy_array_check(a, min, max) for a in value]
        value = np.hstack(value)

    else:
        # It its not a list, it can be either a single number of a 'range
        # expression' that can be parsed with _parse_int_range_expr
        try:
            value = validate.is_integer(value)
            value = np.array([value])
        except validate.VdtTypeError:
            value = _parse_int_range_expr(value)

    # xxxxxxxxxx Validate if minimum and maximum allowed values xxxxxxxxxxx
    if min is not None:
        # maybe "min" was passed as a string and thus we need to convert it
        # to a integer
        min = int(min)
        if value.min() < min:
            raise validate.VdtValueTooSmallError(value.min())

    if max is not None:
        # maybe "min" was passed as a string and thus we need to convert it
        # to a integer
        max = int(max)
        if value.max() > max:
            raise validate.VdtValueTooBigError(value.max())
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return value
# xxxxxxxxxx Module Functions - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationRunner - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# pylint: disable=R0921
class SimulationRunner(object):
    """
    Base class to run Monte Carlo simulations.

    The main idea of the :class:`SimulationRunner` class is that in order
    to implement a Monte Carlo simulation one would subclass
    :class:`SimulationRunner` and implement the :meth:`_run_simulation`
    method (as well as any of the optional methods). The complete procedure
    is described in the documentation of the :mod:`simulations` module.

    The code below illustrates the minimum pseudo code to implement a
    subclass of :class:`SimulationRunner`.

    .. code-block:: python

       class SomeSimulator(SimulationRunner):
       def __init__(self):
           SimulationRunner.__init__(self)
           # Do whatever you need/want

           # Add the simulation parameters to the `params` attribute.
           self.params.add('par1', par1value)
           ...
           self.params.add('parN', parNvalue)
           # Optionally set some parameter(s) to be unpacked
           self.params.set_unpack_parameter('name_of_a_parameter')

       def _run_simulation(self, current_parameters):
           # Get the simulation parameters from the current_parameters
           # object. If no parameter was marked to be unpacked, then
           # current_parameters will be equivalent to self.params.
           par1 = current_parameters['par1']
           ...
           parN = current_parameters['parN']

           # Do the simulation of one iteration using the parameters
           # par1,...parN from the current_parameters object.
           ...

           # Save the results of this iteration to a SimulationResults
           # object and return it
           simResults = SimulationResults()
           simResults.add_new_result(...)  # Each result is some metric of interest
           simResults.add_new_result(...)
           return simResults

    With that, all there is left to run the simulation is to create a
    SomeSimulator object and call its :meth:`simulate` method.

    See Also
    --------
    SimulationResults : Class to store simulation results.
    SimulationParameters : Class to store the simulation parameters.
    Result : Class to store a single simulation result.
    """
    def __init__(self):
        self.rep_max = 1
        self._runned_reps = []  # Number of iterations performed by
                                # simulation when it finished
        self.params = SimulationParameters()
        self.results = SimulationResults()

        self._pbar = None  # This variable will be used later to store the
                           # progressbar object when it is created in the
                           # _get_update_progress_function method

        # xxxxxxxxxx update_progress_function_style xxxxxxxxxxxxxxxxxxxxxxx
        # --- When the simulation is performed Serially -------------------
        # Sets the style of the used progressbar. The allowed values are
        # 'text1', 'text2', None, or a callable object.
        # - If it is 'text1' then the ProgressbarText class will be used.
        # - If it is 'text2' then the ProgressbarText2 class will be used.
        # - If it is None, then no progressbar will be used.
        # - If it is a callable, then that calable object must receive two
        #   arguments, the rep_max and the message values, and return a
        #   function that receives a single argument (the custom
        #   parameters).
        # --- When the simulation is performed in parallel ----------------
        # - If it is None then no progressbar will be used
        # - If it is not None then a socket progressbar will be used, which
        #   employes the same style as 'text1'
        self.update_progress_function_style = 'text1'

        # Dictionary with extra arguments that will be passed to the
        # __init__ method of the progressbar class. For instance, when
        # simulating in parallel and update_progress_function_style is not
        # None a progressbar based on ZMQ sockets will be used. Set
        # progressbar_extra_args to "{'port':3456}" in order for the
        # progressbar to use port 3456.
        self.progressbar_extra_args = {}

        # Additional message printed in the progressbar. The message can
        # contain "{SomeParameterName}" which will be replaced with the
        # parameter value.
        #
        # Note that if the update_progress_function_style is None, then no
        # message will be printed either.
        self.progressbar_message = 'Progress'

        # This variable will be used to store the AsyncMapResult object
        # that will be created in the simulate_in_parallel method. This
        # object is part of IPython parallel framework and is used to get
        # the actual results of performing an asynchronous task in IPython.
        self._async_results = None

        # xxxxxxxxxx Configure saving of simulation results xxxxxxxxxxxxxxx
        # If this variable is set to True the saved partial results will be
        # deleted after the simulation is finished.
        self.delete_partial_results_bool = False
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Internal variables you should not modify xxxxxxxxxxxxxxxxxx
        # Variable to store the name of the file where the simulation
        # results will be stored.
        self.__results_base_filename = None
        # Variable to store all the names for the partial results. Each
        # name in it will be equivalent to the value of
        # __results_base_filename appended with unpack_i where i will be an
        # integer. These names will be used after the simulation has
        # finished and full results were saved to delete the files with the
        # partial results.
        self.__results_base_filename_unpack_list = []
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Interval variables for tracking simulation time xxxxxxxxxxx
        # Note that self._elapsed_time is different from the 'elapsed_time'
        # result returned after the simulation has finished. This variable
        # only tracks the CURRENT SIMULATION and does not account the time
        # any loaded partial results required to be simulated.
        self._elapsed_time = 0.0
        self.__tic = 0.0
        self.__toc = 0.0
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def __update__results_base_filename(self, results_filename):
        """
        Update the internal __results_base_filename member variable.

        This variable will stores the name of the file where the simulation
        results will be saved.

        Parameters
        ----------
        results_filename : str
            The name of the file where the simulation results will be
            stored. If not provided the results will not be automatically
            stored. See the notes for more information.
        """
        if results_filename is not None:
            self.__results_base_filename = replace_dict_values(results_filename,
                                                               self.params.parameters)
        else:
            self.__results_base_filename = None

    def _get_results_filename(self):
        """Get name of the file where the last simulation results were stored."""
        if self.__results_base_filename is None:
            results_filename = None
        else:
            results_filename = '{0}.pickle'.format(self.__results_base_filename)
        return results_filename
    results_filename = property(_get_results_filename)

    def _get_unpack_result_filename(self, current_params):
        """
        Get the name of the file where the partial result will be saved for the
        unpackated result with index `unpack_index`.
        """
        total_unpacks = current_params._original_sim_params.get_num_unpacked_variations()
        num_digits = len(str(total_unpacks))
        unpack_index_str = str(current_params._unpack_index).zfill(num_digits)
        partial_results_filename = '{0}_unpack_{1}.pickle'.format(
            self.__results_base_filename,
            unpack_index_str)
        return partial_results_filename

    def __delete_partial_results(self):
        """
        Delete the files containing partial results.

        This method is called inside the simulate method after the full
        results were saved.

        Notes
        -----
        This method will do nothing if self.delete_partial_results_bool is
        not True.
        """
        import os
        if self.delete_partial_results_bool is True:
            for name in self.__results_base_filename_unpack_list:
                try:
                    os.remove(name)
                except Exception:
                    pass
            self.__results_base_filename_unpack_list = []
        else:
            # Do nothing if self.delete_partial_results_bool is not True
            pass

    def __save_partial_results(self, current_rep, current_params, current_sim_results, partial_results_filename):
        """
        Save the partial simulation results to a file.

        Parameters
        ----------
        current_rep : int
            Current repetition.
        current_params : SimulationParameters
            The current parameters.
        current_sim_results : SimulationResults
            The partial simulations results object to be saved.
        partial_results_filename : str
            The name of the file to save the partial simulation results.

        Notes
        -----
        The name of the results file where the partial results were saved,
        that is, the value of the `partial_results_filename` argument will
        be automatically added to the list of files to be deleted after the
        simulation is finished if delete_partial_results_bool is
        True. However, remember that in the simulate_in_parallel method
        this method will be run in a different object in an IPython
        engine. Therefore, you will need to manually add, the value of the
        partial_results_filename variable to the list of files to be
        deleted (the __delete_partial_results variable).
        """
        # xxxxxxxxxx Save partial results to file xxxxxxxxxxxxxxxxxxxxx
        # First we add the current parameters to the partial simulation
        # results object
        current_sim_results.set_parameters(current_params)
        # Now we can save the partial results to a file.

        current_sim_results.current_rep = current_rep
        current_sim_results.save_to_file(partial_results_filename)

        self.__results_base_filename_unpack_list.append(
            partial_results_filename)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def clear(self, ):  # pragma: no cover
        """Clear the SimulationRunner.

        This will erase any results from previous simulations as well as
        other internal variables. The SimulationRunner object will then be
        as if it was just created, except that the simulation parameters
        will be kept.

        """
        self._elapsed_time = 0.0
        self._runned_reps = []  # Number of iterations performed by
                                # simulation when it finished
        self.results = SimulationResults()

    def __run_simulation_and_track_elapsed_time(self, current_parameters):
        """
        Perform the _run_simulation method and track its execution time. This
        time will be added as a Result to the returned SimulationResults
        object from _run_simulation.

        Parameters
        ----------
        current_parameters : SimulationParameters object
            SimulationParameters object with the parameters for the
            simulation. The self.params variable is not used directly. It
            is first unpacked in the simulate function which then calls
            _run_simulation for each combination of unpacked parameters.

        Notes
        -----
        This method is called in the `simulate` and `simulate_in_parallel`.
        """
        tic = time()
        current_sim_results = self._run_simulation(current_parameters)
        toc = time()
        elapsed_time_result = Result.create('elapsed_time',
                                                    Result.SUMTYPE,
                                                    toc - tic)
        current_sim_results.add_result(elapsed_time_result)

        return current_sim_results

    def _run_simulation(self, current_parameters):
        """Performs one iteration of the simulation.

        This function must be implemented in a subclass. It should take the
        needed parameters from the params class attribute (which was filled
        in the constructor of the derived class) and return the results as
        a SimulationResults object.

        Note that _run_simulation will be called self.rep_max times (or
        less if an early stop criteria is reached, which requires
        reimplementing the _keep_going function in the derived class) and
        the results from multiple repetitions will be merged.

        Parameters
        ----------
        current_parameters : SimulationParameters object
            SimulationParameters object with the parameters for the
            simulation. The self.params variable is not used directly. It
            is first unpacked in the simulate function which then calls
            _run_simulation for each combination of unpacked parameters.

        Returns
        -------
        simulation_results : SimulationResults object
            A SimulationResults object containing the simulation results of
            the run iteration.

        """
        raise NotImplementedError("'_run_simulation' must be implemented in a subclass of SimulationRunner")

    # pylint: disable=W0613,R0201
    def _keep_going(self, current_params, current_sim_results, current_rep):
        """
        Check if the simulation should continue or stop.

        This function may be reimplemented in the derived class if a stop
        condition besides the number of iterations is desired. The idea is
        that _run_simulation returns a SimulationResults object, which is
        then passed to _keep_going, which is then in charge of deciding if
        the simulation should stop or not.

        Parameters
        ----------
        current_params : SimulationParameters object
            SimulationParameters object with the parameters of the
            simulation.
        current_sim_results : SimulationResults object
            SimulationResults object from the last iteration (merged with
            all the previous results)
        current_rep : int
            Number of iterations already run.

        Returns
        -------
        result : bool
            True if the simulation should continue or False otherwise.
        """
        # If this function is not reimplemented in a subclass it always
        # returns True. Therefore, the simulation will only stop when the
        # maximum number of allowed iterations is reached.
        return True

    def _get_serial_update_progress_function(self, current_params):
        """
        Return a function that should be called to update the
        progressbar for the simulation of the current parameters.

        This method is only called in the 'simulate' method.

        The returned function accepts a single argument, corresponding to
        the number of iterations executed so far.

        The progressbar used to get the returned function depend on the
        value of the self.update_progress_function_style attribute.

        Parameters
        ----------
        current_params : SimulationParameters object
            The current combination of simulation parameters. This should
            be used to perform any replacement in the
            self.progressbar_message string that will be written in the
            progressbar.

        Returns
        -------
        func : function that accepts a single integer argument
            Function that accepts a single integer argument and can be
            called to update the progressbar.

        Notes
        -----
        The equivalent of this method which is used in the
        simulate_in_parallel method if the
        _get_parallel_update_progress_function method.
        """
        # If the progressbar_message has any string replacements in the
        # form {some_param} where 'some_param' is a parameter in
        # current_params then it will be replaced by the current value of
        # 'some_param'.
        message = self.progressbar_message.format(**current_params.parameters)
        # By default, the returned function is a dummy function that does
        # nothing
        update_progress_func = lambda value: None

        # If the self.update_progress_function_style attribute matches one of the
        # available styles, then update_progress_func will be appropriately
        # set.
        if self.update_progress_function_style == 'text1':  # pragma: no cover
            # We will use the ProgressbarText class
            self._pbar = ProgressbarText2(self.rep_max, '*', message)
            update_progress_func = self._pbar.progress
        elif self.update_progress_function_style == 'text2':  # pragma: no cover
            # We will use the ProgressbarText2 class
            self._pbar = ProgressbarText2(self.rep_max, '*', message)
            update_progress_func = self._pbar.progress
        elif callable(self.update_progress_function_style) is True:
            # We will use a custom function to update the progress. Note
            # that we call self.update_progress_function_style to return
            # the actual function that will be used to update the
            # progress. That is, the function stored in
            # self.update_progress_function_style should basically do what
            # _get_update_progress_function is supposed to do.
            update_progress_func = self.update_progress_function_style(
                self.rep_max, self.progressbar_message)  # pragma: no cover

        return update_progress_func
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _get_parallel_update_progress_function(self):
        """
        Return a function that should be called to update the
        progressbar for the simulation of the current parameters.

        This method is only called in the 'simulate_in_parallel' method.

        The returned function accepts a single argument, corresponding to
        the number of iterations executed so far.

        # The progressbar used to get the returned function depend on the
        # value of the self.update_progress_function_style attribute.

        Parameters
        ----------
        full_params : SimulationParameters object
            The simulation parameters. This should be used to perform any
            replacement in the self.progressbar_message string that will be
            written in the progressbar.

        Returns
        -------
        func : function that accepts a single integer argument
            Function that accepts a single integer argument and can be
            called to update the progressbar.

        Notes
        -----
        While the _get_serial_update_progress_function method receives the
        "current parameters" this _get_parallel_update_progress_function
        receives the full parameters. The reason for this is that when the
        simulation is performed in parallel multiple process (for different
        parameters) will update the same progressbar. Therefore, it does
        not make sense to perform replacements in the progressbar message
        based on current parameters.
        """
        if self.update_progress_function_style is not None:
            if self._pbar is None:
                parameters = self.params.parameters
                # If the progressbar_message has any string replacements in the
                # form {some_param} where 'some_param' is a parameter in
                # 'full_params' then it will be replaced by the value of
                # 'some_param'.
                message = self.progressbar_message.format(**parameters)
                self._pbar = ProgressbarZMQServer2(message=message,
                                                 **self.progressbar_extra_args)

            # Note that this will be an object of the ProgressbarZMQClient
            # class, but it behaves like a function.
            proxybar = self._pbar.register_client_and_get_proxy_progressbar(self.rep_max)
            proxybar_data = [proxybar.client_id, proxybar.ip, proxybar.port]

        return proxybar_data

    @property
    def elapsed_time(self):
        """property: Get the simulation elapsed time. Do not set this
        value."""
        return pretty_time(self._elapsed_time)

    @property
    def runned_reps(self):
        """Get method for the runned_reps property."""
        return self._runned_reps

    # This method is called when the SimulationRunner class is pickled
    def __getstate__(self):
        # We will pickle everything as default, escept for the "_pbar"
        # member variable that will not be pickled. The reason is that it
        # may be a ProgressbarZMQServer object, which cannot be pickled (uses
        # ZMQ sockets).
        state = dict(self.__dict__)
        del state['_pbar']
        return state

    def get_runned_reps_fix_params(self, fixed_params_dict=dict()):
        """
        Get the number of runned repetitions for a given set of parameters.

        You can get a list of the number of repetitions combination of
        transmit parameters with the "runned_reps" property. However, if
        you have more then one transmit parameter set to be unpacked it
        might be difficult knowing which element in the list corresponds to
        the simulation for a given set of transmit parameters. By using the
        get_runned_reps_fix_params method you will be the number
        repetitions for the desired set of transmit parameters.
        """
        indexes = self.params.get_pack_indexes(fixed_params_dict)
        runned_reps_subset = np.array(self.runned_reps)[indexes]
        return runned_reps_subset

    def simulate(self, results_filename=None):
        """
        Performs the full Monte Carlo simulation (serially).

        Implements the general code for every simulation. Any code
        specific to a single simulation iteration must be implemented in the
        _run_simulation method of a subclass of SimulationRunner.

        The main idea behind the SimulationRunner class is that the general
        code in every simulator is implemented in the SimulationRunner
        class, more specifically in the `simulate` method, while the
        specific code of a single iteration is implemented in the
        _run_simulation method in a subclass.

        Parameters
        ----------
        results_filename : str
            The name of the file where the simulation results will be
            stored. If not provided the results will not be automatically
            stored. See the notes for more information.

        Notes
        -----
        The `results_filename` argument is formatted with the simulation
        parameters. That is, supose there are two parameters Nr=2 and Nt=1,
        then if `results_filename` is equal to "results_for_{Nr}x{Nt}" the
        actual name of the fiel used to stote the simulation parameters
        will be "results_for_2x1.pickle".

        See Also
        --------
        simulate_in_parallel
        """
        # xxxxxxxxxx Update the __results_base_filename variable xxxxxxxxxx
        # The __results_base_filename variable will contain the name of the
        # file where the simulation results should be saved. It will be
        # equivalent to the provided results_filename argument after any
        # parameter replacements.
        self.__update__results_base_filename(results_filename)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Iterator to print the current variation xxxxxxxxxxxxxx
        # This local function returns an iterator that prints the current
        # variation each time its "next" method is called.
        def _print_variation_iter(num_variations):
            if self.update_progress_function_style is None:
                for i in itertools.repeat(''):
                    yield 0
            else:  # pragma: no cover
                variation_pbar = ProgressbarText3(
                    num_variations,
                    progresschar='-',
                    message="Current Variation:")

                for i in range(1, num_variations + 1):
                    variation_pbar.progress(i)
                    print  # print a new line
                    yield i

        # Create the var_print_iter Iterator
        # Each time the 'next' method of var_print_iter is called it will
        # print something like
        # ------------- Current Variation: 4/84 ------------
        # which means the variation 4 of 84 variations.
        var_print_iter = _print_variation_iter(
            self.params.get_num_unpacked_variations())
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.__tic = time()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Store rep_max in the results object xxxxxxxxxxxxxxxxxxxxxxx
        self.results.rep_max = self.rep_max
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Store the Simulation parameters in the SimulationResults object.
        # With this, the simulation parameters will be available for
        # someone that has the SimulationResults object (loaded from a
        # file, for instance).
        self.results.set_parameters(self.params)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Implement the _on_simulate_start method in a subclass if you need
        # to run code at the start of the simulate method.
        self._on_simulate_start()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx FOR UNPACKED PARAMETERS xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Loop through all the parameters combinations
        for current_params in self.params.get_unpacked_params_list():
            next(var_print_iter)
            # Implement the _on_simulate_current_params_start method in a
            # subclass if you need to run code before the _run_simulation
            # iterations for each combination of simulation parameters.
            self._on_simulate_current_params_start(current_params)

            # First we try to Load the partial results for the current
            # parameters.
            try:
                # Name of the file where the partial results will be saved
                if self.__results_base_filename is not None:
                    partial_results_filename = self._get_unpack_result_filename(
                        current_params)
                else:
                    # If __results_base_filename is None there is also no
                    # partial results to load. Therefore, lets raise an
                    # IOError here to go to the except catching
                    raise IOError()

                # If loading partial results succeeds, then we will have
                # partial results. If it fails because the file does not
                # exist, this will thrown a IOError exception and we will
                # execute the except block instead.
                current_sim_results = SimulationResults.load_from_file(
                    partial_results_filename)

                # Note that at this point we have successfully loaded the
                # partial results from the file. However, we still need to
                # make sure it matches our current parameters. It it does
                # not match the user likely used a wrong name and we raise
                # an exception to stop the simulation.
                #
                # NOTE: If the type of this exception is changed in the
                # future make sure it is not caught in the except block.
                if not current_params == current_sim_results.params:
                    raise ValueError("Partial results loaded from file does not match current parameters. \nfile: {0}".format(partial_results_filename))

                # The current_rep will be set to the value or run
                # repetitions in the loaded partial results. This means
                # that the "while" statement after this try/except block
                # will have a head start and if current_rep is greater than
                # or equal to rep_max the while loop won't run at all.
                current_rep = current_sim_results.current_rep

            # If loading partial results failed then we will run the FIRST
            # repetition here and the "while" statement after this
            # try/except block will run as usual.
            except IOError:
                # Perform the first iteration of _run_simulation
                current_sim_results = self.__run_simulation_and_track_elapsed_time(
                    current_params)
                current_rep = 1

            update_progress_func = self._get_serial_update_progress_function(current_params)
            # Run more iterations until one of the stop criteria is reached
            while (self._keep_going(current_params, current_sim_results, current_rep)
                   and
                   current_rep < self.rep_max):
                new_sim_results = self.__run_simulation_and_track_elapsed_time(
                    current_params)
                current_sim_results.merge_all_results(new_sim_results)

                current_rep += 1
                update_progress_func(current_rep)

                # Save partial results each 500 iterations
                if current_rep % 500 == 0 and self.__results_base_filename is not None:
                    self.__save_partial_results(current_rep, current_params, current_sim_results, partial_results_filename)

            # If the while loop ended before rep_max repetitions (because
            # _keep_going returned false) then set the progressbar to full.
            update_progress_func(self.rep_max)

            # Implement the _on_simulate_current_params_finish method in a
            # subclass if you need to run code after all _run_simulation
            # iterations for each combination of simulation parameters
            # finishes.
            self._on_simulate_current_params_finish(current_params,
                                                    current_sim_results)

            # Store the number of repetitions actually ran for the current
            # parameters combination
            self._runned_reps.append(current_rep)
            # Lets append the simulation results for the current parameters
            self.results.append_all_results(current_sim_results)

            # xxxxxxxxxx Save partial results to file xxxxxxxxxxxxxxxxxxxxx
            if self.__results_base_filename is not None:
                self.__save_partial_results(current_rep, current_params, current_sim_results, partial_results_filename)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # This will add a blank line between the simulations for
            # different unpacked variations (when there is more then one)
            if self.params.get_num_unpacked_variations() > 1 and self.update_progress_function_style is not None:
                print("")  # pragma: no cover
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Implement the _on_simulate_finish method in a subclass if you
        # need to run code at the end of the simulate method.
        self._on_simulate_finish()

        # xxxxxxx Save the number of runned iterations xxxxxxxxxxxxxxxxxxxx
        self.results.runned_reps = self._runned_reps
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Update the elapsed time xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.__toc = time()
        self._elapsed_time = self.__toc - self.__tic

        # Also save the elapsed time in the SimulationResults object
        self.results.elapsed_time = self._elapsed_time
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Save the results if results_filename is not None xxxxxxxxxx
        if self.__results_base_filename is not None:
            self.results.save_to_file(self.results_filename)
            # Delete the partial results (this will only delete the partial
            # results if self.delete_partial_results_bool is True)
            self.__delete_partial_results()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def simulate_in_parallel(self, view, wait=True, results_filename=None):
        """
        Same as the simulate method, but the different parameters
        configurations are simulated in parallel.

        Parameters
        ----------
        view : A view of the IPython engines.
            A DirectView of the available IPython engines. The parallel
            processing will happen by calling the 'map' method of the
            provided view to simulate in parallel the different
            configurations of transmission parameters.
        wait : Bool
            If True then the self.wait_parallel_simulation method will be
            automatically called at the end of simulate_in_parallel. If
            False, the YOU NEED to manually call
            self.wait_parallel_simulation at some point after calling
            simulate_in_parallel.

        Notes
        -----
        There is a limitation regarding the partial simulation results. The
        partial results files will be saved in the folder where the IPython
        engines are running, since the "saving part" is performed in an
        IPython engine. However, the deletion of the files is not performed
        in by the IPython engines, but by the main python
        program. Therefore, unless the Ipython engines are running in the
        same folder where the main python program will be run the partial
        result files won't be automatically deleted after the simulation is
        finished.
        """
        # # xxxxx Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # # Get the client which created the view `view` and then get a
        # # direct_view with the same targets
        # cl = view.client
        # dview = cl.direct_view(view.targets)
        # # Now we can use this direct view to perform some imports
        # import os
        # import sys
        # parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
        # sys.path.append(parent_dir)
        # dview.execute('import sys')
        # dview.execute('sys.path.append("{0}")'.format(parent_dir))
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Update the __results_base_filename variable xxxxxxxxxx
        # The __results_base_filename variable will contain the name of the
        # file where the simulation results should be saved. It will be
        # equivalent to the provided results_filename argument after any
        # parameter replacements.
        self.__update__results_base_filename(results_filename)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        #pbar = ProgressbarMultiProcessServer(sleep_time=5)

        # xxxxxxxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.__tic = time()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Store rep_max in the results object xxxxxxxxxxxxxxxxxxxxxxx
        self.results.rep_max = self.rep_max
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Store the Simulation parameters in the SimulationResults object.
        # With this, the simulation parameters will be available for
        # someone that has the SimulationResults object (loaded from a
        # file, for instance).
        self.results.set_parameters(self.params)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Implement the _on_simulate_start method in a subclass if you need
        # to run code at the start of the simulate method.
        self._on_simulate_start()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx FOR UNPACKED PARAMETERS xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # ----- Function that will be called in each IPython engine -------
        def simulate_for_current_params(obj, current_params, proxybar_data=None):
            """
            Parameters
            ----------
            current_params : SimulationParameters object.
                The current parameters
            proxybar_data : list of 3 elements or None
                The elements are the "client_id" (and int), the "ip" (a
                string with an IP address) and the "port". This data should
                be used to create a ProgressbarZMQClient object that can be
                used to update the progressbar (via a ZMQ socket)
            """
            # xxxxxxxxxx Function to update the progress xxxxxxxxxxxxxxxxxx
            if proxybar_data is None:
                update_progress_func = lambda value: None
            else:
                client_id, ip, port = proxybar_data
                proxybar = ProgressbarZMQClient(client_id, ip, port)
                update_progress_func = proxybar.progress
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # Implement the _on_simulate_current_params_start method in a
            # subclass if you need to run code before the _run_simulation
            # iterations for each combination of simulation parameters.
            obj._on_simulate_current_params_start(current_params)

            # First we try to Load the partial results for the current
            # parameters.
            try:
                # Name of the file where the partial results will be saved
                if obj.__results_base_filename is not None:
                    partial_results_filename = obj._get_unpack_result_filename(
                        current_params)
                else:
                    # If __results_base_filename is None there is also no
                    # partial results to load. Therefore, lets raise an
                    # IOError here to go to the except catching
                    raise IOError()

                # If loading partial results succeeds, then we will have
                # partial results. If it fails because the file does not
                # exist, this will thrown a IOError exception and we will
                # execute the except block instead.
                current_sim_results = SimulationResults.load_from_file(
                    partial_results_filename)

                # Note that at this point we have successfully loaded the
                # partial results from the file. However, we still need to
                # make sure it matches our current parameters. It it does
                # not match the user likely used a wrong name and we raise
                # an exception to stop the simulation.
                #
                # NOTE: If the type of this exception is changed in the
                # future make sure it is not caught in the except block.
                if not current_params == current_sim_results.params:
                    raise ValueError("Partial results loaded from file does not match current parameters. \nfile: {0}".format(partial_results_filename))

                # The current_rep will be set to the value or run
                # repetitions in the loaded partial results. This means
                # that the "while" statement after this try/except block
                # will have a head start and if current_rep is greater than
                # or equal to rep_max the while loop won't run at all.
                current_rep = current_sim_results.current_rep

            # If loading partial results failed then we will run the FIRST
            # repetition here and the "while" statement after this
            # try/except block will run as usual.
            except IOError:
                # Perform the first iteration of _run_simulation
                current_sim_results = obj.__run_simulation_and_track_elapsed_time(
                    current_params)
                current_rep = 1

            # Run more iterations until one of the stop criteria is reached
            while (obj._keep_going(current_params, current_sim_results, current_rep)
                   and
                   current_rep < obj.rep_max):
                current_sim_results.merge_all_results(
                    obj.__run_simulation_and_track_elapsed_time(current_params))
                current_rep += 1
                update_progress_func(current_rep)

                # Save partial results each 500 iterations
                if current_rep % 500 == 0 and obj.__results_base_filename is not None:
                    obj.__save_partial_results(current_rep, current_params, current_sim_results, partial_results_filename)

            # If the while loop ended before rep_max repetitions (because
            # _keep_going returned false) then set the progressbar to full.
            update_progress_func(obj.rep_max)

            # Implement the _on_simulate_current_params_finish method in a
            # subclass if you need to run code after all _run_simulation
            # iterations for each combination of simulation parameters
            # finishes.
            obj._on_simulate_current_params_finish(current_params,
                                                   current_sim_results)

            # xxxxxxxxxx Save partial results to file xxxxxxxxxxxxxxxxxxxxx
            if obj.__results_base_filename is not None:
                obj.__save_partial_results(current_rep, current_params, current_sim_results, partial_results_filename)
            else:
                partial_results_filename = None
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # This function returns a tuple containing the number of
            # iterations run as well as the SimulationResults object.
            return (current_rep, current_sim_results, partial_results_filename)
        # -----------------------------------------------------------------

        # Loop through all the parameters combinations
        num_variations = self.params.get_num_unpacked_variations()

        # xxxxxxxxxx Progressbar for the parallel simulation xxxxxxxxxxxxxx
        if self.update_progress_function_style is not None:
            # Create the proxy progressbars
            proxybar_data_list = []
            for i in range(num_variations):
                proxybar_data_list.append(self._get_parallel_update_progress_function())
        else:  # self.update_progress_function_style is None
            # Create the dummy update progress functions
            proxybar_data_list = [None] * num_variations
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxx Perform the actual simulation in asynchronously parallel xxxx
        # NOTE: If this fails because of some pickling error, make sure the
        # class of 'self' (that is, the subclass of SimulationRunner that
        # you are trying to run) is pickle-able.
        self._async_results = view.map(
            simulate_for_current_params,
            # We need to pass the SimulationRunner
            # object to the IPython engine ...
            [self] * num_variations,
            # ... and we also need to pass the
            # simulation parameters for each engine
            self.params.get_unpacked_params_list(),
            proxybar_data_list,
            block=False)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        if self._pbar is not None:
            self._pbar.start_updater()

        if wait is True:
            self.wait_parallel_simulation()

    def wait_parallel_simulation(self):
        """
        Wait for the parallel simulation to finish and then update the
        self.results variable (as well as other internal variables).
        """
        # Note that at the end of this method we set self._async_results to
        # None. Therefore, if wait_parallel_simulation is called multiple
        # times nothing will happen.
        if self._async_results is not None:
            # Wait for the tasks (running in the IPython engines) to finish
            self._async_results.wait()

            # Update the elapsed time
            self._elapsed_time += self._async_results.elapsed

            results = self._async_results.get()

            for reps, r, filename in results:
                self._runned_reps.append(reps)
                self.results.append_all_results(r)
                self.__results_base_filename_unpack_list.append(filename)

            # xxxxx Save the elapsed time in the SimulationResults object xxxxx
            self.results.elapsed_time = self._elapsed_time
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # Implement the _on_simulate_finish method in a subclass if you
            # need to run code at the end of the simulate method.
            self._on_simulate_finish()

            # xxxxxxx Save the number of runned iterations xxxxxxxxxxxxxxxxxxxx
            self.results.runned_reps = self._runned_reps
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Update the elapsed time xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            # Note that for now the elapsed time does not include the time
            # spent at the actual simulation. We still need to sum with the
            # elapsed time from the actual simulation.
            self.__toc = time()
            self._elapsed_time = self.__toc - self.__tic
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Save the results if results_filename is not None xxxxxxxxxx
            if self.__results_base_filename is not None:
                self.results.save_to_file(self.results_filename)
                # Delete the partial results (this will only delete the partial
                # results if self.delete_partial_results_bool is True)
                self.__delete_partial_results()
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # Stop the progressbar updating progress
            if self.update_progress_function_style is not None:
                self._pbar.stop_updater()

            # Erase the self._async_results object, since we already got all
            # information we needed from it
            self._async_results = None

    def _on_simulate_start(self):
        """This method is called only once, in the beginning of the the
        simulate method.

        """
        pass

    def _on_simulate_finish(self):
        """This method is called only once at the end of the simulate method.

        """
        pass

    def _on_simulate_current_params_start(self, current_params):
        """This method is called once for each simulation parameters
        combination before any iteration of _run_simulation is performed
        (for that combination of simulation parameters).

        Parameters
        ----------
        current_params : SimulationParameters object
            The current combination of simulation parameters.

        """
        pass

    def _on_simulate_current_params_finish(self, current_params,
                                           current_params_sim_results):
        """This method is called once for each simulation parameters
        combination after all iterations of _run_simulation are performed
        (for that combination of simulation parameters).

        Parameters
        ----------
        current_params : SimulationParameters object
            The current combination of simulation parameters.
        current_params_sim_results : SimulationResults object
            SimulationResults object with the results for the finished
            simulation with the parameters in current_params.

        """
        pass
# xxxxxxxxxx SimulationRunner - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationParameters - START xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TODO: Save the _unpack_index member variable in the save methods to hdf5
# and pytables. After that, load this information in the corresponding load
# methods.
class SimulationParameters(object):
    """Class to store the simulation parameters.

    A SimulationParameters object acts as a container for all simulation
    parameters. To add a new parameter to the object just call the
    :meth:`add` method passing the name and the value of the parameter. The
    value can be anything as long as the
    :meth:`.SimulationRunner._run_simulation` method can understand it.

    Alternatively, you can create a SimulationParameters object with all
    the parameters with the static method :meth:`create`, which receives a
    dictionary with the parameter names as keys.

    Parameters can be marked to be "unpacked", as long as they are
    iterable, by calling the :meth:`set_unpack_parameter` method. Different
    simulations will be performed for every combination of parameters
    marked to be unpacked, with the other parameters kept constant.

    Examples
    --------

    - Create a new empty SimulationParameters object and add the individual
      parameters to it by calling its :meth:`add` method.

      .. code-block:: python

         params = SimulationParameters()
         params.add('p1', [1,2,3])
         params.add('p2', ['a','b'])
         params.add('p3', 15)

    - Creating a new SimulationParameters object with the static
      :meth:`create` function.

      .. code-block:: python

         p = {'p1':[1,2,3], 'p2':['a','b'],'p3':15}
         params=SimulationParameters.create(p)
         params.set_unpack_parameter('p1')
         params.set_unpack_parameter('p2')

    - We can then set some of the parameters to be unpacked with

      .. code-block:: python

         params.set_unpack_parameter('p1')


    See also
    --------
    SimulationResults : Class to store simulation results.
    SimulationRunner : Base class to implement Monte Carlo simulations.

    """
    def __init__(self):
        # Dictionary that will store the parameters. The key is the
        # parameter name and the value is the parameter value.
        self.parameters = {}

        # A set to store the names of the parameters that will be unpacked.
        # Note there is a property to get the parameters marked to be
        # unpacked, that is, the unpacked_parameters property.
        self._unpacked_parameters_set = set()

        # If this SimulationParameters object was unpacked into a list of
        # SimulationParameters objects then each of these new objects will
        # set this variable with its index in that list. In other words, if
        # this member variable in a SimulationParameters object was set to
        # a non-negative integer then that SimulationParameters object is
        # actually one of the unpacked variations of another
        # SimulationParameters object. The original SimulationParameters
        # object will be stored in the _original_sim_params member
        # variable.
        self._unpack_index = -1
        self._original_sim_params = None

    @property
    def unpacked_parameters(self):
        """Names of the parameters marked to be unpacked."""
        return list(self._unpacked_parameters_set)

    @staticmethod
    def _create(params_dict, unpack_index=-1, original_sim_params=None):
        """
        Creates a new SimulationParameters object.

        This static method provides a different way to create a
        SimulationParameters object, already containing the parameters in
        the `params_dict` dictionary.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters. Each dictionary key
            corresponds to a parameter's name, while the dictionary value
            corresponds to the actual parameter value..
        unpack_index : int
            Index of the created SimulationParameters object when it is
            part of the unpacked variations of another SimulationParameters
            object. See :meth:`get_unpacked_params_list`.
        original_sim_params : SimulationParameters object
            The original SimulationParameters object from which the
            SimulationParameters object that will be created by this method
            came from.

        Returns
        -------
        sim_params : SimulationParameters object
            The corresponding SimulationParameters object.
        """
        sim_params = SimulationParameters()
        sim_params.parameters = copy.deepcopy(params_dict)
        if unpack_index < 0:
            unpack_index = -1
        sim_params._unpack_index = unpack_index
        sim_params._original_sim_params = original_sim_params
        return sim_params

    @staticmethod
    def create(params_dict):
        """Creates a new SimulationParameters object.

        This static method provides a different way to create a
        SimulationParameters object, already containing the parameters in
        the `params_dict` dictionary.

        Parameters
        ----------
        params_dict : dict
            Dictionary containing the parameters. Each dictionary key
            corresponds to a parameter's name, while the dictionary value
            corresponds to the actual parameter value..

        Returns
        -------
        sim_params : SimulationParameters object
            The corresponding SimulationParameters object.
        """
        return SimulationParameters._create(params_dict)

    def add(self, name, value):
        """Adds a new parameter to the SimulationParameters object.

        If there is already a parameter with the same name it will be
        replaced.

        Parameters
        ----------
        name : str
            Name of the parameter.
        value : anything
            Value of the parameter.
        """
        self.parameters[name] = value

    def set_unpack_parameter(self, name, unpack_bool=True):
        """Set the unpack property of the parameter with name `name`.

        The parameter `name` must be already added to the
        SimulationParameters object and be an iterable.

        This is used in the SimulationRunner class.

        Parameters
        ----------
        name : str
            Name of the parameter to be unpacked.
        unpack_bool : bool, optional (default to True)
            True activates unpacking for `name`, False deactivates it.

        Raises
        ------
        ValueError
            If `name` is not in parameters or is not iterable.
        """
        if name in self.parameters.keys():
            if isinstance(self.parameters[name], Iterable):
                if unpack_bool is True:
                    self._unpacked_parameters_set.add(name)
                else:
                    self._unpacked_parameters_set.remove(name)
            else:
                raise ValueError("Parameter {0} is not iterable".format(name))
        else:
            raise ValueError("Unknown parameter: `{0}`".format(name))

    def __getitem__(self, name):
        """Return the parameter with name `name`.

        Easy access to a given parameter using the brackets syntax.

        Parameters
        ----------
        name : str
            Name of the desired parameter.

        Returns
        -------
        desired_param : anything
            The value of the parameter with name `name`.
        """
        return self.parameters[name]

    def __repr__(self):
        def modify_name(name):
            """Add an * in name if it is set to be unpacked"""
            if name in self._unpacked_parameters_set:
                name += '*'
            return name
        repr_list = []
        for name, value in self.parameters.items():
            repr_list.append("'{0}': {1}".format(modify_name(name), value))
        return '{%s}' % ', '.join(repr_list)

    def __len__(self):
        """Get the number of different parameters stored in the
        SimulationParameters object.

        Returns
        -------
        length : int
            The number of different parameters stored in the
            SimulationParameters object

        """
        return len(self.parameters)

    def __iter__(self):  # pragma: no cover
        """Get an iterator to the parameters in the SimulationParameters
        object.
        """
        return iter(self.parameters)

    def __eq__(self, other):
        """
        Compare two SimulationParameters objects.

        Two simulation parameters objects are considered equal if all
        parameters stored in both objects are the same, except for a
        parameter object called 'rep_max'.

        Parameters
        ----------
        other: SimulationParameters
            The other SimulationParameters to be compared with self.

        Returns
        -------
        True if both objects are considered to be equal, returns False
        otherwise.

        Notes
        -----
        The main usage for comparing if two SimulationParameters objects
        are equal is when loading partial results in the SimulationRunner
        class, where we need to assure we are not combining results for
        different simulation parameters. The SimulationRunner class must
        check if the loaded results parameters match the current parameters
        to be simulated and thus require the "==" operator (or the "!="
        operator) to be implemented.

        However, it makes sense to ignore a parameter called 'rep_max',
        since it is not a parameter related to a 'scenario'. It is used in
        the SimulationRunner class to indicate the maximum number of
        iterations to perform and there is no problem when its value is
        different.
        """
        if self is other:
            return True

        if self._unpacked_parameters_set != other._unpacked_parameters_set:
            return False

        if set(self.parameters.keys()) != set(other.parameters.keys()):
            return False

        for key in self.parameters.keys():
            # We care about all keys, except for a key called 'rep_max'
            # whose value does not matter when comparing if two
            # SimulationResults objects are equal or not.
            if key != "rep_max":
                if np.any(self.parameters[key] != other.parameters[key]):
                    return False

        # If we didn't return until we reach this point then the objects are equal
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_num_unpacked_variations(self):
        """Get the number of variations when the parameters are unpacked.

        If no parameter was marked to be unpacked, then return 1.

        Returns
        -------
        num : int
            The number of variations when the parameters are unpacked.

        """
        if len(self._unpacked_parameters_set) == 0:
            return 1
        else:
            # Generator for the lengths of the parameters set to be unpacked
            gen_values = (len(self.parameters[i]) for i in self._unpacked_parameters_set)
            # Just multiply all the lengths
            from functools import reduce
            return reduce(lambda x, y: x * y, gen_values)

    def get_pack_indexes(self, fixed_params_dict=dict()):
        """When you call the function get_unpacked_params_list you get a
        list of SimulationParameters objects corresponding to all
        combinations of the parameters. The function get_pack_indexes
        allows you to provided all parameters marked to be unpacked except
        one, and returns the indexes of the list returned by
        get_unpacked_params_list that you want.

        Parameters
        ----------
        fixed_params_dict : dict
            A ditionary with the name of the fixed parameters as keys and
            the fixed value as value.

        Returns
        -------
        indexes : 1D numpy array or an integer
            The desired indexes.

        Examples
        --------
        Suppose we have

        >>> p={'p1':[1,2,3], 'p2':['a','b'],'p3':15}
        >>> params=SimulationParameters.create(p)
        >>> params.set_unpack_parameter('p1')
        >>> params.set_unpack_parameter('p2')

        If we call params.get_unpacked_params_list we will get a list of
        SimulationParameters objects, one for each combination of the
        values of p1 and p2. That is,

        >>> params.get_unpacked_params_list()
        [{'p2': a, 'p3': 15, 'p1': 1}, {'p2': a, 'p3': 15, 'p1': 2}, {'p2': a, 'p3': 15, 'p1': 3}, {'p2': b, 'p3': 15, 'p1': 1}, {'p2': b, 'p3': 15, 'p1': 2}, {'p2': b, 'p3': 15, 'p1': 3}]

        Likewise, in the simulation the SimulationRunner object will return
        a list of results in the order corresponding to the order of the
        list of parameters. The get_pack_indexes is used to get the index
        of the results corresponding to a specific configuration of
        parameters. Suppose now you want the results when 'p2' is varying,
        but with the other parameters fixed to some specific value. For
        this create a dictionary specifying all parameters except 'p2' and
        call get_pack_indexes with this dictionary. You will get an array
        of indexes that can be used in the results list to get the desired
        results. For instance

        >>> fixed={'p1':3,'p3':15}
        >>> params.get_pack_indexes(fixed)
        array([2, 5])
        """
        # Get the only parameter that was not fixed
        varying_param = list(
            self._unpacked_parameters_set - set(fixed_params_dict.keys()))

        # List to store the indexes (as strings) of the fixed parameters,
        # as well as ":" for the varying parameter,
        param_indexes = []
        for i in self.unpacked_parameters:
            if i in varying_param:
                param_indexes.append(':')
            else:
                fixed_param_value_index = list(self.parameters[i]).index(fixed_params_dict[i])
                param_indexes.append(str(fixed_param_value_index))

        # xxxxx Get the indexes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # For this we create a auxiliary numpy array going from 0 to the
        # number of unpack variations. The we use param_indexes to build a
        # string that we can evaluate using the auxiliary numpy array in
        # order to get the linear indexes.

        # Get the lengths of the parameters marked to be unpacked
        dimensions = [len(self.parameters[i]) for i in self.unpacked_parameters]
        aux = np.arange(0, self.get_num_unpacked_variations())
        aux.shape = dimensions
        indexes = eval("aux" + "[{0}]".format(",".join(param_indexes)))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return indexes

    def get_unpacked_params_list(self):
        """Get a list of SimulationParameters objects, each one
        corresponding to a possible combination of (unpacked) parameters.

        Returns
        -------
        unpacked_parans : list
           A list of SimulationParameters objecs.

        Examples
        --------
        Supose you have a SimulationParameters object with the parameters
        'a', 'b', 'c' and 'd' as below

        >>> simparams = SimulationParameters()
        >>> simparams.add('a', 1)
        >>> simparams.add('b', 2)
        >>> simparams.add('c', [3, 4])
        >>> simparams.add('d', [5, 6])

        and the parameters 'c' and 'd' were set to be unpacked.

        >>> simparams.set_unpack_parameter('c')
        >>> simparams.set_unpack_parameter('d')

        Then get_unpacked_params_list would return a list of four
        SimulationParameters objects as below (the order may be different)

        >>> simparams.get_unpacked_params_list()
        [{'a': 1, 'c': 3, 'b': 2, 'd': 5}, {'a': 1, 'c': 3, 'b': 2, 'd': 6}, {'a': 1, 'c': 4, 'b': 2, 'd': 5}, {'a': 1, 'c': 4, 'b': 2, 'd': 6}]

        That is

        .. code-block:: python

           [{'a': 1, 'c': 3, 'b': 2, 'd': 5},
            {'a': 1, 'c': 3, 'b': 2, 'd': 6},
            {'a': 1, 'c': 4, 'b': 2, 'd': 5},
            {'a': 1, 'c': 4, 'b': 2, 'd': 6}]

        """
        # If unpacked_parameters is empty, return self
        if not self._unpacked_parameters_set:
            return [self]

        # Lambda function to get an iterator to a (iterable) parameter
        # given its name. This only works if self.parameters[name] is an
        # iterable.
        get_iter_from_name = lambda name: iter(self.parameters[name])

        # Dictionary that stores the name and an iterator of a parameter
        # marked to be unpacked
        unpacked_params_iter_dict = OrderedDict()
        for i in self._unpacked_parameters_set:
            unpacked_params_iter_dict[i] = get_iter_from_name(i)
        keys = unpacked_params_iter_dict.keys()

        # Using itertools.product we can convert the multiple iterators
        # (for the different parameters marked to be unpacked) to a single
        # iterator that returns all the possible combinations (cartesian
        # product) of the individual iterators.
        all_combinations = itertools.product(*(unpacked_params_iter_dict.values()))

        # Names of the parameters that don't need to be unpacked
        regular_params = set(self.parameters.keys()) - self._unpacked_parameters_set

        # Constructs a list with dictionaries, where each dictionary
        # corresponds to a possible parameters combination
        unpack_params_length = len(self._unpacked_parameters_set)
        all_possible_dicts_list = []
        for comb in all_combinations:
            new_dict = {}
            # Add current combination of the unpacked parameters
            for index in range(unpack_params_length):
                new_dict[keys[index]] = comb[index]
            # Add the regular parameters
            for param in regular_params:
                new_dict[param] = self.parameters[param]
            all_possible_dicts_list.append(new_dict)

        # Map the list of dictionaries to a list of SimulationParameters
        # objects and return it.
        #
        # Note that since we are passing the index "i" to each new object
        # in the list as well as the original SimulationParameters object
        # "self", then each SimulationParameters object in the returned
        # list will know its index in that list (the _unpack_index
        # variable) as well as the original SimulationParameters object
        # from where it came from (stored in the _original_sim_params
        # variable).
        sim_params_list = [SimulationParameters._create(v, i, self) for i, v in
                           enumerate(all_possible_dicts_list)]
        return sim_params_list

    def save_to_pickled_file(self, filename):
        """
        Save the SimulationParameters object to the file `filename` using
        pickle.

        Parameters
        ----------
        filename : str
            Name of the file to save the parameters.
        """
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_pickled_file(filename):
        """
        Load the SimulationParameters from the file 'filename' previously
        stored (using pickle) with `save_to_pickled_file`.

        Parameters
        ----------
        filename : src
            Name of the file from where the results will be loaded.
        """
        with open(filename, 'rb') as inputfile:
            obj = pickle.load(inputfile)
        return obj

    @staticmethod
    def load_from_config_file(filename, spec=[], save_parsed_file=False):
        """
        Load the SimulationParameters from a config file using the configobj
        module.

        If the config file has a parameter called `unpacked_parameters`,
        which should be a list of strings with the names of other
        parameters, then these parameters will be set to be unpacked.

        Parameters
        ----------
        filename : src
            Name of the file from where the results will be loaded.
        spec : A list of stringsstr
            A list of strings with the confog spec. See "validation" in the
            configobj module documentation for more info.
        save_parsed_file : bool
            If `save_parsed_file` is True, then the parsed config file will
            be written back to disk. This will add any missing values in
            the config file whose default values are provided in the
            `spec`. This will even create the file if all default values
            are provided in `spec` and the file does not exist yet.

        Notes
        -----
        Besides the usual checks that the configobj validation has such as
        `integer`, `string`, `option`, etc., you can also use
        `real_numpy_array` for numpy float arrays. Note that when this
        validation function is used you can set the array in the config
        file in several ways such as
            SNR=0,5,10,15:20
        for instance.
        """
        from configobj import ConfigObj, flatten_errors
        from validate import Validator

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        def add_params(simulation_params, config):
            """
            Add the parameters in `config`.

            Parameters
            ----------
            simulation_params : A SimulationParameters object
                The SimulationParameters object where the parameters will
                be added.
            config : A configobj.ConfigObj or a configobj.Section object
                A ConfigObj object or a Section object. The `config` object
                can contain parameters (called scalars) or sections which
                can contain either parameters or other sections.
            """
            # Add scalar parameters
            for v in config.scalars:
                simulation_params.add(v, config[v])

            for s in config.sections:
                add_params(simulation_params, config[s])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        conf_file_parser = ConfigObj(
            filename,
            list_values=True,
            configspec=spec)

        # Dictionary with custom validation functions. Here we add a
        # validation function for numpy float arrays.
        fdict = {'real_numpy_array': _real_numpy_array_check,
                 'integer_numpy_array': _integer_numpy_array_check}
        validator = Validator(fdict)

        # The 'copy' argument indicates that if we save the ConfigObj
        # object to a file after validating, the default values will also
        # be written to the file.
        result = conf_file_parser.validate(validator,
                                           preserve_errors=True,
                                           copy=True)

        # Note that if thare was no parsing errors, then "result" will be
        # 'True'.  It there was an error, then result will be a dictionary
        # with each parameter as a key. The value of each key will be
        # either 'True' if that parameter was parsed without error or a
        # "validate.something" object (since we set preserve_errors to
        # True) describing the error.

        # if result != True:
        #     print 'Config file validation failed!'
        #     sys.exit(1)

        # xxxxx Test if there was some error in parsing the file xxxxxxxxxx
        # The flatten_errors function will return only the parameters whose
        # parsing failed.
        errors_list = flatten_errors(conf_file_parser, result)
        if len(errors_list) != 0:
            first_error = errors_list[0]
            # The exception will only describe the error for the first
            # incorrect parameter.
            if first_error[2] is False:
                raise Exception("Parameter '{0}' in section '{1}' must be provided.".format(first_error[1], first_error[0][0]))
            else:
                raise Exception("Parameter '{0}' in section '{1}' is invalid. {2}".format(first_error[1], first_error[0][0], first_error[2].message.capitalize()))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Finally add the parsed parameters to the params object xxxx
        params = SimulationParameters()
        add_params(params, conf_file_parser)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        if save_parsed_file is True:  # pragma: no cover
            # xxxxx Write the parsed config file to disk xxxxxxxxxxxxxxxxxx
            # This will add the default values if they are not present. If
            # the file does not exist yet and all default values are
            # provided in the spec then the file will be created. If some
            # parameter without a default value was not provided then when
            # exception would already have been thrown and we wouldn't be
            # here.
            conf_file_parser.write()
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # If the special parameter 'unpacked_parameters' is in the config
        # file, then we will set the parameters whose name are in it to be
        # unpacked
        try:
            unpacked_parameters_list = params['unpacked_parameters']
        except KeyError:
            unpacked_parameters_list = []
        for name in unpacked_parameters_list:
            params.set_unpack_parameter(name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return params

    def save_to_hdf5_group(self, group):
        """Save the contents of the SimulationParameters object into an
        HDF5 group.

        This function is called in the save_to_hdf5_file function in the
        SimulationResults class.

        Parameters
        ----------
        group : An HDF5 group
            The group where the parameters will be saved.

        Notes
        -----
        This method is called from the save_to_hdf5_file method in the
        SimulationResults class. It uses the python h5py library and
        `group` is supposed to be an HDF5 group created with that library.

        See also
        --------
        load_from_hdf5_group
        """
        # Store each parameter in self.parameter in a different dataset
        for name, value in self.parameters.iteritems():
            ds = group.create_dataset(name, data=value)
            # Save the TITTLE attribute to be more consistent with what
            # Pytables would do.
            ds.attrs.create("TITLE", name)

        # Store the _unpacked_parameters_set as an attribute of the group.
        # Note that we need to convert _unpacked_parameters_set to a list,
        # since a set has no native HDF5 equivalent.
        group.attrs.create('_unpacked_parameters_set', data=list(self._unpacked_parameters_set))

    def save_to_pytables_group(self, group):
        """Save the contents of the SimulationParameters object into an
        Pytables group.

        This function is called in the save_to_pytables_file function in the
        SimulationResults class.

        Parameters
        ----------
        group : A Pytables group
            The group where the parameters will be saved.

        Notes
        -----
        This method is called from the save_to_pytables_file method in the
        SimulationResults class. It uses the python pytables library and
        `group` is supposed to be an pytables group created with that library.

        See also
        --------
        load_from_pytables_group
        """
        pytables_file = group._v_file

        # Store each parameter in self.parameter in a different dataset
        for name, value in self.parameters.iteritems():
            pytables_file.createArray(group, name, value, title=name)

        # Store the _unpacked_parameters_set as an attribute of the group.
        # Note that we need to convert _unpacked_parameters_set to a list,
        # since a set has no native HDF5 equivalent.

        # TODO: Currently the atribute will be saved as a python object,
        # but it should be an array of strings.
        pytables_file.setNodeAttr(group, '_unpacked_parameters_set', list(self._unpacked_parameters_set))

    @staticmethod
    def load_from_hdf5_group(group):
        """Load the simulation parameters from an HDF5 group.

        This function is called in the load_from_hdf5_file function in the
        SimulationResults class.

        Notes
        -----
        This method is called from the load_from_hdf5_file method in the
        SimulationResults class. It uses the python h5py library and
        `group` is supposed to be an HDF5 group created with that library.

        Parameters
        ----------
        group : An HDF5 group
            The group from where the parameters will be loaded.

        Returns
        -------
        params : A SimulationParameters object.
            The SimulationParameters object loaded from `group`.

        See also
        --------
        save_to_hdf5_group
        """
        params = SimulationParameters()

        for name, ds in group.iteritems():
            params.add(name, ds.value)

        params._unpacked_parameters_set = set(group.attrs['_unpacked_parameters_set'])
        return params
# xxxxxxxxxx SimulationParameters - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationResults - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SimulationResults(object):
    """Store results from simulations.

    This class is used in the SimulationRunner class in order to store
    results from a simulation. It is able to combine the results from
    multiple iterations (of the :meth:`.SimulationRunner._run_simulation`
    method in the :class:`SimulationRunner` class) as well as append
    results for different simulation parameters configurations.

    .. note::

       Each result stored in the :class:`SimulationResults` object is in
       fact an object of the :class:`Result` class. This is required so
       that multiple :class:`SimulationResults` objects can be merged
       together, since the logic to merge each individual result is in the
       the :class:`Result` class.

    Examples
    --------
    - Creating a SimulationResults onject and adding a few results to it

      .. code-block:: python

         result1 = Result.create(...)  # See the Result class for details
         result2 = Result.create(...)
         result3 = Result.create(...)
         simresults = SimulationResults()
         simresults.add_result(result1)
         simresults.add_result(result2)
         simresults.add_result(result3)

      Instead of explicitly create a Result object and add it to the
      SimulationResults object, we can also create the Result object
      on-the-fly when adding it to the SimulationResults object by using
      the :meth:`add_new_result` method.

      That is

      .. code-block:: python

         simresults = SimulationResults()
         simresults.add_new_result(...)
         simresults.add_new_result(...)
         simresults.add_new_result(...)

    - Merging multiple SimulationResults objects

      .. code-block:: python

         # First SimulationResults object
         simresults = SimulationResults()
         # Create a Result object
         result = Result.create('some_name', Result.SUMTYPE, 4)
         # and add it to the SimulationResults object.
         simresults.add_result(result)

         # Second SimulationResults object
         simresults2 = SimulationResults()
         # We can also create the Result object on-the-fly when adding it
         # to the SimulationResults object to save one line.
         simresults2.add_new_result('some_name', Result.SUMTYPE, 6)

         # We can merge the results in the second SimulationResults object.
         # Since the update type of the single result stored is SUMTYPE,
         # then the simresults will now have a single Result of SUMTYPE
         # type with a value of 10.
         simresults.merge_all_results(simresults)

    See Also
    --------
    SimulationRunner : Base class to implement Monte Carlo simulations.
    SimulationParameters : Class to store the simulation parameters.
    Result : Class to store a single simulation result.

    """
    def __init__(self):
        self._results = dict()

        # This will store the simulation parameters used in the simulation
        # that resulted in the results. This should be set by calling the
        # set_parameters method.
        self._params = SimulationParameters()

    def _get_params(self):
        """Get method for the params property."""
        return self._params
    params = property(_get_params)

    def set_parameters(self, params):
        """Set the parameters of the simulation used to generate the
        simulation results stored in the SimulationResults object.

        Parameters
        ----------
        params : SimulationParameters
            A SimulationParameters object containing the simulation
            parameters.

        """
        if not isinstance(params, SimulationParameters):
            raise ValueError('params must be a SimulationParameters object')
        self._params = params

    def __repr__(self):
        list_of_names = self._results.keys()
        repr_string = "SimulationResults: {0}".format(sorted(list_of_names))
        return repr_string

    def add_result(self, result):
        """Add a result object to the SimulationResults object.

        .. note::

           If there is already a result stored with the same name, this
           will replace it.

        Parameters
        ----------
        result : An object of the :class:`Result` class
            Must be an object of the Result class.

        """
        # Added as a list with a single element
        self._results[result.name] = [result]

    def add_new_result(self, name, update_type, value, total=0):
        """Create a new Result object on the fly and add it to the
        SimulationResults object.

        .. note::

           This is Equivalent to the code below,

           .. code-block:: python

              result = Result.create(name, update_type, value, total)
              self.add_result(result)

           which in fact is exactly how this method is implemented.

        Parameters
        ----------
        name : str
            Name of the Result.
        update_type : {SUMTYPE, RATIOTYPE, MISCTYPE}
            Type of the result (SUMTYPE, RATIOTYPE or MISCTYPE).
        value : anything, but usually a number
            Value of the result.
        total : same type as `value`
            Total value of the result (used only for the RATIOTYPE and
            ignored for the other types).

        """
        result = Result.create(name, update_type, value, total)
        self.add_result(result)

    def append_result(self, result):
        """Append a result to the SimulationResults object. This
        efectivelly means that the SimulationResults object will now store
        a list for the given result name. This allow you, for instance, to
        store multiple bit error rates with the 'BER' name such that
        simulation_results_object['BER'] will return a list with the Result
        objects for each value.

        Parameters
        ----------
        result : An object of the :class:`Result` class
            Must be an object of the Result class.

        Notes
        -----
        If multiple values for some Result are stored, then only the last
        value can be updated with :meth:`merge_all_results`.

        Raises
        ------
        ValueError
            If the `result` has a different type from the result previously
            stored.

        See also
        --------
        append_all_results, merge_all_results

        """
        if result.name in self._results.keys():
            update_type_code = self._results[result.name][0]._update_type_code
            if update_type_code == result._update_type_code:
                self._results[result.name].append(result)
            else:
                raise ValueError("Can only append to results of the same type")
        else:
            self.add_result(result)

    def append_all_results(self, other):
        """Append all the results of the other SimulationResults object
        with self.

        Parameters
        ----------
        other : An object of the :class:`SimulationResults` class.
            Another SimulationResults object

        See also
        --------
        append_result, merge_all_results
        """
        for results in other:
            # There can be more then one value for the same result name
            for result in results:
                self.append_result(result)

    def merge_all_results(self, other):
        """Merge all the results of the other SimulationResults object with
        the results in self.

        When there is more then one result with the same name stored in
        self (for instance two bit error rates) then only the last one will
        be merged with the one in `other`. That also means that only one
        result for that name should be stored in `other`.

        Parameters
        ----------
        other : An object of the :class:`SimulationResults` class.
            Another SimulationResults object

        See also
        --------
        append_result, append_all_results

        """
        # If the current SimulationResults object is empty, we basically
        # copy the Result objects from other
        if len(self) == 0:
            for name in other.get_result_names():
                self._results[name] = other[name]
        # Otherwise, we merge each Result from `self` with the Result from
        # `other`
        else:
            for item in self._results.keys():
                self._results[item][-1].merge(other[item][-1])

    def get_result_names(self):
        """Get the names of all results stored in the SimulationResults
        object.

        Returns
        -------
        names : list
            The names of the results stored in the SimulationResults object.
        """
        return self._results.keys()

    def get_result_values_list(self, result_name, fixed_params={}):
        """
        Get the values for the results with name `result_name`.

        Returns a list with the values.

        Parameters
        ----------
        result_name : str
            The name of the desired result.
        fixed_params : dict
            A python dictionary containing the fixed parameters. If
            `fixed_params` is provided then the returned list will be only
            a subset of the results that match the fixed values of the
            parameters in the `fixed_params` argument, where the key is the
            parameter's name and the value is the fixed value. See the
            notes for an example.

        Returns
        -------
        result_list : list
            A list with the stored values for the result with name
            `result_name`

        Notes
        -----
        As an example of the usage of the `fixed_params` argument, suppose
        the results where obtained in a simulation for three parameters:
        'first', with value 'A', 'second' with value '[1, 2, 3]' and
        'third' with value '[B, C]', where the 'second' and 'third' were
        set to be unpacked. In that case the returned result list would
        have a length of 6 (the number of possible combinations of the
        parameters to be unpacked). If fixed_params is provided with the
        value of "{'second': 2}" that means that only the subset of results
        which corresponding to the second parameters having the value of
        '2' will be provided and the returned list will have a length of
        2. If fixed_params is provided with the value of "{'second': '1',
        'third': 'C'}" then a single result will be provided instead of a
        list.
        """
        # If the fictionary is not empty
        if fixed_params:
            # TODO: Test this part
            indexes = self.params.get_pack_indexes(fixed_params)
            out = [v.get_result() for i, v in enumerate(self[result_name])
                   if i in indexes]
        else:
            # If fixed_params is an empty dictionary (default value) then
            # we return the full list of results
            out = [v.get_result() for v in self[result_name]]
        return out

    def get_result_values_confidence_intervals(self,
                                               result_name,
                                               P=95,
                                               fixed_params={}):
        """
        Get the values for the results with name `result_name`.

        This method is similar to the `get_result_values_list` method, but
        instead of returning a list with the values it will return a list
        with the confidence intervals for those values.

        Parameters
        ----------
        result_name : str
            The name of the desired result.
        P : float
        fixed_params : dict
            A python dictionary containing the fixed parameters. If
            `fixed_params` is provided then the returned list will be only
            a subset of the results that match the fixed values of the
            parameters in the `fixed_params` argument, where the key is the
            parameter's name and the value is the fixed value. See the
            notes in the documentation of :meth:`get_result_values_list`
            for an example.

        Returns
        -------
        confidence_interval_list : list
            A list of Numpy (float) arrays. Each element in the list is an
            array with two elements, corresponding to the lower and upper
            limits of the confidence interval.8

        See also
        --------
        util.misc.calc_confidence_interval
        """
        if fixed_params:
            # TODO: Test this part
            indexes = self.params.get_pack_indexes(fixed_params)
            out = [v.get_confidence_interval(P) for i, v in enumerate(self[result_name])
                   if i in indexes]
        else:
            # If fixed_params is an empty dictionary (default value) then
            # we return the full list of results
            out = [i.get_confidence_interval(P) for i in self[result_name]]
        return out

    def __getitem__(self, key):
        """Get the value of the desired result

        Parameters
        ----------
        key : str
            Name of the desired result.

        Returns
        -------
        value :
            The desired result.
        """

        # if key in self._results.keys():
        return self._results[key]
        # else:
        #     raise KeyError("Invalid key: %s" % key)

    def __len__(self):
        """Get the number of results stored in self.

        Returns
        -------
        length : int
            Number of results stored in self.
        """
        return len(self._results)

    def __iter__(self):  # pragma: no cover
        # """Get an iterator to the internal dictionary. Therefore iterating
        # through this will iterate through the dictionary keys, that is, the
        # name of the results stored in the SimulationResults object.
        # """
        """Get an iterator to the results stored in the SimulationResults
        object.
        """
        try:
            # This is for python 2
            iterator = self._results.itervalues()
        except AttributeError:
            # This is for python 3
            iterator = iter(self._results.values())

        return iterator

    def save_to_file(self, filename):
        """Save the SimulationResults to the file `filename`.

        Parameters
        ----------
        filename : src
            Name of the file to save the results.

        """
        # For python3 compatibility the file must be opened in binary mode
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(filename):
        """Load the SimulationResults from the file `filename`.

        Parameters
        ----------
        filename : src
            Name of the file from where the results will be loaded.

        Returns
        -------
        simresults : A SimulationResults object
            The SimulationResults object loaded from the file `filename`.
        """
        with open(filename, 'rb') as inputfile:
            obj = pickle.load(inputfile)
        return obj

    def save_to_hdf5_file(self, filename, attrs={}):
        """Save the SimulationResults to the file `filename` using the HDF5
        format standard.

        Parameters
        ----------
        filename : src
            Name of the file to save the results.
        attrs : a dictionary
            Extra attributes to add to the HDF5 file.

        See also
        --------
        load_from_hdf5_file
        """
        import h5py
        fid = h5py.File(filename, 'w')

        # Save the TITTLE attribute to be more consistent with what
        # Pytables would do.
        fid.attrs.create("TITLE", "Simulation Results file")

        # Add the attributes, if any
        if isinstance(attrs, dict):  # pragma: no cover
            # attr is a dictionary of attributes
            for name, value in attrs.items():
                fid.attrs.create(name, value)

        # xxxxxxxxxx Save the results in the 'results' group xxxxxxxxxxxxxx
        g = fid.create_group('results')
        # Save the TITTLE attribute to be more consistent with what
        # Pytables would do.
        g.attrs.create("TITLE", "Simulation Results")
        for r in self:
            Result.save_to_hdf5_dataset(g, r)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Save the parameters in the 'parameters' group xxxxxxxx
        pg = fid.create_group('parameters')
        # Save the TITTLE attribute to be more consistent with what
        # Pytables would do.
        pg.attrs.create("TITLE", "Simulation Parameters")
        self.params.save_to_hdf5_group(pg)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        fid.close()

    # TODO: Test if this method saves all the information that the
    # save_to_hdf5_file method saves.
    def save_to_pytables_file(self, filename, attrs={}):
        import tables as tb
        fid = tb.openFile(filename, 'w', title='Simulation Results file')

        # Add the attributes, if any
        if isinstance(attrs, dict):  # pragma: no cover
            # attr is a dictionary of attributes
            for name, value in attrs.items():
                fid.setNodeAttr(fid.root, name, value)

        # xxxxxxxxxx Save the results in the 'results' group xxxxxxxxxxxxxx
        g = fid.createGroup(fid.root, 'results', title="Simulation Results")
        for r in self:
            Result.save_to_pytables_table(g, r)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Save the parameters in the 'parameters' group xxxxxxxx
        pg = fid.createGroup(fid.root, 'parameters', title="Simulation Parameters")
        self.params.save_to_pytables_group(pg)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        fid.close()

    @staticmethod
    def load_from_hdf5_file(filename):
        """Load a SimulationResults object from an HDF5 file saved with the
        save_to_hdf5_file method.

        Parameters
        ----------
        filename : src
            Name of the file to save the results.

        Returns
        -------
        simresults : A SimulationResults object.
            The SimulationResults object loaded from the file.

        See also
        --------
        save_to_hdf5_file
        """
        simresults = SimulationResults()

        import h5py
        fid = h5py.File(filename, 'r')

        # xxxxxxxxxx Results group xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        rg = fid['results']

        for result_name in rg:
            ds = rg[result_name]
            simresults._results[result_name] = Result.load_from_hdf5_dataset(ds)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Parameters grop xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        try:
            # We only set the simulation parameters if it was stored in the
            # hdf5 file.
            pg = fid['parameters']
            simresults._params = SimulationParameters.load_from_hdf5_group(pg)
        except KeyError:
            pass

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        fid.close()
        return simresults

# xxxxxxxxxx SimulationResults - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Result - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Result(object):
    """Class to store a single simulation result.

    A simulation result can be anything, such as the number of errors, a
    string, an error rate, etc. When creating a `Result` object one needs to
    specify only the `name` of the stored result and the result `type`.

    The different types indicate how multiple samples (from multiple
    iterations) of the same Result can be merged (usually to get a result
    with more statistical reliability). The possible values are SUMTYPE,
    RATIOTYPE and MISCTYPE.

    In the `SUMTYPE` the new value should be added to current one in update
    function.

    In the `RATIOTYPE` the new value should be added to current one and
    total should be also updated in the update function. One caveat is that
    rates are stored as a number (numerator) and a total (denominator)
    instead of as a float. For instance, if you need to store a result such
    as a bit error rate, then you could use the a Result with the RATIOTYPE
    type and when updating the result, pass the number of bit errors and
    the number of simulated bits.

    The `MISCTYPE` type can store anything and the update will simple
    replace the stored value with the current value.

    Examples
    --------
    - Example of the SUMTYPE result.

      >>> result1 = Result("name", Result.SUMTYPE)
      >>> result1.update(13)
      >>> result1.update(4)
      >>> result1.get_result()
      17
      >>> result1.num_updates
      2
      >>> result1
      Result -> name: 17
      >>> result1.type_name
      'SUMTYPE'
      >>> result1.type_code
      0

    - Example of the RATIOTYPE result.

      >>> result2 = Result("name2", Result.RATIOTYPE)
      >>> result2.update(4,10)
      >>> result2.update(3,4)
      >>> result2.get_result()
      0.5
      >>> result2.type_name
      'RATIOTYPE'
      >>> result2.type_code
      1
      >>> result2_other = Result("name2", Result.RATIOTYPE)
      >>> result2_other.update(3,11)
      >>> result2_other.merge(result2)
      >>> result2_other.get_result()
      0.4
      >>> result2_other.num_updates
      3
      >>> result2_other._value
      10
      >>> result2_other._total
      25
      >>> result2.get_result()
      0.5
      >>> print result2_other
      Result -> name2: 10/25 -> 0.4

    - Example of the MISCTYPE result.

      The MISCTYPE result 'merge' process in fact simple replaces the
      current stored value with the new value.

    """
    # Like an Enumeration for the type of results.
    (SUMTYPE, RATIOTYPE, MISCTYPE) = range(3)
    _all_types = {
        SUMTYPE: "SUMTYPE",
        RATIOTYPE: "RATIOTYPE",
        MISCTYPE: "MISCTYPE",
    }

    def __init__(self, name, update_type_code, accumulate_values=False):
        self.name = name
        self._update_type_code = update_type_code
        self._value = 0
        self._total = 0
        self.num_updates = 0  # Number of times the Result object was
                              # updated

        # Accumulation of values
        self._accumulate_values_bool = accumulate_values
        self._value_list = []
        self._total_list = []

    @property
    def accumulate_values_bool(self):
        """
        Property to see if values are accumulated of not during a call to the
        `update` method.
        """
        return self._accumulate_values_bool

    @staticmethod
    def create(name, update_type, value, total=0, accumulate_values=False):
        """
        Create a Result object and update it with `value` and `total` at
        the same time.

        Equivalent to creating the object and then calling its
        :meth:`update` method.

        Parameters
        ----------
        name : str
            Name of the Result.
        update_type : {SUMTYPE, RATIOTYPE, MISCTYPE}
            Type of the result (SUMTYPE, RATIOTYPE or MISCTYPE).
        value : anything, but usually a number
            Value of the result.
        total : same type as `value`
            Total value of the result (used only for the RATIOTYPE and
            ignored for the other types).
        accumulate_values : bool
            If True, then the values `value` and `total` will be
            accumulated in the `update` (and merge) method(s). This means
            that the Result object will use more memory as more and more
            values are accumulated, but having all values sometimes is
            useful to perform statistical calculations.

        Returns
        -------
        result : A Result object.
            The new Result object.

        Notes
        -----
        Even if accumulate_values is True the values will not be
        accumulated for the MISCTYPE.

        See also
        --------
        update
        """
        result = Result(name, update_type, accumulate_values)
        result.update(value, total)
        return result

    @property
    def type_name(self):
        """Get the Result type name.

        Returns
        -------
        type_name : str
            The result type string (SUMTYPE, RATIOTYPE or MISCTYPE).
        """
        return Result._all_types[self._update_type_code]

    @property
    def type_code(self):
        """Get the Result type.

        Returns
        -------
        type_code : int
            The returned value is a number corresponding to one of the
            types SUMTYPE, RATIOTYPE or MISCTYPE.

        """
        return self._update_type_code

    def __repr__(self):
        if self._update_type_code == Result.RATIOTYPE:
            v = self._value
            t = self._total
            if t != 0:
                return "Result -> {0}: {1}/{2} -> {3}".format(
                    self.name, v, t, float(v) / t)
            else:
                return "Result -> {0}: {1}/{2} -> NaN".format(
                    self.name, v, t)
        else:
            return "Result -> {0}: {1}".format(self.name, self.get_result())

    def update(self, value, total=None):
        """Update the current value.

        Parameters
        ----------
        value : anything, but usually a number
            Value to be added to (or replaced) the current value

        total : same type as `value`
            Value to be added to the current total (only useful for the
            RATIOTYPE update type)

        Notes
        -----
        The way how this update process depends on the Result type and is
        described below

        - RATIOTYPE: Add "value" to current value and "total" to current total
        - SUMTYPE: Add "value" to current value. "total" is ignored.
        - MISCTYPE: Replace the current value with "value".

        See also
        --------
        create

        """
        self.num_updates += 1

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Python does not have a switch statement. We use dictionaries as
        # the equivalent of a switch statement.
        # First we define a function for each possibility.
        def __default_update(dummy1, dummy2):
            """Default update method.

            This will only be called when the update type is not one of the
            available types. Thus, an exception will be raised.

            """
            raise ValueError("Can't update a Result object of type '{0}'".format(self._update_type_code))

        def __update_SUMTYPE_value(value, dummy):
            """Update the Result object when its type is SUMTYPE."""
            self._value += value
            if self._accumulate_values_bool is True:
                self._value_list.append(value)

        def __update_RATIOTYPE_value(value, total):
            """Update the Result object when its type is RATIOTYPE.

            Raises
            ------
            ValueError
                If the `total` parameter is None (not provided).
            """
            if total is None:
                raise ValueError("A 'value' and a 'total' are required when updating a Result object of the RATIOTYPE type.")

            self._value += value
            self._total += total

            if self._accumulate_values_bool is True:
                self._value_list.append(value)
                self._total_list.append(total)

        def __update_by_replacing_current_value(value, dummy):
            """Update the Result object when its type is MISCTYPE."""
            self._value = value
            if self._accumulate_values_bool is True:
                self._value_list.append(value)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Now we fill the dictionary with the functions
        possible_updates = {
            Result.RATIOTYPE: __update_RATIOTYPE_value,
            Result.MISCTYPE: __update_by_replacing_current_value,
            Result.SUMTYPE: __update_SUMTYPE_value}

        # Call the apropriated update method. If self._update_type_code does
        # not contain a key in the possible_updates dictionary (that is, a
        # valid update type), then the function __default_update is called.
        possible_updates.get(self._update_type_code,
                             __default_update)(value, total)

    def merge(self, other):
        """Merge the result from other with self.

        Parameters
        ----------
        other : Result object
            Another Result object.
        """
        # Both objects must be set to either accumulate or not accumulate
        assert self._accumulate_values_bool == other._accumulate_values_bool, (
            "Both objects must either accumulate or not accumulate values")

        # pylint: disable=W0212
        assert self._update_type_code == other._update_type_code, (
            "Can only merge two objects with the same name and type")
        assert self._update_type_code != Result.MISCTYPE, (
            "Cannot merge results of the MISCTYPE type")
        assert self.name == other.name, (
            "Can only merge two objects with the same name and type")
        self.num_updates += other.num_updates
        self._value += other._value  # pylint: disable=W0212
        self._total += other._total  # pylint: disable=W0212

        if self._accumulate_values_bool is True:
            self._value_list.extend(other._value_list)
            self._total_list.extend(other._total_list)

    def get_result(self):
        """Get the result stored in the Result object.

        Returns
        -------
        results : anything, but usually a number
            For the RATIOTYPE type get_result will return the
            `value/total`, while for the other types it will return
            `value`.

        """
        if self.num_updates == 0:
            return "Nothing yet".format(self.name)
        else:
            if self._update_type_code == Result.RATIOTYPE:
                #assert self._total != 0, 'Total should not be zero'
                return float(self._value) / self._total
            else:
                return self._value

    def get_confidence_interval(self, P=95):
        """
        Get the confidence inteval that contains the true result with a given
        probability `P`.

        Parameters
        ----------
        P : float
            The desired confidence (probability in %) that true value is
            inside the calculated interval. The possible values are
            described in the documentaiton of the
            `util.misc.calc_confidence_interval` function`

        Returns
        -------
        Interval : Numpy (float) array with two elements.

        See also
        --------
        util.misc.calc_confidence_interval
        """
        if len(self._value_list) == 0:
            if self._accumulate_values_bool is False:
                message = "get_confidence_interval: The accumulate_values option must be set to True."
            else:
                message = "get_confidence_interval: There are no stored values yet."
            raise RuntimeError(message)

        values = np.array(self._value_list, dtype=float)
        if self._update_type_code == Result.RATIOTYPE:
            values = values / np.array(self._total_list, dtype=float)

        mean = values.mean()
        std = values.std()
        n = values.size

        return calc_confidence_interval(mean, std, n, P)

    # TODO: Save the _value_list, _total_list and _accumulate_values_bool
    # variables
    @staticmethod
    def save_to_hdf5_dataset(parent, results_list):
        """Create an HDF5 dataset in `parent` and fill it with the Result
        objects in `results_list`.

        Parameters
        ----------
        parent : An HDF5 group (usually) or file.
            The parent that will contain the dataset.
        results_list : A python list of Result objects.
            A list of Result objects. All of these objects must have the
            same name and update type.

        Notes
        -----
        This method is called from the save_to_hdf5_file method in the
        SimulationResults class. It uses the python h5py library and
        `parent` is supposed to be an HDF5 group created with that library.

        See also
        --------
        load_from_hdf5_dataset

        """
        dtype = [('_value', float), ('_total', float), ('num_updates', int)]
        name = results_list[0].name
        size = len(results_list)
        ds = parent.create_dataset(name, shape=(size,), dtype=dtype)

        for i, r in enumerate(results_list):
            ds[i] = (r._value, r._total, r.num_updates)

        ds.attrs.create('update_type_code', data=r._update_type_code)
        # Save the TITTLE attribute to be more consistent with what
        # Pytables would do.
        ds.attrs.create("TITLE", name)

    # TODO: Save the _value_list, _total_list and _accumulate_values_bool
    # variables
    @staticmethod
    def save_to_pytables_table(parent, results_list):
        """
        """
        import tables as tb
        pytables_file = parent._v_file
        name = results_list[0].name
        description = {'_value': tb.FloatCol(), '_total': tb.FloatCol(), 'num_updates': tb.IntCol()}
        table = pytables_file.createTable(parent, name, description,
                                          title=name)
        row = table.row
        for r in results_list:
            row['_value'] = r._value
            row['_total'] = r._total
            row['num_updates'] = r.num_updates
            row.append()

        pytables_file.setNodeAttr(table, 'update_type_code', r._update_type_code)
        table.flush()

    @staticmethod
    def load_from_hdf5_dataset(ds):
        """Load a list of Rersult objects from an HDF5 dataset.

        This dataset was suposelly saved with the save_to_hdf5_dataset
        function.

        Parameters
        ----------
        ds : An HDF5 Dataset
            The dataset to be loaded.

        Returns
        -------
        results_list : A list of Result objects.
            The list of Result objects loaded from the dataset.

        Notes
        -----
        This method is called from the load_from_hdf5_file method in the
        SimulationResults class. It uses the python h5py library and
        `ds` is supposed to be an HDF5 dataset created with that library.

        See also
        --------
        save_to_hdf5_dataset

        """
        results_list = []

        name = ds.name.split('/')[-1]
        update_type_code = ds.attrs['update_type_code']
        for data in ds:
            r = Result.create(name,
                              update_type_code,
                              data['_value'],
                              data['_total'])
            r.num_updates = data['num_updates']
            results_list.append(r)
        return results_list

# xxxxxxxxxx Result - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


## HDF5
## http://scipy-user.10969.n7.nabble.com/Should-I-use-pickle-for-numpy-array-td144.html
# >>> import h5py
# >>> f = h5py.File('example.hdf5', 'w')
# >>> import numpy
# >>> f['my_array'] = numpy.arange(10)
# >>> f.close()

## PyTables
## http://stackoverflow.com/questions/8447926/loading-matlab-sparse-matrix-using-python-pytables

## Conclusions
# Its better to use h5py instead of PyTables. The h5py library provides an
# interface similar to numpy arrays and I don't need the extra abstraction
# layer from PyTables and its complex database like operations.


## Boa resposta
# http://stackoverflow.com/questions/10075661/how-to-save-dictionaries-and-arrays-in-the-same-archive-with-numpy-savez


## Quick Start Guide do h5py
# http://www.h5py.org/docs/intro/quick.html


# Você pode usar o programa vitables par visualizar os arquivos criados com
# o pytables ou hdf5.
