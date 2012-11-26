#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing useful classes to implement Monte Carlo simulations.

The main class for Monte Carlo simulations is the :class:`SimulationRunner`
class, but a few other classes are also implemented to handle simulation
parameters and simulation results.

More specifically, the :mod:`simulations` module implements the classes:
 - :class:`SimulationRunner`
 - :class:`SimulationParameters`
 - :class:`SimulationResults`
 - :class:`Result`

For a description of how to implement Monte Carlo simulations using the
clases defined in the :mod: simulations` module see the section below.

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
implement the _run_simulation method with the code to simulate a single
iteration for that specific simulator. The simulation can then be performed
by calling the "simulate" method of an object of that derived class. After
the simulation is finished, the 'results' parameter of that object will
have the simulation results.

The process of implementing a simulator is described in more details in the
following.

Simulation Parameters
~~~~~~~~~~~~~~~~~~~~~

The simulation parameters can be set in any way as long as they can be
accessed in the _run_simulation method. For parameters that won't be
changed, a simple way that works very well is to store these parameters as
attributes in the __init__ method.

On the other hand, if you want to run multiple simulations, each one with
different values for some of the simulation parameters then store these
parameters in the self.params attribute and set them to be unpacked (See
docummentation of the SimulationParameters class for more details). The
simulate method will automatically get all possible combinations of
parameters and perform a whole Monte Carlo simulation for each of them. The
simulate method will pass the 'current_parameters' (a SimulationParameters
object) to _run_simulation from where _run_simulation can get the current
combination of parameters.

If you want/need to save the simulation parameters for future reference,
however, then you should store all the simulation parameters in the
self.params attribute. This will allow you to call its save_to_file method
to save everything into a file. The simulation parameters can be recovered
latter from this file by calling the static method
SimulationParameters.load_from_file.

Simulation Results
~~~~~~~~~~~~~~~~~~

In the implementation of the _run_simulation method in a subclass of
SimulationRunner it is necessary to create an object of the
SimulationResults class, add each desided result to it (using the
add_result method of the SimulationResults class) and then return this
object at the end of _run_simulation. Note that each result added to this
SimulationResults object must itself be an object of the 'Result' class.

After each run of the _run_simulation method the returned SimulationResults
object is merged with the self.results attribute from where the simulation
results can be retreived after the simulation finishes. Note that the way
the results from each _run_simulation run are merged together depend on the
the Result 'update_type'.

Since you will have the complete simulation results in the self.results
object you can easily save them to a file calling its save_to_file method.

TIP: create a new 'params' attribute with the simulation parameters in the
self.results object before calling its save_to_file method. This way you
will have information about which simulation parameters were used to
generate the results.

Number of iterations the _run_simulation method is performed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of times the _run_simulation method is performed for a given
parameter combination depend on the self.rep_max attribute. It is set by
default to '1' and therefore you should set it to the desired value in the
__init__ method of the SimulationRunner subclass.

Optional methods
~~~~~~~~~~~~~~~~

A few methods can be implemented in the SimulationRunner subclass for extra
functionalities. The most useful one is probably the _keep_going method,
which can speed up the simulation by avoid running unecessary iterations of
the _run_simulation method.

Basically, after each iteration of the _run_simulation method the
_keep_going method is called. If it returns True then more iterations of
_run_simulation will be performed until _keep_going returns False or
rep_max iterations are performed. When the _keep_going method is called it
receives a SimulationResults object with the cumulated results from all
iterations so far, which it can then use to decide it the iterations should
continue or not.

The other optional methods provide hooks to run code at specific points of
the simulate method. They are described briefly below:

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

"""

__version__ = "$Revision$"

import pickle
from collections import OrderedDict, Iterable
import itertools
import copy
import numpy as np

from misc import pretty_time
from progressbar import ProgressbarText


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationRunner - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SimulationRunner(object):
    """Base class to run Monte Carlo simulations.
    """
    def __init__(self):
        self.rep_max = 1
        self._elapsed_time = 0.0
        self._runned_reps = []  # Number of iterations performed by
                                # simulation when it finished
        self.params = SimulationParameters()
        self.results = SimulationResults()

        # Message passed to the _get_update_progress_function function. The
        # message can contain "{SomeParameterName}" which will be replaced
        # with the parameter value.
        #
        # If, on the other hand, progressbar_message is None, then the
        # progressbar will be disabled.
        self.progressbar_message = 'Progress'

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
        raise NotImplementedError("This function must be implemented in a subclass")

    def _keep_going(self, current_sim_results):
        """Check if the simulation should continue or stop.

        This function may be reimplemented in the derived class if a stop
        condition besides the number of iterations is desired.  The idea is
        that _run_simulation returns a SimulationResults object, which is
        then passed to _keep_going, which is then in charge of deciding if
        the simulation should stop or not.

        Parameters
        ----------
        current_sim_results : SimulationResults object
            SimulationResults object from the last iteration (merged with
            all the previous results)

        Returns
        -------
        result : bool
            True if the simulation should continue or False otherwise.

        """
        # If this function is not reimplemented in a subclass it always
        # returns True. Therefore, the simulation will only stop when the
        # maximum number of allowed iterations is reached.
        return True

    def _get_update_progress_function(self, message=''):
        """Return a function that should be called to update the
        progressbar.

        The returned function accepts a single argument, corresponding to
        the number of iterations executed so far.

        Parameters
        ----------
        message : str
            The message to be written in the progressbar, if it is used.

        Returns
        -------
        func : function that accepts a single integer argument
            Function that accepts a single integer argument and can be
            called to update the progressbar.

        """
        # The returned function will update the bar
        self.bar = ProgressbarText(self.rep_max, '*', message)
        return lambda value: self.bar.progress(value)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    @property
    def elapsed_time(self):
        """property: Get the simulation elapsed time. Do not set this
        value."""
        return pretty_time(self._elapsed_time)

    @property
    def runned_reps(self):
        return self._runned_reps

    def simulate(self):
        """Implements the general code for every simulation. Any code
        specific to a single simulation iteration must be implemented in the
        _run_simulation method of a subclass of SimulationRunner.

        The main idea behind the SimulationRunner class is that the general
        code in every simulator is implemented in the SimulationRunner
        class, more specifically in the `simulate` method, while the
        specific code of a single iteration is implemented in the
        _run_simulation method in a subclass.

        """
        # xxxxxxxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        from time import time
        tic = time()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Implement the _on_simulate_start method in a subclass if you need
        # to run code at the start of the simulate method.
        self._on_simulate_start()

        # xxxxx FOR UNPACKED PARAMETERS xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Loop through all the parameters combinations
        for current_params in self.params.get_unpacked_params_list():
            # Get the update_progress_func function
            if self.progressbar_message is None:
                # If self.progressbar_message is None then
                # update_progress_func does nothing
                update_progress_func = lambda value: None
            else:
                update_progress_func = self._get_update_progress_function(
                    # If the progressbar_message has any string
                    # replacements in the form {some_param} where
                    # 'some_param' is a parameter in current_params then it
                    # will be replace by the current value of 'some_param'.
                    self.progressbar_message.format(**current_params.parameters))

            # Implement the _on_simulate_current_params_start method in a
            # subclass if you need to run code before the _run_simulation
            # iterations for each combination of simulation parameters.
            self._on_simulate_current_params_start(current_params)

            # Perform the first iteration of _run_simulation
            current_sim_results = self._run_simulation(current_params)
            current_rep = 1

            # Run more iterations until one of the stop criteria is reached
            while (self._keep_going(current_sim_results)
                   and
                   current_rep < self.rep_max):
                current_sim_results.merge_all_results(
                    self._run_simulation(current_params))
                update_progress_func(current_rep + 1)
                current_rep += 1

            # Implement the _on_simulate_current_params_finish method in a
            # subclass if you need to run code after all _run_simulation
            # iterations for each combination of simulation parameters.
            self._on_simulate_current_params_finish(current_params)

            # If the while loop ended before rep_max repetitions (because
            # _keep_going returned false) then set the progressbar to full.
            update_progress_func(self.rep_max)

            # Store the number of repetitions actually ran for the current
            # parameters combination
            self._runned_reps.append(current_rep)
            # Lets append the simulation results for the current parameters
            self.results.append_all_results(current_sim_results)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Update the elapsed time xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        toc = time()
        self._elapsed_time = toc - tic
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Implement the _on_simulate_finish method in a subclass if you
        # need to run code at the end of the simulate method.
        self._on_simulate_finish()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

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

    def _on_simulate_current_params_finish(self, current_params):
        """This method is called once for each simulation parameters
        combination after all iterations of _run_simulation are performed
        (for that combination of simulation parameters).

        Parameters
        ----------
        current_params : SimulationParameters object
            The current combination of simulation parameters.

        """
        pass
# xxxxxxxxxx SimulationRunner - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationParameters - START xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SimulationParameters(object):
    """Class to store the simulation parameters.

    A SimulationParameters object acts as a container for all simulation
    parameters. To add a new parameter to the object just call the `add`
    method passing the name and the value of the parameter. The value can
    be anything as long as the _run_simulation function can understand it.

    Parameters can be marked to be "unpacked", as long as they are
    iterable, by calling the set_unpack_parameter function. Different
    simulations will be performed for every combination of parameters
    marked to be unpacked, with the other parameters kept constant.
    """
    def __init__(self):
        # Dictionary that will store the parameters. The key is the
        # parameter name and the value is the parameter value.
        self.parameters = {}

        # A set to store the names of the parameters that will be unpacked.
        # Note there is a property to get the parameters marked to be
        # unpacked, that is, the unpacked_parameters property.
        self._unpacked_parameters_set = set()

    @property
    def unpacked_parameters(self):
        """Names of the parameters marked to be unpacked."""
        return list(self._unpacked_parameters_set)

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
        sim_params = SimulationParameters()
        sim_params.parameters = copy.deepcopy(params_dict)
        return sim_params

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
                self._unpacked_parameters_set.add(name)
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

    def get_num_unpacked_variations(self):
        """Get the number of variations when the parameters are unpacked.

        Returns
        -------
        num : int
            The number of variations when the parameters are unpacked.
        """
        # Generator for the lengths of the parameters set to be unpacked
        gen_values = (len(self.parameters[i]) for i in self._unpacked_parameters_set)
        # Just multiply all the lengths
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
        indexes : 1D numpy array
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
        assert len(varying_param) == 1, "All unpacked parameters must be fixed except one"
        # The only parameter still varying. That is, one parameter marked
        # to be unpacked, bu not in fixed_params_dict.
        varying_param = varying_param[0]  # List with one element

        # List to store the indexes (as strings) of the fixed parameters,
        # as well as ":" for the varying parameter,
        param_indexes = []
        for i in self.unpacked_parameters:
            if i == varying_param:
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
        # given its name
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
        # objects and return it
        return map(SimulationParameters.create, all_possible_dicts_list)

    def save_to_file(self, filename):
        """Save the SimulationParameters object to the file `filename`.

        Parameters
        ----------
        filename : str
            Name of the file to save the parameters.
        """
        with open(filename, 'w') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(filename):
        """Load the SimulationParameters from the file 'filename'.

        Parameters
        ----------
        filename : src
            Name of the file from where the results will be loaded.
        """
        with open(filename, 'r') as input:
            obj = pickle.load(input)
        return obj

# xxxxxxxxxx SimulationParameters - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimulationResults - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SimulationResults(object):
    """Store results from simulations.

    This class is used in the SimulationRunner class in order to store
    results from a simulation. It is able to combine the results from
    multiple iterations (of the _run_simulation method in the
    SimulationRunner class) as well as append results for different
    simulation parameters configurations.

    """
    def __init__(self):
        self._results = dict()

    def __repr__(self):
        lista = [i for i in self._results.keys()]
        repr = "SimulationResults: %s" % lista
        return repr

    def add_result(self, result):
        """Add a new result to the SimulationResults object. If there is
        already a result stored with the same name, this will replace it.

        Parameters
        ----------
        result : An object of the :class:`Result` class
            Must be an object of the Result class.
        """
        # Added as a list with a single element
        self._results[result.name] = [result]

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

        See also
        --------
        append_all_results, merge_all_results
        """
        if result.name in self._results.keys():
            self._results[result.name].append(result)
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

    def get_result_values_list(self, result_name):
        """Get the values for the results with name `result_name`.

        Returns a list with the values.

        Parameters
        ----------
        result_name : str
            The name of the desired result.

        Returns
        -------
        result_list : list
            A list with the stored values for the result with name
            `result_name`

        """
        return [i.get_result() for i in self[result_name]]

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

    def __iter__(self):
        # """Get an iterator to the internal dictionary. Therefore iterating
        # through this will iterate through the dictionary keys, that is, the
        # name of the results stored in the SimulationResults object.
        # """
        """Get an iterator to the results stored in the SimulationResults
        object.
        """
        return self._results.itervalues()

    def save_to_file(self, filename):
        """Save the SimulationResults to the file `filename`.

        Parameters
        ----------
        filename : src
            Name of the file to save the results.

        """
        with open(filename, 'w') as output:
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
        with open(filename, 'r') as input:
            obj = pickle.load(input)
        return obj

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

    def __init__(self, name, update_type_code):
        self.name = name
        self._update_type_code = update_type_code
        self._value = 0
        self._total = 0
        self.num_updates = 0  # Number of times the Result object was
                              # updated

    @staticmethod
    def create(name, update_type, value, total=0):
        """Create a Result object and update it with `value` and `total` at
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

        Returns
        -------
        result : A Result object.
            The new Result object.

        See also
        --------
        update
        """
        result = Result(name, update_type)
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
            raise ValueError("Can't update a Result object of type '{0}'".format(self._update_type_code))
            # print("Warning: update not performed for unknown type %s" %
            #       self._update_type_code)

        def __update_SUMTYPE_value(value, dummy):
            self._value += value

        def __update_RATIOTYPE_value(value, total):
            if total is None:
                raise ValueError("A 'value' and a 'total' are required when updating a Result object of the RATIOTYPE type.")

            assert value <= total, ("__update_RATIOTYPE_value: "
                                    "'value cannot be greater then total'")
            self._value += value
            self._total += total

        def __update_by_replacing_current_value(value, dummy):
            self._value = value
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
        assert self._update_type_code == other._update_type_code, (
            "Can only merge two objects with the same name and type")
        assert self._update_type_code != Result.MISCTYPE, (
            "Cannot merge results of the MISCTYPE type")
        assert self.name == other.name, (
            "Can only merge two objects with the same name and type")
        self.num_updates += other.num_updates
        self._value += other._value
        self._total += other._total

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
# xxxxxxxxxx Result - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
