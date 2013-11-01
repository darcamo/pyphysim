#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement classes to represent the progress of a task.

Use the ProgressbarText class for tasks that do not use multiprocessing,
and the ProgressbarMultiProcessServer class for tasks using multiprocessing.

Basically, the task code must call the "progress" function to update the
progress bar and pass a number equivalent to the increment in the progress
since the last call. The progressbar must know the maximum value equivalent
to all the progress, which is passed during object creator for
ProgressbarText class.

The ProgressbarMultiProcessServer is similar to ProgressbarText class,
accounts for the progress of multiple processes. For each process you need
to call the register_client_and_get_proxy_progressbar to get a proxy
progressbar, where the maximum value equivalent to all the progress that
will be performed by that process is passed in this proxy creation. Each
process then calls the progress method of the proxy progressbar.

Note that there is also a DummyProgressbar whose progress function does
nothing. This is useful when you want to give the user a choice to show or
not the progressbar such that the task code can always call the progress
method and you only change the progressbar object.
"""

from __future__ import print_function

__revision__ = "$Revision$"

import sys
import multiprocessing
import time

try:
    import zmq
except ImportError:  # pragma: no cover
    # We don't have a fallback for zmq, but only the ProgressbarZMQServer and
    # ProgressbarZMQClient classes require it
    pass

__all__ = ['DummyProgressbar', 'ProgressbarText', 'ProgressbarText2', 'ProgressbarText3', 'ProgressbarMultiProcessServer', 'ProgressbarZMQServer', 'center_message']


# TODO: Move this function to the misc module.
def center_message(message, length=50, fill_char=' ', left='', right=''):
    """Return a string with `message` centralized and surrounded by
    `fill_char`.

    Parameters
    ----------
    message: str
        The message to be centered.
    length : int
        Total length of the centered message (original + any fill).
    fill_char : str
        Filling character.
    left : str
        Left part of the filling.
    right : str
       Right part of the filling.

    Returns
    -------
    cent_message : str
        The centralized message.

    Examples
    --------
    >>> print(center_message("Hello World", 50, '-', 'Left', 'Right'))
    Left-------------- Hello World --------------Right
    """
    message_size = len(message)
    left_size = len(left)
    right_size = len(right)
    fill_size = (length - (message_size + 2) - left_size - right_size)
    left_fill_size = fill_size // 2 + (fill_size % 2)
    right_fill_size = (fill_size // 2)

    new_message = "{0}{1} {2} {3}{4}".format(
        left,
        fill_char * left_fill_size,
        message,
        fill_char * right_fill_size,
        right)
    return new_message


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx DummyProgressbar - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class DummyProgressbar(object):  # pragma: no cover
    """Dummy progress bar that don't really do anything.

    The idea is that it can be used in place of the
    :class:`ProgressbarText` class, but without actually doing anything.

    See also
    --------
    ProgressbarText
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the DummyProgressbar object.

        This method accepts any argument without errors, but they won't
        matter, since this class does nothing.
        """
        pass

    def progress(self, count):
        """This `progress` method has the same signature from the one in the
        :class:`ProgressbarText` class.

        Nothing happens when this method is called.

        Parameters
        ----------
        count : int
            Ignored
        """
        pass
# xxxxxxxxxx DummyProgressbar - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarTextBase - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# The code here and in some of the derived classes is inspired in the code
# located in
# http://nbviewer.ipython.org/url/github.com/ipython/ipython/raw/master/examples/notebooks/Progress%20Bars.ipynb
class ProgressbarTextBase(object):
    def __init__(self, finalcount, progresschar='*', message='', output=sys.stdout):
        """Initializes the progressbar object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        progresschar : str, optional (default to '*')
            The character used to represent progress.
        message : str, optional
            A message to be shown in the top of the progressbar.
        output : File like object
            Object with a 'write' method, which controls where the
            progress-bar will be printed. By default sys.stdout is used,
            which means that the progress will be printed in the standard
            output.
        """
        self.prog_bar = ""

        self.finalcount = finalcount
        self.progresschar = progresschar
        self._width = 50  # This should be a multiple of 10 and the lower
                          # possible value is 40.

        # By default, self._output points to sys.stdout so I can use the
        # write/flush methods to display the progress bar.
        self._output = output
        self._message = message  # THIS WILL BE IGNORED

        self._initialized = False

        # This will be set to true when the progress reaches 100%. When
        # this is True, any subsequent calls to the `progress` method will
        # be ignored.
        self._finalized = False

        # This variable will store the time when the `start` method was
        # called for the first time (either manually or in the `progress`
        # method. It will be used for tracking the elapsed time.
        self._start_time = 0.0
        # This variable will store the time when the `stop` method was
        # called for the first time (either manually or in the `progress`
        # method. It will be used for tracking the elapsed time.
        self._stop_time = 0.0

    def _get_elapsed_time(self):
        """Get method for the elapsed_time property."""
        from util.misc import pretty_time

        elapsed_time = 0.0
        if self._initialized is True:
            if self._finalized is False:
                elapsed_time = time.time() - self._start_time
            else:
                elapsed_time = self._stop_time - self._start_time
        return pretty_time(elapsed_time)
    elapsed_time = property(_get_elapsed_time)

    def _count_to_percent(self, count):
        """Convert a given count into the equivalent percentage.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount

        Returns
        -------
        percentage : float
            The percentage that `count` is of self.finalcount (between 0 and 100)
        """
        percentage = (count / float(self.finalcount)) * 100.0
        return percentage

    def _set_width(self, value):
        """Set method for the width property."""
        # If value is not a multiple of 10, width will be set to the
        # largest multiple of 10 which is lower then value.
        if value < 40:
            self._width = 40
        self._width = value - (value % 10)

    def _get_width(self):
        """Get method for the width property."""
        return self._width

    width = property(_get_width, _set_width)

    def _get_percentage_representation(self, percent, central_message='{percent}%', left_side='[', right_side=']'):
        """
        Parameters
        ----------
        percent : float
            The percentage to be represented.
        central_message : str
            A message that will be in the middle of the percentage bar. If
            there is the label '{percent}' in the central_message it will
            be replaced by the percentage. If there is the label
            '{elapsed_time}' in the central_message it will be replaced by
            the elapsed time. Note that this message should be very small,
            since it hides the progresschars.
        left_side : str
            The left side of the bar.
        - `right_side`:

        Returns
        -------
        representation : str
            A string with the representation of the percentage.
        """
        # Remove any fractinonal part
        percent_done = int(percent)
        elapsed_time = self.elapsed_time

        # Calculates how many characters are spent just for the sides.
        sides_length = len(left_side) + len(right_side)
        # The width should be large enough to contain both the left_side
        # and right_side and still have (reasonable) enough space for the
        # characters representing the progress.
        assert(self.width > sides_length + 20)

        # Space that will be used bu the characters representing the
        # progress
        all_full = self.width - sides_length

        # Calculates how many characters will be used to represent the
        # `percend_done` value
        num_hashes = int((percent_done / 100.0) * all_full)

        prog_bar = left_side + self.progresschar * num_hashes + ' ' * (all_full - num_hashes) + right_side

        # Replace the center of prog_bar with the message
        central_message = central_message.format(percent=percent_done, elapsed_time=elapsed_time)
        pct_place = (len(prog_bar) // 2) - (len(str(central_message)) // 2)
        prog_bar = prog_bar[0:pct_place] + central_message + prog_bar[pct_place + len(central_message):]

        return prog_bar

    def _perform_initialization(self):
        """
        Perform the initializations.

        This method should be derived in sub-classes if any initialization
        code should be run.
        """
        pass

    def start(self):
        """
        Start the progressbar.

        This method should be called just before the progressbar is used
        for the first time. Among possible other things, it will store the
        current time so that the elapsed time can be tracked.
        """
        if self._initialized is False:
            self._start_time = time.time()
            self._perform_initialization()
            self._initialized = True

    def stop(self):
        """
        Stop the progressbar.

        This method is automatically called in the `progress` method when
        the progress reaches 100%. If manually called, any subsequent call
        to the `progress` method will be ignored.
        """
        if self._finalized is False:
            self._stop_time = time.time()

            # Print an empty line after the last iteration to be consistent
            # with the ProgressbarText class
            self._output.write("\n")

            # When progress reaches 100% we set the internal variable
            # to True so that any subsequent calls to the `progress`
            # method will be ignored.
            self._finalized = True

    def progress(self, count):
        """
        Updates the progress bar.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount

        Notes
        -----
        How the progressbar is actually represented depends on the
        `_update_iteration` method, which is left to be implemented in a
        subclass.
        """
        if self._finalized is False:
            # Start the progressbar. This only have an effect the first
            # time it is called. It initializes the elapsed time tracking
            # and call the _perform_initialization method to perform any
            # initialization.
            self.start()

            # Sanity check. If count is greater then self.finalcount we set
            # it to self.finalcount
            if count > self.finalcount:
                count = self.finalcount

            # Update the prog_bar variable
            self._update_iteration(count)

            # We simple change the cursor to the beginning of the line and
            # write the string representation of the prog_bar variable.
            self._output.write('\r')
            self._output.write(str(self.prog_bar))

            # If count is equal to self.finalcount we have reached 100%. In
            # that case, we also write a final newline character.
            if count == self.finalcount:
                self.stop()

            # Flush everything to guarantee that at this point everything is
            # written to the output.
            self._output.flush()

    def __str__(self):
        return str(self.prog_bar)

    def _update_iteration(self, count):
        """
        Update the self.prog_bar member variable according with the new
        `count`.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount
        """
        raise NotImplemented("Implement this method in a subclass")
# xxxxxxxxxx ProgressbarTextBase - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarText - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# http://code.activestate.com/recipes/299207-console-text-progress-indicator-class/
# CLASS NAME: ProgressbarText
#
# Original Author of the ProgressbarText class:
# Larry Bates (lbates@syscononline.com)
# Written: 12/09/2002
#
# Modified by Darlan Cavalcante Moreira in 10/18/2011
# Released under: GNU GENERAL PUBLIC LICENSE
class ProgressbarText(ProgressbarTextBase):
    """Class that prints a representation of the current progress as
    text.

    You can set the final count for the progressbar, the character that
    will be printed to represent progress and a small message indicating
    what the progress is related to.

    In order to use this class, create an object outsize a loop and inside
    the loop call the `progress` function with the number corresponding to
    the progress (between 0 and finalcount). Each time the `progress`
    function is called a number of characters will be printed to show the
    progress. Note that the number of printed characters correspond is
    equivalent to the progress minus what was already printed.

    See also
    --------
    DummyProgressbar

    Examples
    --------
    >> pb = ProgressbarText(100, 'o', "Hello Simulation")
    >> pb.progress(20)
    ---------------- Hello Simulation ---------------1
        1    2    3    4    5    6    7    8    9    0
    ----0----0----0----0----0----0----0----0----0----0
    oooooooooo
    >> pb.progress(40)
    oooooooooooooooooooo
    >> pb.progress(50)
    ooooooooooooooooooooooooo
    >> pb.progress(100)
    oooooooooooooooooooooooooooooooooooooooooooooooooo
    """
    def __init__(self, finalcount, progresschar='*', message='', output=sys.stdout):
        """Initializes the ProgressbarText object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        progresschar : str, optional (default to '*')
            The character used to represent progress.
        message : str, optional
            A message to be shown in the top of the progressbar.
        output : File like object
            Object with a 'write' method, which controls where the
            progress-bar will be printed. By default sys.stdout is used,
            which means that the progress will be printed in the standard
            output.
        """
        ProgressbarTextBase.__init__(self, finalcount, progresschar, message, output)

        self.progresscharcount = 0  # stores how many characters where
                                    # already printed in a previous call to
                                    # the `progress` function

    def __get_initialization_bartitle(self):
        """
        Get the progressbar title.

        The title is the first line of the progressbar initialization
        message.

        The bar title is something like the line below

        ------------------- % Progress ------------------1\n

        when there is no message.

        Returns
        -------
        bartitle : str
            The bar title.

        Notes
        -----
        This method is only a helper method called in the
        `_write_initialization` method.
        """
        if(len(self._message) != 0):
            message = self._message
        else:
            message = '% Progress'

        bartitle = center_message(message, self.width + 1, '-', '', '1\n')
        return bartitle

    def __get_initialization_markers(self):
        """
        The initialization markers 'mark' the current progress in the
        progressbar that will apear below it.

        Returns
        -------
        (marker_line1, marker_line2) : a tuple of two strings
            A tuple containing the 'two lines' with the progress markers.

        Notes
        -----
        This method is only a helper method called in the
        `_write_initialization` method.
        """
        steps = self.width / 10  # This division must be exact

        values1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        values2 = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

        line1sep = ' ' * (steps - 1)
        line1 = '{0}{1}\n'.format(line1sep, line1sep.join(values1))

        line2sep = '-' * (steps - 1)
        line2 = '{0}{1}\n'.format(line2sep, line2sep.join(values2))

        return (line1, line2)

    def _perform_initialization(self):
        bartitle = self.__get_initialization_bartitle()
        marker_line1, marker_line2 = self.__get_initialization_markers()

        self._output.write(bartitle)
        self._output.write(marker_line1)
        self._output.write(marker_line2)

    def _update_iteration(self, count):
        percentage = self._count_to_percent(count)

        # Set the self.prog_bar variable simply as a string containing as
        # many self.progresschar characters as necessary.
        self.prog_bar = self._get_percentage_representation(percentage, left_side='', right_side='', central_message='')

    # def _write_progress(self, count):
    #     # Make sure I don't try to go off the end (e.g. >100%)
    #     count = min(count, self.finalcount)

    #     if self.finalcount:
    #         percentcomplete = int(round(100 * count / self.finalcount))
    #         if percentcomplete < 1:
    #             percentcomplete = 1
    #     else:
    #         # If we are here, that means self.finalcount is zero and thus
    #         # we are already done. Just set percentcomplete to 100
    #         percentcomplete = 100

    #     # The progresscharcount variable will give us how many characters
    #     # we need to represent the correct percentage of completeness.
    #     progresscharcount = int(percentcomplete * self.width / 100)
    #     if progresscharcount > self.progresscharcount:
    #         # The self.progresscharcount stores how many characters where
    #         # already printed in previous calls to the `progress`
    #         # function. Therefore, we only need to print the remaining
    #         # characters until we reach `progresscharcount`.
    #         for i in range(self.progresscharcount, progresscharcount):  # pylint:disable=W0612
    #             self._output.write(self.progresschar)
    #             self._output.flush()
    #         # Update self.progresscharcount
    #         self.progresscharcount = progresscharcount

    #     # If we completed the bar, print a newline
    #     if percentcomplete == 100:
    #         self._output.write("\n")
# xxxxxxxxxx ProgressbarText - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarText2 - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarText2(ProgressbarTextBase):
    def __init__(self, finalcount, progresschar='*', message='', output=sys.stdout):
        """
        Initializes the progressbar object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        progresschar : str, optional (default to '*')
            The character used to represent progress.
        message : str, optional
            A message to be shown in the right of the progressbar. If this
            message contains "{elapsed_time}" it will be replaced by the
            elapsed time.
        output : File like object
            Object with a 'write' method, which controls where the
            progress-bar will be printed. By default sys.stdout is used,
            which means that the progress will be printed in the standard
            output.
        """
        ProgressbarTextBase.__init__(self, finalcount, progresschar, message, output)

    def _update_iteration(self, count):
        """
        Update the self.prog_bar member variable according with the new
        `count`.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount
        """
        # Update the self.prog_bar variable with the current count, but
        # without the message (if there is one)
        self._update_prog_bar(self._count_to_percent(count))

        # Append the message to the self.prog_bar variable if there is one
        # (or a default message if there is no message set)..
        if(len(self._message) != 0):
            message = self._message.format(elapsed_time=self.elapsed_time)
            self.prog_bar += "  {0}".format(message)
        else:
            self.prog_bar += '  %d of %d complete' % (count, self.finalcount)

    def _update_prog_bar(self, count):
        self.prog_bar = self._get_percentage_representation(
            count,
            central_message='{percent}%',
            left_side='[',
            right_side=']')
# xxxxxxxxxx ProgressbarText2 - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarText3 - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarText3(ProgressbarTextBase):
    def __init__(self, finalcount, progresschar='*', message='', output=sys.stdout):
        """Initializes the progressbar object.

        Parameters
        ----------
        finalcount : int
            The total amount that corresponds to 100%. Each time the
            progress method is called with a number that number is added
            with the current amount in the progressbar. When the amount
            becomes equal to `finalcount` the bar will be 100% complete.
        progresschar : str, optional (default to '*')
            The character used to represent progress.
        message : str, optional
            A message to be shown in the progressbar.
        output : File like object
            Object with a 'write' method, which controls where the
            progress-bar will be printed. By default sys.stdout is used,
            which means that the progress will be printed in the standard
            output.
        """
        ProgressbarTextBase.__init__(self, finalcount, progresschar, message, output)

    def _update_iteration(self, count):
        """
        Update the self.prog_bar member variable according with the new
        `count`.

        Parameters
        ----------
        count : int
            The current count to be represented in the progressbar. The
            progressbar represents this count as a percent value of
            self.finalcount
        """
        full_count = "{0}/{1}".format(count, self.finalcount)

        if len(self._message) != 0:
            self.prog_bar = center_message(
                "{0} {1}".format(self._message, full_count),
                length=self.width,
                fill_char=self.progresschar)
        else:
            self.prog_bar = center_message(full_count,
                                           length=self.width,
                                           fill_char=self.progresschar)

# xxxxxxxxxx ProgressbarText3 - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarServerBase xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarDistributedServerBase(object):
    """
    Base class for progressbars for distributed computations.

    In order to track the progress of distributed computations two classes
    are required, one that acts as a central point and is responsible to
    actually show the progress (the server), and other class that acts as a
    proxy (the client) and is responsible to sending the current progress
    to the server. There will be one object of the "server class" and one
    or more objects of the "client class", each one tracking the progress
    of one of the distributed computations.

    This class is a base class for the "server part", while the
    :class:`ProgressbarDistributedClientBase` class is a base class for the
    "client part".

    For a full implementation, see the :class:`ProgressbarMultiProcessServer`
    and :class:`ProgressbarMultiProcessClient` classes.
    """

    def __init__(self,
                 progresschar='*',
                 message='',
                 sleep_time=1,
                 filename=None):
        """
        Initializes the ProgressbarDistributedServerBase object.

        Parameters
        ----------
        progresschar : str
            Character used in the progressbar.
        message : str
            Message writen in the progressbar.
        sleep_time : float
            Time between progressbar updates (in seconds).
        filename : str
            If filename is None (default) then progress will be output to
            sys.stdout. If it is not None then the progress will be output
            to a file with name `filename`. This is usually useful for
            debugging and testing purposes.
        """
        self._progresschar = progresschar
        self._message = message

        self._sleep_time = sleep_time
        self._last_id = -1

        self._filename = filename

        self._manager = multiprocessing.Manager()
        self._client_data_list = self._manager.list()  # pylint: disable=E1101

        # total_final_count will be updated each time the register_*
        # function is called.
        #
        # Note that we use a Manager.Value object to store the value
        # instead of using a simple integer because we want modifications
        # to this value to be seem by the other updating process even after
        # start_updater has been called if we are still in the
        # 'start_delay' time.
        self._total_final_count = self._manager.Value('L', 0)

        # self._update_process will store the process responsible to update
        # the progressbar. It will be created in the first time the
        # start_updater method is called.
        self._update_process = None

        # The event will be set when the process updating the progressbar
        # is running and unset (clear) when it is stopped.
        self.running = multiprocessing.Event()  # Starts unset. Is is set
                                                # in the _update_progress
                                                # function

        # Each time the start_updater method is called this variable is
        # increased by one and each time the stop_updater method is called
        # it is decreased by one. A started update process will only be
        # stopped in the stop_updater method if this variables reaches 0.
        # This control is useful so that we can share the same progressbar
        # for multiple SimulationRunner objects.
        self._start_updater_count = 0

        # # Used for time tracking
        # self._tic = multiprocessing.Value('f', 0.0)
        # self._toc = multiprocessing.Value('f', 0.0)

    def _get_total_final_count(self):
        """Get method for the total_final_count property."""
        return self._total_final_count.get()
    total_final_count = property(_get_total_final_count)

    def _update_client_data_list(self):
        """
        This method process the communication between the client and the
        server.

        It should gather the information sent by the clients (proxy
        progressbars) and update the member variable self._client_data_list
        accordingly, which will then be automatically represented in the
        progressbar.
        """
        # Implement this method in a derived class.
        pass  # pragma: no cover

    def register_client_and_get_proxy_progressbar(self, total_count):
        """
        Register a new "client" for the progressbar and returns a new proxy
        progressbar that the client can use to update its progress by
        calling the `progress` method of this proxy progressbar.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% for function.

        Returns
        -------
        obj : Object of a class derived from ProgressbarDistributedClientBase
            The proxy progressbar.
        """
        # Implement this method in a derived class
        #
        # Note: The first thing the implementation of this method in a
        # derived class must do is call the _register_client method to
        # register the new client and get its client_id, like the example
        # below
        # >>> client_id = self._register_client(total_count)
        #
        # After that the implementation of
        # register_client_and_get_proxy_progressbar can create the
        # corresponding proxy progressbar passing the client_id and any
        # other required data.
        pass  # pragma: no cover

    def _register_client(self, total_count):
        """
        Register a new "client" for the progressbar and return its `client_id`.

        These returned values must be passed to the corresponding proxy
        progressbar.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% progress for the
            function.

        Returns
        -------
        (client_id, client_data_list) : tuple
            A tuple with the client_id and the client_data_list. The
            function whose process is tracked by the
            ProgressbarMultiProcessServer must update the element
            `client_id` of the list `client_data_list` with the current
            count.
        """
        # Set self._total_final_count to the value currently stored plus
        # total_count.
        # Remember that the self._total_final_count variable is actually a
        # proxy to the true value (that is, self._total_final_count is a
        # multiprocessing.Manager.Value object). That is way we need the
        # two lines below.
        total_final_count = self.total_final_count
        self._total_final_count.set(total_final_count + total_count)

        # Update the last_id
        self._last_id += 1

        # client_id that will be used by the function
        client_id = self._last_id

        self._client_data_list.append(0)
        return client_id

    # This method will be run in a different process. Because of this the
    # coverage program does not see that this method in run in the test code
    # even though we know it is run (otherwise no output would
    # appear). Therefore, we put the "pragma: no cover" line in it
    def _update_progress(self, filename=None, start_delay=0.0):  # pragma: no cover
        """
        Collects the progress from each registered proxy progressbar and
        updates the actual visible progressbar.

        Parameters
        ----------
        filename : str
            Name of a file where the data will be written to. If this is
            None then all progress will be printed in the standard output
            (defaut)
        start_delay : float (defaut is 0.0)
            Delay in seconds before starting the progressbar. During this
            time it is still possible to register new clients and the
            progressbar will only be shown after this delay..
        """
        if start_delay > 0.0:
            time.sleep(start_delay)

        if self.total_final_count == 0:
            import warnings
            warnings.warn('No clients registered in the progressbar')

        if filename is None:
            import sys
            output = sys.stdout
        else:
            output = open(filename, 'w')

        pbar = ProgressbarText2(self.total_final_count,
                                self._progresschar,
                                self._message,
                                output=output)
        count = 0
        while count < self.total_final_count and self.running.is_set():
            time.sleep(self._sleep_time)
            # Gather information from all client proxybars and update the
            # self._client_data_list member variable
            self._update_client_data_list()

            # Calculates the current total count from the
            # self._client_data_list
            count = sum(self._client_data_list)

            # Maybe a new client was registered. In that case
            # self.total_final_count changed and we need to reflect this in
            # the pbar.finalcount variable.
            pbar.finalcount = self.total_final_count

            # Represents the current total count in the progressbars
            pbar.progress(count)

        # If the self.running event was cleared (because the stop_updater
        # method was called) we most likely exited the while loop before
        # the progressbar was full (count is lower then the total final
        # count). If that is the case, let's set the progressbar to full
        # here.
        if count < self.total_final_count:
            pbar.progress(self.total_final_count)

        # It may exit the while loop in two situations: if count reached
        # the maximum allowed value, in which case the progressbar is full,
        # or if the self.running event was cleared in another
        # process. Since in the first case the event is still set, we clear
        # it here to have some consistence (a cleared event will always
        # mean that the progressbar is not running).
        self.running.clear()
        # self._toc.value = time.time()

    def start_updater(self, start_delay=0.0):
        """
        Start the process that updates the progressbar.

        Parameters
        ----------
        start_delay : float
            Delay in seconds before starting the progressbar. During this
            time it is still possible to register new clients and the
            progressbar will only be shown after this delay..

        Notes
        -----
        If this method is called multiple times then the `stop_updater`
        method must be called the same number of times for the updater
        process to actually stop.
        """
        if self.running.is_set() is False:
            # self._update_process stores the process responsible to update
            # the progressbar. It may be finished anytime by calling the
            # stop_updater method. Also, it is set as a daemon process so
            # that we don't get errors if the program closes before the
            # process updating the progressbar ends (because the user
            # forgot to call the stop_updater method).
            self._update_process = multiprocessing.Process(name="ProgressBarUpdater", target=self._update_progress, args=[self._filename, start_delay])
            self._update_process.daemon = True

            self.running.set()
            self._update_process.start()

        self._start_updater_count += 1

    def stop_updater(self, timeout=None):
        """
        Stop the process updating the progressbar.

        You should always call this function in your main process (the same
        that created the progressbar) after joining all the processes that
        update the progressbar. This guarantees that the progressbar
        updated any pending change and exited clearly.

        Parameters
        ----------
        timeout : float
            The timeout to join for the process to stop.

        Notes
        -----
        If the `start_updater` was called multiple times the process will
        only be stopped when `stop_updater` is called the same number of
        times.
        """
        self._start_updater_count -= 1
        if self._start_updater_count == 0:
            self.running.clear()
            # self._toc.value = time.time()
            self._update_process.join(timeout)

    # # TODO: Check if the duration property work correctly
    # @property
    # def duration(self, ):
    #     """Duration of the progress.

    #     Returns
    #     -------
    #     toc_minus_tic : float
    #         The duration passed until the progressbar reaches 100%.
    #     """
    #     # The progressbar is still running, calculate the duration since
    #     # the beginning
    #     if self.running.is_set():
    #         toc = time.time()
    #     else:
    #         toc = self._toc.value

    #     return toc - self._tic.value


class ProgressbarDistributedClientBase(object):
    """
    Proxy progressbar that behaves like a ProgressbarText object, but is
    actually updating a shared (with other clients) progressbar.

    The basic idea is that this proxy progressbar has the "progress" method
    similar to the standard ProgressbarText class. However, when this
    method is called it will update a value that will be read by a "server
    progressbar" object which is responsible to actually show the current
    progress.
    """

    def __init__(self, client_id):
        """
        """
        self.client_id = client_id

    # Implement this method in a derived class
    def progress(self, count):
        """Updates the proxy progress bar.

        Parameters
        ----------
        count : int
            The new amount of progress.

        """
        pass  # pragma: no cover


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarMultiProcessServer - START xxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarMultiProcessServer(ProgressbarDistributedServerBase):
    """Class that prints a representation of the current progress of
    multiple process as text.

    While the :class:`ProgressbarText` class only tracks the progress of a
    single process, the :class:`ProgressbarMultiProcessServer` class can
    track the joint progress of multiple processes. This may be used, for
    instance, when you parallelize some task using the multiprocessing
    module.

    Using the ProgressbarMultiProcessServer class requires a little more work
    than using the ProgressbarText class, as it is described in the
    following:

     1. First you create an object of the ProgressbarMultiProcessServer class
        as usual. However, differently from the ProgressbarText class you
        don't pass the `finalcount` value to the progressbar yet.
     2. After that, for each process to be tracked, call the
        :meth:`register_client_and_get_proxy_progressbar` method passing
        the number equivalent to full progress for **that process**. This
        function returns a "proxy progressbar" that behaves like a regular
        ProgressbarText. Pass that proxy progressbar as an argument to that
        process so that it can call its "progress" method. Each process
        that calls the "progress" method of the received proxy progressbar
        will actually update the progress of the main
        ProgressbarMultiProcessServer object.
     3. Start all the processes and call the start_updater method of
        ProgressbarMultiProcessServer object so that the bar is updated by
        the different processes.
     4. After joining all the process (all work is finished) call the
        stop_updater method of the ProgressbarMultiProcessServer object.

    Examples
    --------

    .. code-block:: python

       import multiprocessing
       # Create a ProgressbarMultiProcessServer object
       pb = ProgressbarMultiProcessServer(message="some message")
       # Creates a proxy progressbar for one process passing the value
       # corresponding to 100% progress for the first process
       proxybar1 = pb.register_client_and_get_proxy_progressbar(60)
       # Creates a proxy progressbar for another process
       proxybar2 = pb.register_client_and_get_proxy_progressbar(80)
       # Create the first process passing the first proxy progressbar as
       # an argument
       p1 = multiprocessing.Process(target=some_function, args=[proxybar1])
       # Creates another process
       p2 = multiprocessing.Process(target=some_function, args=[proxybar2])
       # Start both processes
       p1.start()
       p2.start()
       # Call the start_updater method of the ProgressbarMultiProcessServer
       pb.start_updater()
       # Joint the process and then call the stop_updater method of the
       # ProgressbarMultiProcessServer
       p1.join()
       p2.join()
       pb.stop_updater()

    """

    def __init__(self,
                 progresschar='*',
                 message='',
                 sleep_time=1,
                 filename=None):
        """
        Initializes the ProgressbarMultiProcessServer object.

        Parameters
        ----------
        progresschar : str
            Character used in the progressbar.
        message : str
            Message writen in the progressbar.
        sleep_time : float
            Time between progressbar updates (in seconds).
        filename : str
            If filename is None (default) then progress will be output to
            sys.stdout. If it is not None then the progress will be output
            to a file with name `filename`. This is usually useful for
            debugging and testing purposes.
        """
        ProgressbarDistributedServerBase.__init__(self,
                                            progresschar, message, sleep_time, filename)

    def _update_client_data_list(self):
        """
        This method process the communication between the client and the
        server.
        """
        # Note that since the proxybar (ProgressbarMultiProcessClient class)
        # for multiprocessing will directly modify the
        # self._client_data_list we don't need to implement a
        # _update_client_data_list method here in the
        # ProgressbarMultiProcessServer class.
        pass  # pragma: no cover

    def register_client_and_get_proxy_progressbar(self, total_count):
        """
        Register a new "client" for the progressbar and returns a new proxy
        progressbar that the client can use to update its progress by
        calling the `progress` method of this proxy progressbar.

        The function whose process is tracked by the
        ProgressbarMultiProcessServer must must call the `progress` method of
        the returned ProgressbarMultiProcessClient object with the current
        count. This is a little less intrusive regarding the tracked
        function.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% for function.

        Returns
        -------
        obj : ProgressbarMultiProcessClient object
            The proxy progressbar.
        """
        client_id = self._register_client(total_count)
        return ProgressbarMultiProcessClient(client_id, self._client_data_list)


# Used by the ProgressbarMultiProcessServer class
class ProgressbarMultiProcessClient(ProgressbarDistributedClientBase):
    """
    Proxy progressbar that behaves like a ProgressbarText object,
    but is actually updating a ProgressbarMultiProcessServer progressbar.

    The basic idea is that this proxy progressbar has the "progress" method
    similar to the standard ProgressbarText class. However, when this
    method is called it will update a value that will be read by a
    ProgressbarMultiProcessServer object instead.
    """
    def __init__(self, client_id, client_data_list):
        """Initializes the ProgressbarMultiProcessClient object."""
        ProgressbarDistributedClientBase.__init__(self, client_id)
        self._client_data_list = client_data_list

    def progress(self, count):
        """Updates the proxy progress bar.

        Parameters
        ----------
        count : int
            The new amount of progress.

        """
        self._client_data_list[self.client_id] = count
# xxxxxxxxxx ProgressbarMultiProcessServer - END xxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarZMQServer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarZMQServer2(ProgressbarDistributedServerBase):
    """
    Distributed "server" progressbar using ZMQ sockets.

    In order to track the progress of distributed computations two classes
    are required, one that acts as a central point and is responsible to
    actually show the progress (the server), and other class that acts as a
    proxy (the client) and is responsible to sending the current progress
    to the server. There will be one object of the "server class" and one
    or more objects of the "client class", each one tracking the progress
    of one of the distributed computations.

    This class acts like the server. It creates a ZMQ socket which expects
    (string) messages in the form "client_id:current_progress", where the
    client_id is the ID of one client progressbar previously registered
    with the register_client_and_get_proxy_progressbar method while the
    current_progress is a "number" with the current progress of that
    client.

    Note that the client proxybar for this class is implemented in the
    ProgressbarZMQClient class.

    Parameters
    ----------
    progresschar : str
        Character used in the progressbar.
    message : str
        Message writen in the progressbar.
    sleep_time : float
        Time between progressbar updates (in seconds).
    filename : str
        If filename is None (default) then progress will be output to
        sys.stdout. If it is not None then the progress will be output to a
        file with name `filename`. This is usually useful for debugging and
        testing purposes.
    ip : string
        An string representing the address of the server socket.
        Ex: '192.168.0.117', 'localhost', etc.
    port : int
        The port to bind the socket.
    """

    def __init__(self,
                 progresschar='*',
                 message='',
                 sleep_time=1,
                 filename=None,
                 ip='localhost',
                 port=7396):
        """
        Initializes the ProgressbarDistributedServerBase object.

        Parameters
        ----------
        progresschar : str
            Character used in the progressbar.
        message : str
            Message writen in the progressbar.
        sleep_time : float
            Time between progressbar updates (in seconds).
        filename : str
            If filename is None (default) then progress will be output to
            sys.stdout. If it is not None then the progress will be output
            to a file with name `filename`. This is usually useful for
            debugging and testing purposes.
        ip : string
            An string representing the address of the server socket.
            Ex: '192.168.0.117', 'localhost', etc.
        port : int
            The port to bind the socket.
        """
        ProgressbarDistributedServerBase.__init__(self,
                                            progresschar, message, sleep_time, filename)

        # Create a Multiprocessing namespace
        self._ns = self._manager.Namespace()

        # We store the IP and port of the socket in the Namespace, since
        # the socket will be created in a different process
        self._ns.ip = ip
        self._ns.port = port

    def _get_ip(self):
        """Get method for the ip property."""
        return self._ns.ip
    ip = property(_get_ip)

    def _get_port(self):
        """Get method for the port property."""
        return self._ns.port
    port = property(_get_port)

    def register_client_and_get_proxy_progressbar(self, total_count):
        client_id = self._register_client(total_count)
        proxybar = ProgressbarZMQClient(client_id, self.ip, self.port)
        return proxybar

    def _update_progress(self, filename=None, start_delay=0.0):
        """
        Collects the progress from each registered proxy progressbar and
        updates the actual visible progressbar.

        Parameters
        ----------
        filename : str
            Name of a file where the data will be written to. If this is
            None then all progress will be printed in the standard output
            (defaut)
        start_delay : float (default is 0.0)
            Delay in seconds before starting the progressbar. During this
            time it is still possible to register new clients and the
            progressbar will only be shown after this delay..

        Notes
        -----
        We re-implement it here only to create the ZMQ socket. After that we
        call the base class implementation of _update_progress method. Note
        that the _update_progress method in the base class calls the
        _update_client_data_list and we indeed re-implement this method in
        this class and use the socket created here in that implementation.
        """
        # First we create the context and the socket. Then we bind the
        # socket to the respective ip:port.
        self._zmq_context = zmq.Context()
        self._zmq_pull_socket = self._zmq_context.socket(zmq.PULL)
        self._zmq_pull_socket.bind("tcp://*:%s" % self.port)
        ProgressbarDistributedServerBase._update_progress(self, filename)

    def _update_client_data_list(self):
        """
        This method process the communication between the client and the
        server.

        This method will read the received messages in the socket which
        were sent by the clients (ProgressbarZMQClient objects) and update
        self._client_data_list variable accordingly. The messages are in
        the form "client_id:current_progress", which is parsed by the
        _parse_progress_message method.

        Notes
        -----
        This method is called inside a loop in the _update_progress method.
        """

        pending_mensages = True
        while pending_mensages is True and self.running.is_set():
            try:
                # Try to read a message. If this fail we will get a
                # zmq.ZMQError exception and then pending_mensages will be
                # set to False so that we exit the while loop.
                message = self._zmq_pull_socket.recv_string(flags=zmq.NOBLOCK)

                # If we are here that means that a new message was
                # successfully received from the client.  Let's call the
                # _parse_progress_message method to parse the message and
                # update the self._client_data_list member variable.
                self._parse_progress_message(message)
            except zmq.ZMQError:
                pending_mensages = False

    def _parse_progress_message(self, message):
        """
        Parse the message sent from the client proxy progressbars.

        The messages sent from the proxy progressbars are in the form
        'client_id:current_count'. We need to set the element of index
        "client_id" in self._client_data_list to the value of
        "current_count". This method will simply parse the message and
        perform this operation.

        Parameters
        ----------
        message : str
            A string in the form 'client_id:current_count'.
        """
        client_id, current_count = map(int, message.split(":"))
        self._client_data_list[client_id] = current_count


class ProgressbarZMQServer(object):
    """Progressbar using ZMQ sockets.
    """
    def __init__(self,
                 progresschar='*',
                 message='',
                 sleep_time=2,
                 filename=None):
        """
        Initializes the ProgressbarMultiProcessServer object.

        Parameters
        ----------
        progresschar : str
            Character used in the progressbar.
        message : str
            Message writen in the progressbar.
        sleep_time : float
            Time between progressbar updates (in seconds).
        filename : str
            If filename is None (default) then progress will be output to
            sys.stdout. If it is not None then the progress will be output
            to a file with name `filename`. This is usually useful for
            debugging and testing purposes.
        """
        # total_final_count will be updated each time the register_*
        # function is called
        self._total_final_count = 0
        self._progresschar = progresschar
        self._message = message

        self._sleep_time = sleep_time
        self._last_id = -1

        self._filename = filename

        # Each time we register a client we add a value here for that
        # client. When the client updates its value we update the
        # corresponding value here. In order to obtain the total amount
        # already run all we need to do is to is to sum all the values
        # here.
        self._client_data_list = []

        self.running = False

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Server information
        self._ip = 'localhost'
        self._port = None

        # Bind the server socket
        self._zmq_context = zmq.Context()
        self._zmq_pull_socket = self._zmq_context.socket(zmq.PULL)
        if self._port is None:
            self._port = self._zmq_pull_socket.bind_to_random_port("tcp://*")
        else:
            self._zmq_pull_socket.bind("tcp://*:%s" % self._port)

    def _register_client(self, total_count):
        """
        Register a new client for the progressbar and return its `client_id` as
        well as the IP and PORT that the client should use to update its
        progress.

        These returned values must be passed to the corresponding proxy
        progressbar.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% progress for the
            function.

        Returns
        -------
        (client_id, IP, PORT) : tuple of 3 integers
            A tuple with the client_id, the IP and the PORT that will be
            necessary to create the proxy progressbar (see the
            :class:`ProgressbarZMQClient` class)
        """
        self._total_final_count += total_count

        # Update the last_id
        self._last_id += 1

        # client_id that will be used by the function
        client_id = self._last_id

        self._client_data_list.append(0)
        return (client_id, self._ip, self._port)

    def register_client_and_get_proxy_progressbar(self, total_count):
        """
        Creates a new proxy progressbar that can be used to update the main
        progressbar.

        The returned progressbar should be passed to the processing that
        wants to update the progress.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% for function.

        Returns
        -------
        obj : ProgressbarZMQClient object
            The proxy progressbar.
        """
        return ProgressbarZMQClient(*self._register_client(total_count))

    def _parse_progress_message(self, message):
        """
        Parse the message sent from the client proxy progressbars.

        The messages sent from the proxy progressbars are in the form
        'client_id:current_count'. We need to set the element of index
        "client_id" in self._client_data_list to the value of
        "current_count". This method will simply parse the message and
        perform this operation.

        Parameters
        ----------
        message : str
            A string in the form 'client_id:current_count'.
        """
        client_id, current_count = map(int, message.split(":"))
        self._client_data_list[client_id] = current_count

    def start_updater(self):
        """
        Start the updating of the progressbar that updates the progressbar.

        This will create the socket that receives the progress from the
        clients and update the actual progressbar.
        """
        if self.total_final_count == 0:
            import warnings
            warnings.warn('No clients registered in the progressbar')

        if self._filename is None:
            import sys
            output = sys.stdout
        else:
            output = open(filename, 'w')

        pbar = ProgressbarText(self.total_final_count,
                               self._progresschar,
                               self._message,
                               output=output)

        self.running = True
        count = 0
        while count < self.total_final_count and self.running is True:
            try:
                # Try to receive something in the socket.
                message = self._zmq_pull_socket.recv_string(flags=zmq.NOBLOCK)
                # This will update self._client_data_list
                self._parse_progress_message(message)

                count = sum(self._client_data_list)
                pbar.progress(count)
                # and print the received value
            except zmq.ZMQError:
                # If we could not receive anything in the socket it will
                # trown the ZMQError exception. In that case we sleep for
                # "self._sleep_time" seconds.
                time.sleep(self._sleep_time)

        self.running = False

    def stop_updater(self, timeout=None):
        """Stop the process updating the progressbar.

        You should always call this function in your main process (the same
        that created the progressbar) after joining all the processes that
        update the progressbar. This guarantees that the progressbar
        updated any pending change and exited clearly.

        """
        pass


# Used by the ProgressbarZMQServer class
class ProgressbarZMQClient(object):
    """
    Proxy progressbar that behaves like a ProgressbarText object,
    but is actually updating a ProgressbarZMQServer progressbar.

    The basic idea is that this proxy progressbar has the "progress" method
    similar to the standard ProgressbarText class. However, when this
    method is called it will update a value that will be read by a
    ProgressbarZMQServer object instead.
    """
    def __init__(self, client_id, ip, port):
        """Initializes the ProgressbarZMQClient object."""
        # We import zmq here inside the class to avoid the whole module not
        # working if zmq is not available. That means that we will only get
        # the import error when zmq is not available if we actually try to
        # instantiate ProgressbarZMQServer.
        self.client_id = client_id
        self.ip = ip
        self.port = port

        # Function that will be called to update the progress. This
        # variable is initially set to the "_connect_and_update_progress"
        # method that will create the socket, connect it to the main
        # progressbar and finally set "_progress_func" to the "_progress"
        # method that will actually update the progress.
        self._progress_func = ProgressbarZMQClient._connect_and_update_progress

        # ZMQ Variables: These variables will be set the first time the
        # progress method is called.
        self._zmq_context = None
        self._zmq_push_socket = None

    def progress(self, count):
        """Updates the proxy progress bar.

        Parameters
        ----------
        count : int
            The new amount of progress.

        """
        self._progress_func(self, count)

    def __call__(self, count):
        """
        Updates the proxy progress bar.

        This method is the same as the :meth:`progress`. It is define so
        that a ProgressbarZMQClient object can behave like a function.
        """
        self._progress_func(self, count)

    def _progress(self, count):
        """

        Parameters
        ----------
        count : int
        """
        # The mensage is a string composed of the client ID and the current
        # count
        message = "{0}:{1}".format(self.client_id, count)
        self._zmq_push_socket.send_string(message, flags=zmq.NOBLOCK)

    def _connect_and_update_progress(self, count):
        """
        Creates the "push socket", connects it to the socket of the main
        progressbar and then updates the progress.

        This function will be called only in the first time the "progress"
        method is called. Subsequent calls to "progress" will actually
        calls the "_progress" method.

        Parameters
        ----------
        count : int
            The new amount of progress.
        """
        self._zmq_context = zmq.Context()
        self._zmq_push_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_push_socket.connect("tcp://{0}:{1}".format(self.ip, self.port))

        # The default LINGER value for a ZMQ socket is -1, which means
        # "wait forever". That means that if the message was not received
        # by the server (the main progressbar) the process with the
        # push_socket will hang. Since we don't want that, we set the
        # LINGER option to 0 so that it does not wait for the message to be
        # received.
        self._zmq_push_socket.setsockopt(zmq.LINGER, 0)

        self._progress_func = ProgressbarZMQClient._progress
        self._progress_func(self, count)
# xxxxxxxxxx ProgressbarZMQServer - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
