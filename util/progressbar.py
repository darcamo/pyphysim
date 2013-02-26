#!/usr/bin/env python
# -*- coding: utf-8 -*-

# http://code.activestate.com/recipes/299207-console-text-progress-indicator-class/
# CLASS NAME: ProgressbarText
#
# Original Author of the ProgressbarText class:
# Larry Bates (lbates@syscononline.com)
# Written: 12/09/2002
#
# Modified by Darlan Cavalcante Moreira in 10/18/2011
# Released under: GNU GENERAL PUBLIC LICENSE

"""Implement classes to represent the progress of a task.

Use the ProgressbarText class for tasks that do not use multiprocessing,
and the ProgressbarMultiProcessText class for tasks using multiprocessing.

Basically, the task code must call the "progress" function to update the
progress bar and pass a number equivalent to the increment in the progress
since the last call. The progressbar must know the maximum value equivalent
to all the progress, which is passed during object creator for
ProgressbarText class.

The ProgressbarMultiProcessText is similar to ProgressbarText class,
accounts for the progress of multiple processes. For each process you need
to call the register_function_and_get_proxy_progressbar to get a proxy
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

__all__ = ['DummyProgressbar', 'ProgressbarText', 'ProgressbarText2', 'ProgressbarMultiProcessText']


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

    def __init__(self, ):
        """Initializes the DummyProgressbar object."""
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
# xxxxxxxxxxxxxxx ProgressbarText - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarText(object):
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
    >>> pb = ProgressbarText(100, 'o', "Hello Simulation")
    ---------------- Hello Simulation ---------------1
        1    2    3    4    5    6    7    8    9    0
    ----0----0----0----0----0----0----0----0----0----0
    >>> pb.progress(20)
    oooooooooo
    >>> pb.progress(40)
    oooooooooo
    >>> pb.progress(50)
    ooooo
    >>> pb.progress(100)
    ooooooooooooooooooooooooo
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
        """
        self.finalcount = finalcount
        self.blockcount = 0  # stores how many characters where already
                             # printed in a previous call to the `progress`
                             # function
        self.block = progresschar  # The character printed to indicate progress
        #
        # By default, self.output points to sys.stdout so I can use the
        # write/flush methods to display the progress bar.
        self.output = output
        #
        # If the final count is zero, don't start the progress gauge
        #
        if not self.finalcount:
            return
        if(len(message) != 0):
            bartitle = '{0}\n'.format(ProgressbarText.center_message(message, 50, '-', '', '1'))
        else:
            bartitle = '\n------------------ % Progress -------------------1\n'

        self.output.write(bartitle)
        self.output.write('    1    2    3    4    5    6    7    8    9    0\n')
        self.output.write('----0----0----0----0----0----0----0----0----0----0\n')
        return

    def progress(self, count):
        """Updates the progress bar.

        The value of `count` will be added to the current amount and a
        number of characters used to represent progress will be printed.

        Parameters
        ----------
        count : int
            The new amount of progress. The actual percentage of progress
            is equal to count/finalcount.

        """
        #
        # Make sure I don't try to go off the end (e.g. >100%)
        #
        count = min(count, self.finalcount)
        #
        # If finalcount is zero, I'm done
        #
        if self.finalcount:
            percentcomplete = int(round(100 * count / self.finalcount))
            if percentcomplete < 1:
                percentcomplete = 1
        else:
            percentcomplete = 100

        # Divide percentcomplete by two, since we use 50 characters for the
        # full bar. Therefore, the blockcount variable will give us how
        # many characters we need to write to represent the correct
        # percentage of completeness.
        blockcount = int(percentcomplete / 2)
        if blockcount > self.blockcount:
            # The self.blockcount stores how many characters where already
            # printed in a previous call to the `progress`
            # function. Therefore, we only need to print the remaining
            # characters until we reach `blockcount`.
            for i in range(self.blockcount, blockcount):  # pylint:disable=W0612
                self.output.write(self.block)
                self.output.flush()
            # Update self.blockcount
            self.blockcount = blockcount

        # If we completed the bar, print a newline
        if percentcomplete == 100:
            self.output.write("\n")

    @staticmethod
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
        >>> print(ProgressbarText.center_message("Hello World", 50, '-', 'Left', 'Right'))
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
# xxxxxxxxxx ProgressbarText - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarText2 - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# The original Code is in
# http://nbviewer.ipython.org/url/github.com/ipython/ipython/raw/master/examples/notebooks/Progress%20Bars.ipynb
# but it was modified to make it more similar to the ProgressbarText class
class ProgressbarText2:
    def __init__(self, finalcount, progresschar='*', message=''):
        self.finalcount = finalcount
        self.prog_bar = '[]'
        self.progresschar = progresschar
        self.width = 50
        self._update_amount(0)
        self._message = message

    def progress(self, iter):
        self._update_iteration(iter)
        print('\r', self, end='')
        sys.stdout.flush()

    def _update_iteration(self, elapsed_iter):
        # Note that self._update_amount will change self.prog_bar
        self._update_amount((elapsed_iter / float(self.finalcount)) * 100.0)

        if(len(self._message) != 0):
            self.prog_bar += "  {0}".format(self._message)
        else:
            self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.finalcount)

    def _update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.progresschar * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

# xxxxxxxxxx ProgressbarText2 - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarMultiProcessText - START xxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarMultiProcessText(object):
    """Class that prints a representation of the current progress of
    multiple process as text.

    While the :class:`ProgressbarText` class only tracks the progress of a
    single process, the :class:`ProgressbarMultiProcessText` class can
    track the joint progress of multiple processes. This may be used, for
    instance, when you parallelize some task using the multiprocessing
    module.

    Using the ProgressbarMultiProcessText class requires a little more work
    than using the ProgressbarText class, as it is described in the
    following:

     1. First you create an object of the ProgressbarMultiProcessText class
        as usual. However, differently from the ProgressbarText class you
        don't pass the `finalcount` value to the progressbar yet.
     2. After that, for each process to be tracked, call the
        :meth:`register_function_and_get_proxy_progressbar` method passing
        the number equivalent to full progress for **that process**. This
        function returns a "proxy progressbar" that behaves like a regular
        ProgressbarText. Pass that proxy progressbar as an argument to that
        process so that it can call its "progress" method. Each process
        that calls the "progress" method of the received proxy progressbar
        will actually update the progress of the main
        ProgressbarMultiProcessText object.
     3. Start all the processes and call the start_updater method of
        ProgressbarMultiProcessText object so that the bar is updated by
        the different processes.
     4. After joining all the process (all work is finished) call the
        stop_updater method of the ProgressbarMultiProcessText object.

    Examples
    --------

    .. code-block:: python

       import multiprocessing
       # Create a ProgressbarMultiProcessText object
       pb = ProgressbarMultiProcessText(message="some message")
       # Creates a proxy progressbar for one process passing the value
       # corresponding to 100% progress for the first process
       proxybar1 = pb.register_function_and_get_proxy_progressbar(60)
       # Creates a proxy progressbar for another process
       proxybar2 = pb.register_function_and_get_proxy_progressbar(80)
       # Create the first process passing the first proxy progressbar as
       # an argument
       p1 = multiprocessing.Process(target=some_function, args=[proxybar1])
       # Creates another process
       p2 = multiprocessing.Process(target=some_function, args=[proxybar2])
       # Start both processes
       p1.start()
       p2.start()
       # Call the start_updater method of the ProgressbarMultiProcessText
       pb.start_updater()
       # Joint the process and then call the stop_updater method of the
       # ProgressbarMultiProcessText
       p1.join()
       p2.join()
       pb.stop_updater()

    """

    def __init__(self,
                 progresschar='*',
                 message='',
                 sleep_time=1):
        """Initializes the ProgressbarMultiProcessText object.

        Parameters
        ----------
        progresschar : str
            Character used in the progressbar.
        message : str
            Message writen in the progressbar.
        sleep_time : float
            Time between progressbar updates (in seconds).

        """
        # total_final_count will be updated each time the register_*
        # function is called
        self._total_final_count = 0
        self._progresschar = progresschar
        self._message = message

        self._manager = multiprocessing.Manager()
        self._process_data_list = self._manager.list()  # pylint: disable=E1101

        self._sleep_time = sleep_time
        self._last_id = -1

        # Process responsible to update the progressbar. It will be started
        # by the start_updater method and it may be finished anytime by
        # calling the finish_updater function. Also, it is set as a daemon
        # process so that we don't get errors if the program closes before
        # the process updating the progressbar ends (because the user
        # forgot to call the finish_updater method).
        self._update_process = multiprocessing.Process(target=self._update_progress)
        self._update_process.daemon = True

        # The event will be set when the process updating the progressbar
        # is running and unset (clear) when it is stopped.
        self.running = multiprocessing.Event()  # Starts unset. Is is set
                                                # in the _update_progress
                                                # function

        # Used for time tracking
        self._tic = multiprocessing.Value('f', 0.0)
        self._toc = multiprocessing.Value('f', 0.0)

    def register_function(self, total_count):
        """Return the `process_id` and a "process_data_list". These must be
        passed as arguments to the function that will run in another
        process.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% progress for the
            function.

        Returns
        -------
        (process_id, process_data_list) : tuple
            A tuple with the process_id and the process_data_list. The
            function whose process is tracked by the
            ProgressbarMultiProcessText must update the element
            `process_id` of the list `process_data_list` with the current
            count.

        """
        # update self._total_final_count
        self._total_final_count += total_count

        # Update the last_id
        self._last_id += 1

        # process_id that will be used by the function
        process_id = self._last_id

        self._process_data_list.append(0)
        return (process_id, self._process_data_list)

    def register_function_and_get_proxy_progressbar(self, total_count):
        """Similar to the `register_function` method, but returns a
        ProgressbarMultiProcessProxy object.

        The function whose process is tracked by the
        ProgressbarMultiProcessText must must call the `progress` method of
        the returned ProgressbarMultiProcessProxy object with the current
        count. This is a little less intrusive regarding the tracked
        function.

        Parameters
        ----------
        total_count : int
            Total count that will be equivalent to 100% for function.

        Returns
        -------
        obj : ProgressbarMultiProcessProxy object
            The proxy progressbar.

        """
        # xxxxx Inline class definition - Start xxxxxxxxxxxxxxxxxxxxxxxxxxx
        class ProgressbarMultiProcessProxy:
            """Proxy progressbar that behaves like a ProgressbarText object,
            but is actually updating a ProgressbarMultiProcessText progressbar.

            """
            def __init__(self, process_id, process_data_list):
                """Initializes the ProgressbarMultiProcessProxy object."""
                self.process_id = process_id
                self._process_data_list = process_data_list

            def progress(self, count):
                """Updates the proxy progress bar.

                Parameters
                ----------
                count : int
                    The new amount of progress.

                """
                self._process_data_list[self.process_id] = count
        # xxxxx Inline class definition - End xxxxxxxxxxxxxxxxxxxxxxxxxxx
        return ProgressbarMultiProcessProxy(*self.register_function(total_count))

    def _update_progress(self):
        """Collects the progress from each registered proxy progressbar and
        updates the actual visible progressbar.

        """
        pbar = ProgressbarText(self._total_final_count, self._progresschar, self._message)
        self.running.set()
        count = 0
        while count < self._total_final_count and self.running.is_set():
            time.sleep(self._sleep_time)
            count = sum(self._process_data_list)
            pbar.progress(count)

        # It may exit the while loop in two situations: if count reached
        # the maximum allowed value, in which case the progressbar is full,
        # or if the self.running event was cleared in another
        # process. Since in the first case the event is still set, we clear
        # it here to have some consistence (a cleared event will always
        # mean that the progressbar is not running).
        self.running.clear()
        self._toc.value = time.time()

    def start_updater(self):
        """Start the process that updates the progressbar.
        """
        self._tic.value = time.time()
        self._update_process.start()

    def stop_updater(self):
        """Stop the process updating the progressbar.

        You should always call this function in your main process (the same
        that created the progressbar) after joining all the processes that
        update the progressbar. This guarantees that the progressbar
        updated any pending change and exited clearly.

        """
        self.running.clear()
        self._toc.value = time.time()
        self._update_process.join()

    # TODO: Check if the duration property work correctly
    @property
    def duration(self, ):
        """Duration of the progress.

        Returns
        -------
        toc_minus_tic : float
            The duration passed until the progressbar reaches 100%.
        """
        # The progressbar is still running, calculate the duration since
        # the beginning
        if self.running.is_set():
            toc = time.time()
        else:
            toc = self._toc.value

        return toc - self._tic.value
# xxxxxxxxxx ProgressbarMultiProcessText - END xxxxxxxxxxxxxxxxxxxxxxxxxxxx
