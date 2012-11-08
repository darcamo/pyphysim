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

__version__ = "$Revision$"
# $Source$

import multiprocessing
import time


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx DummyProgressbar - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class DummyProgressbar(object):  # pragma: no cover
    """Dummy progress bar that don't really do anything."""

    def __init__(self, ):
        pass

    def progress(self, count):
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

    Example:
    >>> pb = ProgressbarText(100, 'o', "Hello Simulation")
    <BLANKLINE>
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
    def __init__(self, finalcount, progresschar='*', message=''):
        import sys
        self.finalcount = finalcount
        self.blockcount = 0
        self.block = progresschar
        #
        # Get pointer to sys.stdout so I can use the write/flush
        # methods to display the progress bar.
        #
        self.f = sys.stdout
        #
        # If the final count is zero, don't start the progress gauge
        #
        if not self.finalcount:
            return
        if(len(message) != 0):
            bartitle = '\n{0}\n'.format(self.center_message(message, 50, '-', '', '1'))
        else:
            bartitle = '\n------------------ % Progress -------------------1\n'

        self.f.write(bartitle)
        self.f.write('    1    2    3    4    5    6    7    8    9    0\n')
        self.f.write('----0----0----0----0----0----0----0----0----0----0\n')
        return

    def progress(self, count):
        """Updates the progress bar.

        Arguments:
        - `count`: The current percentage of completeness.
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
            for i in range(self.blockcount, blockcount):
                self.f.write(self.block)
                self.f.flush()
            # Update self.blockcount
            self.blockcount = blockcount

        # If we completed the bar, print a newline
        if percentcomplete == 100:
            self.f.write("\n")

    @classmethod
    def center_message(cls, message, length=50, fill_char=' ', left='', right=''):
        """Return a string with `message` centralized and surrounded by
        fill_char.

        Arguments:
        - `cls`: This Class. This is required because this method was
                 marked as a classmethod
        - `message`: The message to be centered
        - `length`: Total length of the centered message (original + any fill)
        - `fill_char`:
        - `left`:
        - `right`:

        >>> print ProgressbarText.center_message("Hello Progress", 50, '-', 'Left', 'Right')
        Left------------- Hello Progress ------------Right
        """
        message_size = len(message)
        left_size = len(left)
        right_size = len(right)
        fill_size = (length - (message_size + 2) - left_size - right_size)
        left_fill_size = fill_size // 2 + (fill_size % 2)
        right_fill_size = (fill_size // 2)

        new_message = "{0}{1} {2} {3}{4}".format(left,
                                               fill_char * left_fill_size,
                                               message,
                                               fill_char * right_fill_size,
                                               right)
        return new_message
# xxxxxxxxxx ProgressbarText - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ProgressbarMultiProcessText - START xxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProgressbarMultiProcessText(object):
    """Class that prints a representation of the current progress of
    multiple process as text.

    While the ProgressbarText only tracks the progress of a single process,
    the ProgressbarMultiProcessText class can track the joint progress of
    multiple processes. This is used when you parallelize some task using
    the multiprocessing module.

    The usage requires a little more work than ProgressbarText and is
    described as follows:
      - First you create the ProgressbarMultiProcessText as usual (although
        it does not receive the value equivalent to the full progress yet).
      - Then, for each process, call the
        register_function_and_get_proxy_progressbar function passing the
        number equivalent to full progress for that process. This function
        returns a "proxy progressbar" that must be passed as argument to
        that process. Each process will call the "progress" method of that
        proxy as if it was a ProgressbarText object.
      - Start all the processes and call the start_updater method of
        ProgressbarMultiProcessText object so that the bar is updated by
        the different processes.
      - After joining all the process (all work is finished) call the
        stop_updater method of the ProgressbarMultiProcessText object.

    Ex:
    TODO: Write an example here. See the find_codebook.py application.
    """

    def __init__(self,
                 progresschar='*',
                 message='',
                 sleep_time=1):
        """
        Arguments:
        - `progress_queue`: Queue where the multiple processes will put
                            their progress. This must be a
                            multiprocessing.Manager.Queue object. The
                            multiprocessing.Queue version will break
                            things.
        - `total_final_count`: Total count of all progress.
        - `progresschar`: Character used in the progressbar.
        - `message`: Message writen in the progressbar
        - `sleep_time`: Time between progressbar updates
        """
        # total_final_count will be updated each time the register_*
        # function is called
        self._total_final_count = 0
        self._progresschar = progresschar
        self._message = message

        self._manager = multiprocessing.Manager()
        self._process_data_list = self._manager.list()

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

        Arguments:
        - `total_count`: Total count that will be equivalent to 100% for
                         function.

        Returns:
        - Tuple with the `process_id` and the process_data_list. The
          function whose process is tracked by the
          ProgressbarMultiProcessText must update the element `process_id`
          of the list `process_data_list` with the current count.
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
        """Similar to the register_function method, but returns a
        ProgressbarMultiProcessProxy object.

        The function whose process is tracked by the
        ProgressbarMultiProcessText must must call the `progress` method of
        the returned ProgressbarMultiProcessProxy object with the current
        count. This is a little less intrusive regarding the tracked
        function.

        Arguments:
        - `total_count`: Total count that will be equivalent to 100% for
                         function.

        Returns: ProgressbarMultiProcessProxy object
        """
        # xxxxx Inline class definition - Start xxxxxxxxxxxxxxxxxxxxxxxxxxx
        class ProgressbarMultiProcessProxy:
            def __init__(self, process_id, process_data_list):
                self.process_id = process_id
                self._process_data_list = process_data_list

            def progress(self, count):
                self._process_data_list[self.process_id] = count
        # xxxxx Inline class definition - End xxxxxxxxxxxxxxxxxxxxxxxxxxx
        return ProgressbarMultiProcessProxy(*self.register_function(total_count))

    def progress(self):
        """This function should not be called."""
        print "ProgressbarMultiProcessText.progress: This function should not be called directly. Call the register_function_and_get_proxy_progressbar function to get a proxy object for the progressbar and call the progress method of that proxy."

    def _update_progress(self):
        bar = ProgressbarText(self._total_final_count, self._progresschar, self._message)
        self.running.set()
        count = 0
        while count < self._total_final_count and self.running.is_set():
            time.sleep(self._sleep_time)
            count = sum(self._process_data_list)
            bar.progress(count)

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

        Returns:
         - Process that updates the bar. Call the join method of the
           returned value at the end of your program.
        """
        self._tic.value = time.time()
        self._update_process.start()

    def stop_updater(self, ):
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
        """
        # The progressbar is still running, calculate the duration since
        # the beginning
        if self.running.is_set():
            toc = time.time()
        else:
            toc = self._toc.value

        return toc - self._tic.value
# xxxxxxxxxx ProgressbarMultiProcessText - END xxxxxxxxxxxxxxxxxxxxxxxxxxxx
