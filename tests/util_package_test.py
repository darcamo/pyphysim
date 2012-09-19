#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the util package.

Each module has doctests for its functions and all we need to do is run all
of them them.
"""

import unittest
import doctest
import numpy as np

from util import misc, progressbar, simulations, conversion


class UtilDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the util
    package.
    """

    def test_progressbar(self):
        """Run progressbar doctests"""
        doctest.testmod(progressbar)

    def test_misc(self):
        """Run misc doctests"""
        doctest.testmod(misc)

    def test_simulations(self):
        """Run simulations doctests"""
        doctest.testmod(simulations)

    def test_conversion(self):
        """Run conversion doctests"""
        doctest.testmod(conversion)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx misc Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MiscFunctionsTestCase(unittest.TestCase):
    """Test the functions in the module."""
    def test_pgig(self, ):
        """
        """
        A = np.array(
            [[2 - 0j, 3 + 12j, 7 + 1j],
             [3 - 12j, 6 + 0j, 5 + 3j],
             [7 - 1j, 5 - 3j, 4 + 0j]])

        # xxxxx Test for n==3 (all columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n3, D_n3] = misc.peig(A, 3)

        expected_V_n3 = np.array(
            [[0.27354856 + 0.54286421j, 0.15266747 - 0.35048035j, 0.69593520],
             [0.68522942, -0.24255902 + 0.37567057j, -0.02693857 + 0.57425752j],
             [0.38918583 + 0.09728652j, 0.80863645, -0.40625488 - 0.14189355j]])
        np.testing.assert_array_almost_equal(V_n3, expected_V_n3)

        # xxxxx Test for n==2 (two columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n2, D_n2] = misc.peig(A, 2)

        expected_V_n2 = np.array(
            [[0.27354856 + 0.54286421j, 0.15266747 - 0.35048035j],
             [0.68522942, -0.24255902 + 0.37567057j],
             [0.38918583 + 0.09728652j, 0.80863645]])
        np.testing.assert_array_almost_equal(V_n2, expected_V_n2)

        # xxxxx Test for n==1 (one column) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n1, D_n1] = misc.peig(A, 1)

        expected_V_n1 = np.array(
            [[0.27354856 + 0.54286421j],
             [0.68522942],
             [0.38918583 + 0.09728652j]])
        np.testing.assert_array_almost_equal(V_n1, expected_V_n1)

    def test_leig(self):
        A = np.array(
            [[2 - 0j, 3 + 12j, 7 + 1j],
             [3 - 12j, 6 + 0j, 5 + 3j],
             [7 - 1j, 5 - 3j, 4 + 0j]])

        # xxxxx Test for n==3 (all columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n3, D_n3] = misc.leig(A, 3)

        expected_V_n3 = np.array(
            [[0.69593520, 0.15266747 - 0.35048035j, 0.27354856 + 0.54286421j],
             [-0.02693857 + 0.57425752j, -0.24255902 + 0.37567057j, 0.68522942],
             [-0.40625488 - 0.14189355j, 0.80863645, 0.38918583 + 0.09728652j]])
        np.testing.assert_array_almost_equal(V_n3, expected_V_n3)

        # xxxxx Test for n==2 (two columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n2, D_n2] = misc.leig(A, 2)

        expected_V_n2 = np.array(
            [[0.69593520, 0.15266747 - 0.35048035j],
             [-0.02693857 + 0.57425752j, -0.24255902 + 0.37567057j],
             [-0.40625488 - 0.14189355j, 0.80863645]])
        np.testing.assert_array_almost_equal(V_n2, expected_V_n2)

        # xxxxx Test for n==1 (one column) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n1, D_n1] = misc.leig(A, 1)

        expected_V_n1 = np.array(
            [[0.69593520],
             [-0.02693857 + 0.57425752j],
             [-0.40625488 - 0.14189355j]])
        np.testing.assert_array_almost_equal(V_n1, expected_V_n1)



# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# def producer(queue, sleep_time=0):
#     process_pid = multiprocessing.current_process().pid
#     for i in range(1, 11):
#         sleep(sleep_time)
#         # TODO put a tuple in the queue instead of just the value. The
#         # tuple should have the process identifier and the value.
#         queue.put((process_pid, i))
#     queue.put((process_pid, -1))
#
#
# def consumer(queue):
#     print "consumer started"
#     while True:
#         sleep(0.8)
#         if queue.empty() == False:
#             value = queue.get()
#             if value < 0:
#                 # Negative value means stop
#                 print "received poison pill"
#                 break
#             print "Consumer read the value: {0}".format(value)
#     print "consumer ended"
#
#
# if __name__ == '__main__1':
#     # Extract the code below somewhere later
#     import multiprocessing
#     from time import sleep
#
#     queue = multiprocessing.Queue()
#
#     p = multiprocessing.Process(target=producer, args=[queue])
#     c = multiprocessing.Process(target=consumer, args=[queue])
#
#     p.start()
#     c.start()
#     p.join()
#     #queue.put(-1)
#     c.join()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#
#
# from progressbar import ProgressbarText
#
# def progress_bar_background(queue):
#     bar = ProgressbarText(10, "o", "Teste")
#     simulating = True
#     # Processes that put something in the queue
#     #runnin_processes = set()
#     while simulating:
#         if queue.empty() == False:
#             (process_pid, value) = queue.get()
#             if value < 0:
#                 # Negative value means stop
#                 simulating = False
#             else:
#                 #sleep(1)
#                 bar.progress(value)
#
#
# if __name__ == '__main__2':
#     import multiprocessing
#     from time import sleep
#     from progressbar import ProgressbarMultiProcessText
#
#     # Runs in a different process and owns the queue
#     manager = multiprocessing.Manager()
#
#     queue = manager.Queue()
#     # args: queue, job_id, sleep_time
#     p2 = multiprocessing.Process(target=producer, args=[queue, 0, 0.8])
#     p1 = multiprocessing.Process(target=producer, args=[queue, 1, 0.5])
#     #p2 = multiprocessing.Process(target=producer, args=[queue])
#
#     # bar = ProgressbarMultiProcessText(queue, total_final_count=40, progresschar='o', message="Teste")
#     #c = progressbar.start_updater()
#     c = multiprocessing.Process(target=progress_bar_background, args=[queue])
#
#     p1.start()
#     p2.start()
#     c.start()
#     #c = progressbar.start_updater()
#
#     p1.join()
#     p2.join()
#     queue.put(-1)
#     c.join()
#
#     print "FIM"
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def progress_producer2(bar, sleep_time=0.5):
    total_count = 20
    for i in range(1, total_count + 1):
        sleep(sleep_time)
        bar.progress(i)


def progress_producer(process_id, process_data_list, sleep_time=0.5):
    total_count = 20
    for i in range(1, total_count + 1):
        sleep(sleep_time)
        process_data_list[process_id] = i


if __name__ == '__main__1':
    from time import sleep
    import multiprocessing
    from progressbar import ProgressbarMultiProcessText
    bar = ProgressbarMultiProcessText(sleep_time=1)

    # # xxxxx Option 1: register_function xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # # Register two functions with count 20, each, in the progressbar
    # func_1_data = bar.register_function(20)
    # func_2_data = bar.register_function(20)

    # # Create the processes to run the functions
    # p1 = multiprocessing.Process(target=progress_producer, args=(func_1_data[0], func_1_data[1], 0.2))
    # p2 = multiprocessing.Process(target=progress_producer, args=(func_2_data[0], func_2_data[1], 0.3))
    # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Option 2: register_function_and_get_proxy_progressbar xxxxxxxxx
    # Register two functions with count 20, each, in the progressbar
    proxybar1 = bar.register_function_and_get_proxy_progressbar(20)
    proxybar2 = bar.register_function_and_get_proxy_progressbar(20)

    # Create the processes to run the functions
    p1 = multiprocessing.Process(target=progress_producer2, args=(proxybar1, 0.2))
    p2 = multiprocessing.Process(target=progress_producer2, args=(proxybar2, 0.3))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Start the processes and the progressbar updating
    bar.start_updater()
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    # Stop the process that updates the progressbar.
    bar.stop_updater()

    print "FIM"
