#!/usr/bin/env python

# pylint: disable=E1101,E1103,W0403
"""
Tests for the modules in the progressbar package.

Each module has doctests for its functions and all we need to do is run all
of them.
"""

import doctest
import os
import unittest
from io import StringIO
from time import sleep, time

from pyphysim.progressbar import progressbar
from pyphysim.util.misc import pretty_time


def delete_file_if_possible(filename):
    """
    Try to delete the file with name `filename`.

    Parameters
    ----------
    filename : str
        The name of the file to be removed.
    """
    try:
        os.remove(filename)
    except OSError:  # pragma: no cover
        pass


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# noinspection PyMethodMayBeStatic
class SimulationsDoctestsTestCase(unittest.TestCase):
    """
    Test case that run all the doctests in the modules of the simulations
    package.
    """
    def test_progressbar(self):
        """Run progressbar doctests"""
        doctest.testmod(progressbar)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Progressbar Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# This function is used in test methods of progressbars. It simply opens a
# file and return its content as a string.
# noinspection PyUnboundLocalVariable
def _get_progress_string_from_file(filename):
    try:
        # fid = open(filename, 'r', newlines='\n')
        # except Exception as _:
        fid = open(filename, 'r')
    except FileNotFoundError:
        # Sometimes the file is not found because the progressbar did not
        # created it yet. Let's wait a little if that happen and try one
        # more time.
        import time
        time.sleep(3)
        fid = open(filename, 'r')
    finally:
        content_string = fid.read()
        fid.close()

    output = content_string.split('\n')[-1]
    if output == '':
        output = "{0}\n".format(content_string.split('\n')[-2])
    return output


# This function is used in test methods for the ProgressbarText class (and
# other classes that use ProgressbarText)
def _get_clear_string_from_stringio_object(mystring):  # pragma: no cover
    if isinstance(mystring, StringIO):
        # mystring is actually a StringIO object
        value = mystring.getvalue()
    else:
        # mystring is a regular string
        value = mystring

    output = value.split('\r')
    if len(output) > 1:
        output = output[0] + output[-1].strip(' ')
    else:
        # This will be the case when this file is run in Python 3, since
        # there will be no '\r' character in mystring.
        output = output[0]

    return output


class ProgressbarTextTestCase(unittest.TestCase):
    def setUp(self):
        self.message = "ProgressbarText Unittest"
        # The progress will be printed to the StringIO object instead of
        # sys.stdout
        self.out = StringIO()
        self.tic = time()
        self.pbar = progressbar.ProgressbarText(50,
                                                '*',
                                                self.message,
                                                output=self.out)

        self.out2 = StringIO()
        self.pbar2 = progressbar.ProgressbarText(25, 'x', output=self.out2)

        # For testing purposes we set _display_interval to zero
        self.pbar._display_interval = 0.0
        self.pbar2._display_interval = 0.0

    def test_write_initialization(self):
        out = StringIO()
        out2 = StringIO()

        pbar = progressbar.ProgressbarText(50,
                                           '*',
                                           self.message,
                                           output=out,
                                           width=80)
        # Setting the width to a value below 40 should actually set the
        # width to 40
        pbar2 = progressbar.ProgressbarText(25,
                                            'x',
                                            message="Just a Message",
                                            output=out2,
                                            width=30)

        self.assertEqual(pbar.finalcount, 50)
        self.assertEqual(pbar2.finalcount, 25)

        # self.pbar.width = 80
        self.assertEqual(
            out.getvalue(),
            ("--------------------------- ProgressbarText Unittest -------"
             "-------------------1\n       1       2       3       4      "
             " 5       6       7       8       9       0\n-------0-------0"
             "-------0-------0-------0-------0-------0-------0-------0----"
             "---0\n"))

        # In the constructor we asked for a width of 30, but setting the width
        # to a value below 40 should actually set the width to 40
        self.assertEqual(pbar2.width, 40)
        self.assertEqual(
            out2.getvalue(), "------------ Just a Message -----------1\n"
            "   1   2   3   4   5   6   7   8   9   0\n"
            "---0---0---0---0---0---0---0---0---0---0\n")

    def test_progress(self):
        self.assertEqual(self.pbar.finalcount, 50)
        self.assertEqual(self.pbar2.finalcount, 25)
        self.assertEqual(self.pbar.n, 0)
        self.assertEqual(self.pbar2.n, 0)

        # Before the first time the progress method is called, the
        # _start_time and _stop_time variables used to track the elapsed
        # time are equal to zero.
        self.assertEqual(self.pbar.elapsed_time, '0.00s')
        self.assertAlmostEqual(self.pbar._start_time, self.tic, places=2)
        self.assertEqual(self.pbar._stop_time, 0.0)

        # Progress 20% (10 is equivalent to 20% of 50)
        self.pbar.progress(10)
        self.assertEqual(self.pbar.n, 10)
        self.assertEqual(
            _get_clear_string_from_stringio_object(self.out),
            # self.out.getvalue(),
            "------------ ProgressbarText Unittest -----------1\n"
            "    1    2    3    4    5    6    7    8    9    0\n"
            "----0----0----0----0----0----0----0----0----0----0\n"
            "**********")

        # After calling the "progress" method but before the progress
        # reaches 100% the _start_time is greater than zero while the
        # _stop_time is still zero.
        self.assertTrue(self.pbar._start_time > 0.0)
        self.assertEqual(self.pbar._stop_time, 0.0)

        # Progress to 70%
        self.pbar.progress(35)
        self.assertEqual(self.pbar.n, 35)
        self.assertEqual(
            _get_clear_string_from_stringio_object(self.out),
            "------------ ProgressbarText Unittest -----------1\n"
            "    1    2    3    4    5    6    7    8    9    0\n"
            "----0----0----0----0----0----0----0----0----0----0\n"
            "***********************************")

        sleep(0.01)

        # Progress to 100% -> Note that in the case of 100% a new line is
        # added at the end.
        #
        # Anything greater than or equal the final count will set the
        # progress to 100%
        self.pbar.progress(55)
        self.assertEqual(self.pbar.n, 50)
        self.assertEqual(
            _get_clear_string_from_stringio_object(self.out),
            "------------ ProgressbarText Unittest -----------1\n"
            "    1    2    3    4    5    6    7    8    9    0\n"
            "----0----0----0----0----0----0----0----0----0----0\n"
            "**************************************************\n")

        # After progress reaches 100 both _start_time and _stop_time
        # variables are greater than zero.
        self.assertTrue(self.pbar._start_time > 0.0)
        self.assertTrue(self.pbar._stop_time > 0.0)
        self.assertEqual(
            self.pbar.elapsed_time,
            pretty_time(self.pbar._stop_time - self.pbar._start_time))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with pbar2, which uses the default progress message and the
        # character 'x' to indicate progress.
        self.pbar2.progress(20)
        self.assertEqual(
            _get_clear_string_from_stringio_object(self.out2),
            "------------------- % Progress ------------------1\n"
            "    1    2    3    4    5    6    7    8    9    0\n"
            "----0----0----0----0----0----0----0----0----0----0\n"
            "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

        self.assertEqual(self.pbar.finalcount, 50)
        self.assertEqual(self.pbar2.finalcount, 25)

    def test_str(self):
        self.pbar.progress(10)
        self.assertEqual(str(self.pbar),
                         '**********                                        ')

        self.pbar.progress(25)
        self.assertEqual(str(self.pbar),
                         '*************************                         ')

    def test_small_progress_and_zero_finalcount(self):
        # Test the case when the progress is lower then 1%.
        out = StringIO()
        pbar3 = progressbar.ProgressbarText(finalcount=200, output=out)
        pbar3.progress(1)
        self.assertEqual(
            _get_clear_string_from_stringio_object(out),
            "------------------- % Progress ------------------1\n"
            "    1    2    3    4    5    6    7    8    9    0\n"
            "----0----0----0----0----0----0----0----0----0----0\n")

    def test_deleting_progress_file_after_progress_finished(self):
        out = open('test_progress_file1.txt', 'w')
        pbar = progressbar.ProgressbarText(50,
                                           '*',
                                           'Progress message',
                                           output=out)

        out2 = open('test_progress_file2.txt', 'w')
        pbar2 = progressbar.ProgressbarText(25,
                                            'x',
                                            'Progress message',
                                            output=out2)

        out3 = open('test_progress_file3.txt', 'w')
        pbar3 = progressbar.ProgressbarText(30,
                                            'o',
                                            'Progress message',
                                            output=out3)

        pbar.delete_progress_file_after_completion = True
        pbar.progress(15)
        pbar.progress(37)
        # Progress finishes and there is not explicit call to the stop method.
        pbar.progress(50)

        pbar2.delete_progress_file_after_completion = True
        pbar2.progress(7)
        pbar2.progress(21)
        # Explicitly call the stop method to finish the progress.
        pbar2.stop()

        pbar3.delete_progress_file_after_completion = True
        pbar3.progress(10)
        pbar3.progress(21)
        pbar3.progress(28)
        # Progress will not finish, but we will explicitly delete pbar3
        # here to test if the file it is writing to is delete in that case.
        del pbar3

        # Close the output files.
        out.close()
        out2.close()
        out3.close()

        # The first progressbar was marked to erase the file after the
        # progress finishes. therefore, if we try to delete it here python
        # should raise an OSError exception.
        with self.assertRaises(OSError):
            os.remove('test_progress_file1.txt')

        # The second progressbar was marked to erase the file after the
        # progress finishes. therefore, if we try to delete it here python
        # should raise an OSError exception.
        with self.assertRaises(OSError):
            os.remove('test_progress_file2.txt')

        # The third progressbar was marked to erase the file after the
        # progress finishes. therefore, if we try to delete it here python
        # should raise an OSError exception.
        with self.assertRaises(OSError):
            os.remove('test_progress_file3.txt')


class ProgressbarText2TestCase(unittest.TestCase):
    def setUp(self):
        message = "ProgressbarText Unittest"
        # The progress will be printed to the StringIO object instead of
        # sys.stdout
        self.out = StringIO()
        self.pbar = progressbar.ProgressbarText2(50,
                                                 '*',
                                                 message,
                                                 output=self.out)

        self.out2 = StringIO()
        self.pbar2 = progressbar.ProgressbarText2(50, '*', output=self.out2)

    def test_get_percentage_representation(self):
        # xxxxxxxxxx Tests for bar width of 50 xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.assertEqual(
            self.pbar._get_percentage_representation(50,
                                                     central_message='',
                                                     left_side='',
                                                     right_side=''),
            '*************************                         ')

        self.assertEqual(
            self.pbar._get_percentage_representation(50, central_message=''),
            '[************************                        ]')

        self.assertEqual(self.pbar._get_percentage_representation(30),
                         '[**************         30%                      ]')

        self.assertEqual(
            self.pbar._get_percentage_representation(
                70,
                central_message='{percent}% (Time: {elapsed_time})',
                left_side='',
                right_side=''),
            '*****************70% (Time: 0.00s)*               ')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Tests for bar width of 80 xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.pbar2.width = 80
        self.assertEqual(
            self.pbar2._get_percentage_representation(50,
                                                      central_message='',
                                                      left_side='',
                                                      right_side=''),
            "****************************************"
            "                                        ")

        self.assertEqual(
            self.pbar2._get_percentage_representation(50, central_message=''),
            '[***************************************'
            '                                       ]')

        self.assertEqual(
            self.pbar2._get_percentage_representation(25, central_message=''),
            '[*******************'
            '                                                           ]')

        self.assertEqual(
            self.pbar2._get_percentage_representation(
                70, central_message='Progress: {percent}'),
            '[*********************************Progress: 70*********'
            '                        ]')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_progress(self):
        self.assertEqual(self.pbar.finalcount, 50)
        self.assertEqual(self.pbar2.finalcount, 50)

        self.pbar.progress(15)
        self.assertEqual(
            self.out.getvalue(),
            "\r[**************         30%                      ]"
            "  ProgressbarText Unittest")

        self.pbar.progress(50)
        self.assertEqual(
            self.out.getvalue(),
            "\r[**************         30%                      ]"
            "  ProgressbarText Unittest\r"
            "[**********************100%**********************]"
            "  ProgressbarText Unittest\n")

        # Progressbar with no message -> Use a default message
        self.pbar2.progress(15)
        self.assertEqual(
            self.out2.getvalue(),
            "\r[**************         30%                      ]"
            "  15 of 50 complete")

        self.assertEqual(self.pbar.finalcount, 50)
        self.assertEqual(self.pbar2.finalcount, 50)


class ProgressbarText3TestCase(unittest.TestCase):
    def setUp(self):
        message = "ProgressbarText Unittest"
        # The progress will be printed to the StringIO object instead of
        # sys.stdout
        self.out = StringIO()
        self.pbar = progressbar.ProgressbarText3(50,
                                                 '*',
                                                 message,
                                                 output=self.out)

        self.out2 = StringIO()
        self.pbar2 = progressbar.ProgressbarText3(50, '*', output=self.out2)

    def test_progress(self):
        self.assertEqual(self.pbar.finalcount, 50)
        self.assertEqual(self.pbar2.finalcount, 50)

        self.pbar.progress(15)

        self.assertEqual(
            self.out.getvalue(),
            "\r********* ProgressbarText Unittest 15/50 *********")

        self.pbar.progress(50)
        self.assertEqual(
            self.out.getvalue(),
            "\r********* ProgressbarText Unittest 15/50 *********\r"
            "********* ProgressbarText Unittest 50/50 *********")

        # Test with no message (use default message)
        self.pbar2.progress(40)
        self.assertEqual(
            self.out2.getvalue(),
            "\r********************** 40/50 *********************")

        self.assertEqual(self.pbar.finalcount, 50)
        self.assertEqual(self.pbar2.finalcount, 50)


class ProgressbarMultiProcessTextTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.output_filename = "ProgressbarMultiProcessTextTestCase.out"

        self.mpbar = progressbar.ProgressbarMultiProcessServer(
            message="Some message",
            sleep_time=0.001,
            filename=self.output_filename)
        self.proxybar1 = self.mpbar.register_client_and_get_proxy_progressbar(
            10)
        self.proxybar2 = self.mpbar.register_client_and_get_proxy_progressbar(
            15)

    def test_register(self):
        # Test last_id and total_final_count of the main progress bar
        self.assertEqual(self.mpbar._last_id, 1)
        self.assertEqual(self.mpbar.finalcount, 25)
        self.assertEqual(self.mpbar.num_clients, 2)

        # Register a new proxy progressbar and test the last_id and
        # total_final_count again.
        proxybar3 = self.mpbar.register_client_and_get_proxy_progressbar(13)
        self.assertEqual(self.mpbar._last_id, 2)
        self.assertEqual(self.mpbar.finalcount, 38)
        self.assertEqual(proxybar3.client_id, 2)

    def test_proxy_progressbars(self):
        # Test the information in the proxybar1
        self.assertEqual(self.proxybar1.client_id, 0)
        self.assertTrue(
            self.proxybar1._client_data_list is self.mpbar._client_data_list)

        # Test the information in the proxybar2
        self.assertEqual(self.proxybar2.client_id, 1)
        self.assertTrue(
            self.proxybar2._client_data_list is self.mpbar._client_data_list)

    # Note: This method will sleep for 0.01 seconds thus adding to the total
    # amount of time required to run all tests. Unfortunately, this is a
    # necessary cost.
    def test_updater(self):
        # Remove old file from previous test run
        delete_file_if_possible(self.output_filename)

        # Suppose that the first process already started and called the
        # proxybar1 to update its progress.
        self.proxybar1.progress(6)

        # Then we start the "updater" of the main progressbar.
        self.mpbar.start_updater(start_delay=0.03)

        # Register a new proxybar after start_updater was called. This only
        # works because we have set a start_delay
        proxy3 = self.mpbar.register_client_and_get_proxy_progressbar(25)
        proxy3.progress(3)

        # Then the second process updates its progress
        self.proxybar2.progress(6)
        # self.mpbar.stop_updater()

        # Sleep for a very short time so that the
        # ProgressbarMultiProcessServer object has time to create the file
        # with the current progress
        sleep(1.0)

        self.mpbar.stop_updater(0)

        # Open and read the progress from the file
        progress_string = _get_progress_string_from_file(self.output_filename)

        # Expected string with the progress output
        expected_progress_string = ("[**************         30%             "
                                    "         ]  Some message")
        self.assertEqual(
            _get_clear_string_from_stringio_object(progress_string),
            expected_progress_string)

    def test_start_and_stop_updater_process(self):
        self.assertFalse(self.mpbar.is_running)
        self.assertEqual(self.mpbar._start_updater_count, 0)
        self.mpbar.start_updater()
        # We need some time for the process to start and self.mpbar.running
        # is set
        sleep(0.1)
        self.assertEqual(self.mpbar._start_updater_count, 1)
        self.assertTrue(self.mpbar.is_running)

        # Call the start_updater a second time. This should not really try
        # to start the updater process, since it is already started.
        self.mpbar.start_updater()
        self.assertEqual(self.mpbar._start_updater_count, 2)

        # Since we called start_updater two times, calling stop_updater
        # only once should not stop the updater process. We need to
        # stop_updater as many times as start_updater so that the updater
        # process is actually stopped.
        self.mpbar.stop_updater()
        self.assertEqual(self.mpbar._start_updater_count, 1)
        self.assertTrue(self.mpbar.is_running)

        self.mpbar.stop_updater()
        self.assertEqual(self.mpbar._start_updater_count, 0)
        self.assertFalse(self.mpbar.is_running)


class ProgressBarIPythonTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.pbar = progressbar.ProgressBarIPython(50, "message")

    def test_atributes(self):
        self.assertEqual(self.pbar.finalcount, 50)
        self.assertEqual(self.pbar.message, "message")


# TODO: finish implementation
class ProgressbarZMQTextTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.output_filename = "ProgressbarZMQTextTestCase.out"

        self.zmqbar = progressbar.ProgressbarZMQServer(
            message="Some message",
            sleep_time=0.1,
            filename=self.output_filename,
            port=7755)
        self.proxybar1 \
            = self.zmqbar.register_client_and_get_proxy_progressbar(10)
        self.proxybar2 \
            = self.zmqbar.register_client_and_get_proxy_progressbar(15)

    def tearDown(self):
        self.zmqbar.stop_updater()

    def test_register(self):
        # Test last_id and total_final_count of the main progress bar
        self.assertEqual(self.zmqbar._last_id, 1)
        self.assertEqual(self.zmqbar.finalcount, 25)
        self.assertEqual(self.zmqbar.num_clients, 2)

        # Register a new proxy progressbar and test the last_id and
        # total_final_count again.
        proxybar3 = self.zmqbar.register_client_and_get_proxy_progressbar(13)
        self.assertEqual(self.zmqbar._last_id, 2)
        self.assertEqual(self.zmqbar.finalcount, 38)

        # Test IP and port of the proxy progress bars
        self.assertEqual(self.proxybar1.ip, self.zmqbar.ip)
        self.assertEqual(self.proxybar1.port, self.zmqbar.port)
        self.assertEqual(self.proxybar2.ip, self.zmqbar.ip)
        self.assertEqual(self.proxybar2.port, self.zmqbar.port)
        self.assertEqual(proxybar3.ip, self.zmqbar.ip)
        self.assertEqual(proxybar3.port, self.zmqbar.port)

    def test_proxy_progressbars(self):
        # Test the information in the proxybar1
        self.assertEqual(self.proxybar1.client_id, 0)
        self.assertEqual(self.proxybar1.ip, self.zmqbar.ip)
        self.assertEqual(self.proxybar1.port, self.zmqbar.port)

        # Test the information in the proxybar2
        self.assertEqual(self.proxybar2.client_id, 1)
        self.assertEqual(self.proxybar2.ip, self.zmqbar.ip)
        self.assertEqual(self.proxybar2.port, self.zmqbar.port)

        # When the ProgressbarZMQClient object is created the socket variables
        # are None. In the first time the `progress` method is called the zmq
        # sockets will be set
        self.assertIsNone(self.proxybar1._zmq_push_socket)
        self.assertIsNone(self.proxybar1._zmq_context)
        self.assertIsNone(self.proxybar2._zmq_push_socket)
        self.assertIsNone(self.proxybar2._zmq_context)

    def test_update_progress(self):
        try:
            # noinspection PyUnresolvedReferences
            import zmq
            assert zmq  # Avoid unused import warning for DataFrame
        except ImportError:  # pragma: no cover
            self.skipTest("The zmq module is not installed")

        self.zmqbar.start_updater()
        self.proxybar1.progress(5)
        # We can also use a "call syntax" for the progress progressbars
        self.proxybar2(10)
        sleep(0.3)

        # Open and read the progress from the file. We open in binary mode
        # to avoid a possible conversion of '\r' to '\n' by the 'read'
        # method.
        progress_string = _get_progress_string_from_file(self.output_filename)

        # Expected string with the progress output
        expected_progress_string = ("[***********************60%**          "
                                    "          ]  Some message")
        self.assertEqual(
            _get_clear_string_from_stringio_object(progress_string),
            expected_progress_string)

        # ------------------------
        self.zmqbar.stop_updater()
        # ------------------------

        # After the stop_updater method the progressbar should be full
        progress_string2 = _get_progress_string_from_file(self.output_filename)
        expected_progress_string2 = ("[**********************100%************"
                                     "**********]  Some message\n")
        self.assertEqual(
            _get_clear_string_from_stringio_object(progress_string2),
            expected_progress_string2)


# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
