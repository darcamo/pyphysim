Speeding up PyPhysim
====================

There a re a number of alternatives to speed-up python code.

- you can use the weave module (inline or blitz methods) from scipy to
  spped up things here. See
  http://docs.scipy.org/doc/scipy/reference/tutorial/weave.html
  and
  http://www.scipy.org/PerformancePython
- You could use Cython
- You could try numexpr
  http://code.google.com/p/numexpr/
- You could try Numba
  http://jakevdp.github.com/blog/2012/08/24/numba-vs-cython/
- Use smart numpy broadcast tricks to avoid loops
  This is fast, but uses more memory. See :meth:`.Modulator.demodulate`


Profiling the Code
------------------

There are multiple ways to profile python code and visualize the profile
output. An easy way is to use the `%run` command in the ipython interpreter
with the `-p` option to run and profile a python script.

If you prefer to graphically visualize the profile data you can use the
`runsnakerun`_ program. In order to use it, first you need to run the
script using the cProfile module to generate the profile data and then you
can visualize the profile data with the runsnakerun program. That is

.. code-block:: bash

   $ python -m cProfile -o <outputfilename> <script-name> <options>
   $ runsnake <outputfilename>

Here is an image (from http://www.vrplumber.com/programming/runsnakerun/)
showing runsnakerun in action.

.. image:: _images/runsnakerun_screenshot-2.0.png
   :scale: 80%
   
.. _runsnakerun: http://www.vrplumber.com/programming/runsnakerun/

