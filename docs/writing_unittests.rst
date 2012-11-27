Writing Unittests for PyPhysim
==============================

The standard :mod:`unittest` module is used to implement automated tests
for the several packages in the PyPhysim library. The tests are located in
the `tests` folder, which should contain a single test file for each
package in PyPhysim.

Each test file contains several *test case* classes, as usual in the
unittests framework. The first test case class is always a *Doctests test
case*, that is, a test case that simple run all the doctests in each module
of the package.

After that, a test case class must be implemented for each module in the
package, which in turn should test all the classes and functions in that
module.


Test Coverage
-------------

Ideally unittests should be implemented at the same time the actual code is
implemented (or even before that). However, sometimes the tests are
implemented the tested code and this may leave code untested.

A good way to make sure that we have a good test coverage in PyPhysim is
using the `python-coverage` program. With it we can run all the implemented
unittests and get a report of the lines in any module in PyPhysim that was
not run by any unittest (thus finding untested code). The `bin` folder
contains a script called **run_python_coverage.sh** that will do exactly
that.


Code Quality
------------

A good way to ensure code quality, besides implementing unittests, is to
employ code analisys tools such as pylint, pep8, pychecker, etc.

For the quality of PyPhysim as a package, see the cheesecake_index tool.
 - http://pycheesecake.org/
 - http://infinitemonkeycorps.net/docs/pph/

.. todo:: Run pylint on the files in PyPhysim.
