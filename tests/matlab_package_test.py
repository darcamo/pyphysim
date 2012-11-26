#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the MATLAB package.

Each module has doctests for its functions and all we need to do is run all
of them.
"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import numpy as np

from MATLAB import python2MATLAB


class MATLABDoctestsTestCase(unittest.TestCase):
    """Test case that run all the doctests in the modules of the MATLAB
    package.

    """
    def test_python2MATLAB(self):
        """Run python2MATLAB doctests"""
        doctest.testmod(python2MATLAB)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
