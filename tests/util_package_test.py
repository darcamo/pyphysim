#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the util package.

Each module has doctests for its functions and all we need to do is run all
of them them.

"""
import unittest
import doctest
import sys

sys.path.append("..")

from util import misc, progressbar


class Test(unittest.TestCase):
    """Unit tests for misc."""

    def test_progressbar(self):
        """Run progressbar doctests"""
        doctest.testmod(progressbar)

    def test_misc(self):
        """Run misc doctests"""
        doctest.testmod(misc)

if __name__ == "__main__":
    unittest.main()
