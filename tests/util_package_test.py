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

from util import darlan, progressbar


class Test(unittest.TestCase):
    """Unit tests for darlan."""

    def test_progressbar(self):
        """Run progressbar doctests"""
        doctest.testmod(progressbar)

    def test_darlan(self):
        """Run darlan doctests"""
        doctest.testmod(darlan)

if __name__ == "__main__":
    unittest.main()
