#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the simulations module."""
import unittest
import doctest
import sys

sys.path.append("..")

import simulations


class Test(unittest.TestCase):
    """Unit tests for misc."""

    def test_simulations(self):
        """Run simulations doctests"""
        doctest.testmod(simulations)

if __name__ == "__main__":
    unittest.main()
