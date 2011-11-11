#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modulators module."""
import unittest
import doctest
import sys

sys.path.append("..")

import modulators


class Test(unittest.TestCase):
    """Unit tests for darlan."""

    def test_modulators(self):
        """Run modulators doctests"""
        doctest.testmod(modulators)

if __name__ == "__main__":
    unittest.main()
