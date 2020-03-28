#!/usr/bin/env python
"""
Package with Interference Alignment (IA) algorithms.

Note that all IA algorithms require the channel object and any change to
the channel object must be performed before calling the `solve` method of
the IA algorithm object. This includes generating the channel and setting
the noise variance.
"""

from . import algorithms, iabase
