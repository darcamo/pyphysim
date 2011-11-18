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


# import unittest
# from googlemaps import GoogleMaps

# class Test(unittest.TestCase):
#     """Unit tests for googlemaps."""

#     def test_local_search(self):
#         """Test googlemaps local_search()."""
#         gmaps = GoogleMaps(GMAPS_API_KEY,
#                            referrer_url='http://www.google.com/')
#         local = gmaps.local_search('sushi san francisco, ca')
#         result = local['responseData']['results'][0]
#         self.assertEqual(result['titleNoFormatting'], 'Sushi Groove')

# if __name__ == "__main__":
#     unittest.main()
