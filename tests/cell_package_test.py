#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the cell package.

Each module has several doctests that we run in addition to the unittests
defined here.

"""

# TODO: Implement all the tests for the cell package.

import unittest
import doctest
import numpy as np

from cell import shapes, cell


# UPDATE THIS CLASS if another module is added to the comm package
class CellDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the cell
    package.
    """

    def test_shapes(self):
        """Run doctests in the shapes module.
        """
        doctest.testmod(shapes)

    def test_cell(self):
        """Run doctests in the cell module.
        """
        doctest.testmod(cell)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SHAPES module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class CoordinateTestCase(unittest.TestCase):
    def test_calc_dist(self):
        C1 = shapes.Coordinate(0 + 3j)
        C2 = shapes.Coordinate(2 - 4j)
        C3 = shapes.Coordinate(5 - 0j)

        # Sanity check
        self.assertEqual(C1.calc_dist(C2), C2.calc_dist(C1))

        self.assertAlmostEqual(np.sqrt((2 ** 2) + (7 ** 2)), C1.calc_dist(C2))
        self.assertAlmostEqual(np.sqrt((5 ** 2) + (3 ** 2)), C1.calc_dist(C3))
        self.assertAlmostEqual(np.sqrt((3 ** 2) + (4 ** 2)), C2.calc_dist(C3))


class ShapeTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        #         shapes.Shape(pos, radius, rotation)
        self.S1 = shapes.Shape(0 + 0j, 1.5, 0)
        self.S2 = shapes.Shape(2 + 3j, 2, 0)
        self.S3 = shapes.Shape(3 + 5j, 1.2, 30)

    def test_radius_property(self):
        self.assertAlmostEqual(self.S1.radius, 1.5)
        self.assertAlmostEqual(self.S2.radius, 2.0)
        self.assertAlmostEqual(self.S3.radius, 1.2)
        self.S1.radius = 3.3
        self.assertAlmostEqual(self.S1.radius, 3.3)

    def test_rotation_property(self):
        self.assertAlmostEqual(self.S1.rotation, 0)
        self.assertAlmostEqual(self.S3.rotation, 30)
        self.S1.rotation = 25
        self.assertAlmostEqual(self.S1.rotation, 25)

    def test_get_vertex_positions(self):
        # This method must be implemented in subclasses and in the Shape
        # class it should only raise an exception.
        self.assertRaises(NotImplementedError, self.S1._get_vertex_positions)

    def test_is_point_inside_shape(self):
        # This method uses the _get_vertex_positions method. Therefore, it
        # can only be tested in subclasses of the Shape class.
        #
        # TODO: After implementing a test for this method in a subclass
        # write here in place of this comment the name f the subclass where
        # this method is tested.
        pass

    def test_get_border_point(self):
        # This method uses the _get_vertex_positions method. Therefore, it
        # can only be tested in subclasses of the Shape class.
        #
        # TODO: After implementing a test for this method in a subclass
        # write here in place of this comment the name f the subclass where
        # this method is tested.
        pass

    def test_rotate(self):
        """Test the static method Shape._rotate
        """
        cur_pos = 1 - 2j
        self.assertAlmostEqual(shapes.Shape._rotate(cur_pos, 90), 2 + 1j)
        self.assertAlmostEqual(shapes.Shape._rotate(cur_pos, 180), -1 + 2j)


class HexagonTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.H1 = shapes.Hexagon(0 + 0j, 1.5, 0)
        self.H2 = shapes.Hexagon(2 + 3j, 2, 0)
        self.H3 = shapes.Hexagon(3 + 5j, 1.5, 30)

    def test_height_property(self):
        # The height of an Hexagon is given by sqrt(3) * side_length, where
        # the side of the Hexagon is the radius property of the Shape.
        self.assertAlmostEqual(self.H1.height,
                               np.sqrt(3.0) * self.H1.radius / 2.0)
        self.assertAlmostEqual(self.H2.height,
                               np.sqrt(3.0) * self.H2.radius / 2.0)
        self.assertAlmostEqual(self.H3.height,
                               np.sqrt(3.0) * self.H3.radius / 2.0)

    def test_get_vertex_positions(self):
        # TODO: Implement-me
        pass
        # Even though H1 and H2 have different position and rottion, they
        # have the same radius and since _get_vertex_positions does not
        # account translation and rotation then the _get_vertex_positions
        # method should return the same value for both of them.
        np.testing.assert_array_equal(self.H1._get_vertex_positions(),
                                      self.H3._get_vertex_positions())

        expected_vertexes = 1.5 * np.exp(
            1j * np.pi * np.linspace(0, 300, 6) / 180.)
        # Lets sort the expected vertexes (regarding the angle of the
        # vertexes)
        expected_vertexes = expected_vertexes[np.argsort(np.angle(expected_vertexes))]

        # The way the obtained_vertexes are calculated the vertexes are
        # already in ascending order regarding their angle.
        obtained_vertexes = self.H1._get_vertex_positions()

        np.testing.assert_array_almost_equal(expected_vertexes,
                                             obtained_vertexes)

        expected_vertexes2 = (2. / 1.5) * expected_vertexes
        obtained_vertexes2 = self.H2._get_vertex_positions()
        np.testing.assert_array_almost_equal(expected_vertexes2,
                                             obtained_vertexes2)

    def test_is_point_inside_shape(self, ):
        # TODO: Implement-me
        pass

    def test_get_border_point(self):
        # TODO: Implement-me
        pass


# TODO: Implement-me
class RectangleTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass


# TODO: Implement-me
class CircleTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass


class ShapesModuleMethodsTestCase(unittest.TestCase):
    def test_from_complex_array_to_real_matrix(self, ):
        pass
        # TODO: Implement-me

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx CELL module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class CellTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_some_method(self):
        pass
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
