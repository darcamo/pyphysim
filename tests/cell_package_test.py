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

import sys
sys.path.append("../")

from cell import shapes, cell


# def positivy_angle_rad(angle):
#     """Return the positive angle corresponding to `angle`..

#     Arguments:
#     - `angle`: An angle (in radians)

#     """
#     if angle >= 0:
#         return angle
#     else:
#         return 2 * np.pi + angle

# def get_sorted_array_by_angle(x):
#     """Function to return the complex array sorted by the angles
#     ("positivated angles")

#     """
#     angles_rad = np.angle(x)
#     #while any(x < 0):

#     angles_rad = map(positivy_angle_rad, angles_rad)
#     return x[np.argsort(angles_rad)]


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
        # This method was tested in the Hexagon unittests.
        pass

    def test_get_border_point(self):
        # This method uses the _get_vertex_positions method. Therefore, it
        # can only be tested in subclasses of the Shape class.
        #
        # This method was tested in the Hexagon unittests.
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

    def test_get_vertices(self):
        # H1 does not have any translation or rotation. Therefore,
        # _get_vertex_positions should return the same values as the
        # vertices property
        np.testing.assert_array_almost_equal(self.H1._get_vertex_positions(),
                                             self.H1.vertices)

        # H2 has translation.
        np.testing.assert_array_almost_equal(
            self.H2._get_vertex_positions() + complex(2, 3),
            self.H2.vertices)

        # H3 has rotation and translation.
        np.testing.assert_array_almost_equal(
            shapes.Shape._rotate(self.H3._get_vertex_positions(), 30) + 3 + 5j,
            self.H3.vertices)

    def test_is_point_inside_shape(self, ):
        # If the point is exactly in the shape's border, such as the
        # shape's vertexes, then is_point_inside_shape could return either
        # true or false.
        for v in self.H1.vertices:
            # Because the hexagons center is at the origin, we know that if
            # we multiply a vertex by a number lower then one it is
            # guaranteed be inside the shape and vice-verse.
            self.assertTrue(self.H1.is_point_inside_shape(0.9999999 * v))
            self.assertFalse(self.H1.is_point_inside_shape(1.0000001 * v))

        # This test is very imcomplete. If any bugs are found in
        # is_point_inside_shape add tests for them here

    def test_get_border_point(self):
        # Test for an angle of 30 degrees (that is one of the vertexes)
        point_30_degrees = self.H3.get_border_point(30., 1.)
        point_30_degrees_expected = self.H3.pos + self.H3.radius * np.exp(1j * np.pi / 6.)
        self.assertAlmostEqual(point_30_degrees, point_30_degrees_expected)

        # Test for an angle of 30 degrees, but now with a ratio of 0.8
        point_30_degrees_ratio08 = self.H3.get_border_point(30., 0.8)
        point_30_degrees_expected_ratio08 = self.H3.pos + 0.8 * self.H3.radius * np.exp(1j * np.pi / 6.)
        self.assertAlmostEqual(point_30_degrees_ratio08, point_30_degrees_expected_ratio08)

        # Test for an angle of 180 degrees
        point_180_degrees = self.H3.get_border_point(180., 1.)
        point_180_degrees_expected = self.H3.pos - self.H3.height
        self.assertAlmostEqual(point_180_degrees, point_180_degrees_expected)

        # Test for an angle of 180 degrees but with a ratio of 0.5
        point_180_degrees_ratio05 = self.H3.get_border_point(180, 0.5)
        point_180_degrees_expected_ratio05 = self.H3.pos - (self.H3.height / 2.0)
        self.assertAlmostEqual(point_180_degrees_ratio05, point_180_degrees_expected_ratio05)


class RectangleTestCase(unittest.TestCase):
    def test_get_vertex_positions(self):
        A1 = 0 + 0j
        B1 = 1 + 1j
        R1 = shapes.Rectangle(A1, B1)
        # With these coordinates, the central position of R1 is 0.5 + 0.5j
        self.assertAlmostEqual(R1.pos, 0.5 + 0.5j)

        # The rectangle vertexes are then -0.5-0.5j, 0.5-0.5j, 0.5+0.5j and
        # -0.5+0.5j
        np.testing.assert_array_almost_equal(
            R1._get_vertex_positions(),
            np.array([-0.5 - 0.5j, 0.5 - 0.5j, 0.5 + 0.5j, -0.5 + 0.5j]))

        # Note that though A2 and B2 are different from A1 and B1, they are
        # all coordinates of the same rectangle and therefore R2 should
        # have the same position and vertexes as R1.
        A2 = 1 + 0j
        B2 = 0 + 1j
        R2 = shapes.Rectangle(A2, B2)
        self.assertAlmostEqual(R1.pos, R2.pos)
        np.testing.assert_almost_equal(R1._get_vertex_positions(), R2._get_vertex_positions())

        # Now lets create a rectangle with rotation and translates to a
        # different position. The _get_vertex_positions method should not
        # be affected by this
        trans_step = 1.4 - 3j
        R3 = shapes.Rectangle(A1 + trans_step, B1 + trans_step, 30)
        np.testing.assert_almost_equal(R1._get_vertex_positions(), R3._get_vertex_positions())


# TODO: Implement-me
class CircleTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.C1 = shapes.Circle(3 - 5j, 2)

    def test_get_border_point(self):
        C1 = self.C1
        point = C1.get_border_point(90, 1)
        expected_point = C1.pos + 1j * C1.radius
        self.assertAlmostEqual(point, expected_point)

        # from matplotlib import pylab
        # ax = pylab.axes()
        # C1.plot(ax)
        # for angle in np.arange(0, 359, 30):
        #     border_point = C1.get_border_point(angle, 1)
        #     border_point2  = C1.get_border_point(angle, 0.5)
        #     ax.plot(border_point.real, border_point.imag, 'go')
        #     ax.plot(border_point2.real, border_point2.imag, 'bo')
        # pylab.show()

    def test_get_vertex_positions(self):
        num_vertexes = 12
        # The angles need to be in degrees for the get_border_point method
        angles = np.linspace(0,
                             (num_vertexes - 1.) / num_vertexes * 360,
                             num_vertexes)
        expected_vertexes = np.array(
            map(self.C1.get_border_point,
                angles, np.ones(angles.shape))) - self.C1.pos
        vertexes = self.C1._get_vertex_positions()
        np.testing.assert_array_almost_equal(expected_vertexes, vertexes)

    def test_is_point_inside_shape(self):
        # Note that the Shape class version of the is_point_inside_shape
        # method will fail in these tests. That is because it uses the
        # shape's vertexes to determine if a point is inside of the shape
        # of not, which is not good for a circle. The circle class version
        # of the is_point_inside_shape compares the distance from the point
        # to the circle center with the circle radius to determine of the
        # point is inside or outside the circle.

        # This point is inside the circle
        point = self.C1.get_border_point(89, 0.99999999)
        self.assertTrue(self.C1.is_point_inside_shape(point))

        # This point is NOT inside the circle
        point2 = self.C1.get_border_point(89, 1.00000001)
        self.assertFalse(self.C1.is_point_inside_shape(point2))


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
