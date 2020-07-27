import numpy as np


def generate_random_points_in_circle(num_points: int,
                                     max_radius: float,
                                     min_radius: float = 0.0) -> np.ndarray:
    """
    Generate `numPoints` points uniformly inside a circle of radius `max_radius`
    and outside a circle with radius `min_radius`

    The circle center is at the origin.

    Parameters
    ----------
    num_points
        The desired number of points
    max_radius
        The circle radius
    min_radius
        The minimum radius

    Returns
    -------
    np.ndarray
        The random points inside the circle.
    """
    radius_all_points = np.sqrt(np.random.random_sample(
        size=num_points)) * (max_radius - min_radius) + min_radius
    angles_all_points = np.random.random_sample(size=num_points) * 2 * np.pi

    return radius_all_points * np.exp(-1j * angles_all_points)


def generate_random_points_in_rectangle(num_points: int, width: float,
                                        height: float):
    """
    Generate `num_points` points uniformly inside a rectangle.

    The rectangle center is at the origin.

    Parameters
    ----------
    num_points
        The desired number of points
    width
        The rectangle width
    height
        The rectangle height

    Returns
    -------
    np.ndarray
        The random points inside the circle.
    """
    return width * (0.5 - np.random.random_sample(size=num_points)) + \
                    1j * height * (0.5 - np.random.random_sample(size=num_points))
