#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module that implements Cell related classes."""


from shapes import Coordinate


class CellBaseClass(Coordinate):
    """Base class for all cell types.
    """

    def __init__(self, pos, radius, cell_id=None):
        """
        """
        Coordinate.__init__(self, pos)
        self.id = cell_id
        self.radius = radius  # Should be in a Shape class


if __name__ == '__main__':
    c = CellBaseClass(0, 1)
    print c
