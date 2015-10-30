#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1103

"""Module with implementation of channel related classes.

The :class:`MultiUserChannelMatrix` and
:class:`MultiUserChannelMatrixExtInt` classes implement the MIMO
Interference Channel (MIMO-IC) model, where the first one does not include
an external interference source while the last one includes it. The MIMO-IC
model is shown in the Figure below.

.. figure:: /_images/mimo_ic.svg
   :align: center

   MIMO Interference Channel

"""
