#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module with implementation of the Enhanced Block Diagonalization
Algorithm.

In gerenal, the Block Diagonalization (BD) algorithm is applied to an MIMO
Interference Channel (MIMO-IC) scenario, where we have pairs of
transmitters and receivers, each transmitter sending information only to
its intending receiver, but interfering with the other
receivers. Alternativelly, an external interference source may also be
presented, which will interfere with all receivers. In order to model the
MIMO-IC one can use the :class:`.channels.MultiUserChannelMatrix` and
:class:`.channels.MultiUserChannelMatrixExtInt` classes.

The BD algorithm may or may not take the external interference source into
consideration. The BD algorithm is implemented here as the
:class:`EnhancedBD` class and the different external interference handling
metrics are described in the following section.


External Interference Hangling Metrics
--------------------------------------

The way the external interference is treated in the EnhancedBD class
basically consists of sacrificing streams to avoid dimensions strongly
occupied by external interference. In other words, instead of using all
available spatial dimensions only a subset (containing less or no external
interference) of these dimensions is used. One has to decised how many (if
any) dimensions will be sacrificed and for that difference metrics can be
used.

The different metrics implemented in the EnhancedBD class are:

- None: No stream reduction and this external interference handling is
  performed.
- Naive: Stream reduction is performed, but not in any particular
  direction.
- capacity: The Shannon capacity is used.
- effective_throughput: The expected throughput is used. The effective
  throughput consists of the nominal data rate (considering a modulator and
  number of used streams) times 1 minus the package error rate. Sacrificing
  streams will reduce the nominal data rate but the gained interference
  reduction also means better SIRN values and thus a lower package error
  rate.

The usage of the EnhancedBD class is described in the following section.


EnhancedBD usage
----------------

1. First create a EnhancedBD object.
2. Set the desired external interference handling metric by calling the :meth:`.set_ext_int_handling_metric` method.
3. Call the :meth:`.perform_BD_no_waterfilling` method.
"""

__revision__ = "$Revision$"


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__1':  # pragma: no cover
    pass
