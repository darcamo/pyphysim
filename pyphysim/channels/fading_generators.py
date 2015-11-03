#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1103
import math
import numpy as np
from ..util.misc import randn_c


# noinspection PyPep8,PyPep8
def generate_jakes_samples(Fd, Ts=1e-3, NSamples=100, L=8, shape=None,
                           RS=None):
    """
    Generates channel samples according to the Jakes model.

    This functions generates channel samples for a single tap according to
    the Jakes model given by

    .. math::
       :label: jakes_model

       h(t) = \\frac{1}{\\sqrt{L}}\\sum_{l=0}^{L-1}\\exp\\{j[2\\pi f_D \\cos(\\phi_l)t+\\psi_l]\\}


    Parameters
    ----------
    Fd : double
        The Doppler frequency (in Hetz).
    Ts : double
        The sample interval (in seconds).
    NSamples : int
        The number of samples to generate.
    L : int
        The number of rays for the Jakes model.
    shape : tuple (of integers)
        The shape of the generated channel. This is used to generate MIMO
        channels. For instance, in order to generate channels samples for a
        MIMO scenario with 3 receive antennas and 2 transmit antennas use a
        shape of (3, 2).
    RS : A numpy.random.RandomState object.
        The RandomState object used to generate the random values. If not
        provided, the global RandomState in numpy will be used.

    Returns
    -------
    h : Numpy array
        The generated channel. If `shape` is None the the shape of the
        returned h is equal to (NSamples,). That is, h is a 1-dimensional
        numpy array. If `shape` was provided then the shape of h is the
        provided shape with an additional dimension for the time (the last
        dimension). For instance, if a `shape` of (3, 2) was provided then
        the shape of the returned h will be (3, 2, NSamples).
    """
    # $h(t) = \frac{1}{\sqrt{L}}\sum_{l=0}^{L-1}\exp\{j[2\pi f_D \cos(\phi_l)t+\psi_l]\}$

    obj = JakesSampleGenerator(Fd, Ts, L, shape, RS)
    return obj.generate_channel_samples(NSamples)


class FadingSampleGenerator(object):
    """Base class for fading generators.
    """

    def __init__(self):
        # Set this variable in a derived class with the next samples
        # everytime the generate_next_samples method is called.
        self._samples = None

    def get_samples(self):
        """Get the last generated sample.
        """
        return self._samples

    def generate_next_samples(self):  # pragma: nocover
        """Generate next samples."""
        raise NotImplementedError("Implement in a subclass")


class RayleighSampleGenerator(FadingSampleGenerator):
    """
    Class that generates a Raleigh fading matrix.

    Parameters
    ----------
    num_rows : int
        Number of rows to create.
    num_cols : int (optional)
        Number of columns. If not provided, then it will be equal to the number
        of rows.
    """

    def __init__(self, num_rows, num_cols=None, ):
        super(RayleighSampleGenerator, self).__init__()
        self._num_rows = num_rows
        self._num_cols = num_cols

        # Generate first sample and set self._H
        self.generate_next_samples()

    def generate_next_samples(self):
        """Generate next samples.
        """
        if self._num_cols is None:
            num_cols = self._num_rows
        else:
            num_cols = self._num_cols
        self._samples = randn_c(self._num_rows, num_cols)


# TODO: Make this class inherit from FadingSampleGenerator and implement
# required interface
# noinspection PyPep8
class JakesSampleGenerator(object):
    """
    The purpose of this class is to generate channel samples according to
    the Jakes model given by

    .. math:: h(t) = \\frac{1}{\\sqrt{L}}\\sum_{l=0}^{L-1}\\exp\\{j[2\\pi f_D \\cos(\\phi_l)t+\\psi_l]\\}

    This class is actually a wrapper to the :meth:`generate_jakes_samples`
    function in this module. Its main purpose is to allow easier usage of
    generate_jakes_samples as well as generating "more samples" continuing
    a previous call to generate_jakes_samples.

    Parameters
    ----------
    Fd : double
        The Doppler frequency (in Hetz).
    Ts : double
        The sample interval (in seconds).
    L : int
        The number of rays for the Jakes model.
    shape : tuple (of integers)
        The shape of the generated channel. This is used to generate MIMO
        channels. For instance, in order to generate channels samples for a
        MIMO scenario with 3 receive antennas and 2 transmit antennas use a
        shape of (3, 2).
    RS : A numpy.random.RandomState object.
        The RandomState object used to generate the random values. If not
        provided, the global RandomState in numpy will be used.

    See also
    --------
    generate_jakes_samples
    """
    # $h(t) = \frac{1}{\sqrt{L}}\sum_{l=0}^{L-1}\exp\{j[2\pi f_D \cos(\phi_l)t+\psi_l]\}$

    def __init__(self, Fd=100, Ts=1e-3, L=8,
                 shape=None, RS=None):
        self.Fd = Fd
        self.Ts = Ts
        self.L = L
        self._shape = shape
        self._phi_l = None  # This will be set in the set_shape method
        self._psi_l = None  # This will be set in the set_shape method

        if RS is None:
            # If RS was not provided, we set it to the numpy.random
            # module. That way, when the rand "method" in RS is called it
            # will actually call the global rand function in numpy.random.
            # RandomState object in numpy.
            RS = np.random
        self.RS = RS

        # self._current_time will be update after each call to the
        # generate_channel_samples method.
        self._current_time = 0.0

        self.shape = shape

    @property
    def shape(self):
        """Get the shape of the JakesSampleGenerator"""
        return self._shape

    @shape.setter
    def shape(self, shape):
        """
        Set the shape of the channel that will be generated by
        `generate_channel_samples`.

        This will also generate `phi_l` and `psi_l` for the new shape.

        Parameters
        ----------
        shape : tuple (of integers)
            The shape of the generated channel. This is used to generate
            MIMO channels. For instance, in order to generate channels
            samples for a MIMO scenario with 3 receive antennas and 2
            transmit antennas use a shape of (3, 2).
        """
        # xxxxx Generate phi_l and psi_l xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self._shape = shape
        if shape is None:
            # The dimension of phi_l and psi_l will be L x 1. We set the last
            # dimensions as 1, instead of setting the dimension of phi_l and
            # psi_l simply as (L,), because it will be broadcasted later by
            # numpy when we multiply with the time.
            self._phi_l = 2 * np.pi * self.RS.rand(self.L, 1)
            self._psi_l = 2 * np.pi * self.RS.rand(self.L, 1)
        else:
            # The dimension of phi_l and psi_l will be L x Shape x 1. We set
            # the last dimensions as 1, instead of setting the dimension of
            # phi_l and psi_l simply as (L,), because it will be broadcasted
            # later by numpy when we multiply with the time.
            new_shape = [self.L]
            new_shape.extend(shape)
            new_shape.append(1)
            self._phi_l = 2 * np.pi * self.RS.rand(*new_shape)
            self._psi_l = 2 * np.pi * self.RS.rand(*new_shape)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _generate_time_samples(self, NSamples):
        """
        Generate the time samples that will be used internally in
        generate_channel_samples method.

        Parameters
        ----------
        NSamples : int
            Number of samples to be generated.

        Returns
        -------
        t : Numpy Array
            The numpy array with the time samples. The shape of the
            generated time variable is "(1, A, NSamples)", where 'A' is has
            as many '1's as the length of self._shape.
            Ex: If self._shape is None then the shape of the returned 't'
            variable is (1, NSamples). If self._shape is (2,3) then the
            shape of the returned 't' variable is (1, 1, 1, NSamples)

        Notes
        -----
        Each time _generate_time_samples is called it will update
        self._current_time to reflect the advance of the time after
        generating the new samples.
        """
        # Generate a 1D numpy with the time samples
        t = np.arange(
            self._current_time,  # Start time
            NSamples * self.Ts + self._current_time,
            self.Ts * 1.0000000001)

        # Update the self._current_time variable with the value of the next
        # time sample that should be generated when _generate_time_samples
        # is called again.
        self._current_time = t[-1] + self.Ts

        # Now we will change the shape of the 't' variable to an
        # appropriated shape for later use.
        if self._shape is not None:
            # Ex: If self._shape is (2,3) then the shape of the generated
            # 't' variable should be (1,1,1,NSNSamples). The first
            # dimension correspond to the number of taps (that is, self.L),
            # the following two dimensions correspond to the dimensions in
            # self._shape, and the last dimension corresponds to the number
            # of time samples.
            #
            # Note that we use '1' for all dimensions except the last one
            # and numpy will replicate to the correct value later thanks to
            # broadcast.
            t.shape = [1] * (len(self._shape) + 1) + [int(NSamples)]
        else:
            # Since self._shape is None, we only need one dimension for the
            # taps (that is, self.L) and another dimension for the actual
            # time samples.
            #
            # Note that we use '1' for all dimensions except the last one
            # and numpy will replicate to the correct value later thanks to
            # broadcast.
            t.shape = (1, NSamples)

        return t

    def generate_channel_samples(self, num_samples):
        """
        Generate more samples for the Jakes model.

        Note that any subsequent call to this method continues from the
        point where the last call stopped. That is, if you generate 10
        samples and then 15 more samples, you will get the same samples you
        would have got if you had generated 25 samples.

        Parameters
        ----------
        num_samples : int
            Number of samples to be generated.

        Returns
        -------
        h : Numpy array
            The generated channel samples. The shape is in the form SHAPE x
            num_samples, where SHAPE is a tuple with the shape provided in the
            constructor of the JakesSampleGenerator class.

        Notes
        -----
        This method will update the self._current_time variable.
        """
        # This method will also update the _current_time member variable
        t = self._generate_time_samples(num_samples)

        # Finally calculate the channel samples
        h = (math.sqrt(1.0 / self.L) *
             np.sum(np.exp(1j * (2 * np.pi * self.Fd
                                 * np.cos(self._phi_l) * t + self._psi_l)),
                    axis=0))
        return h
