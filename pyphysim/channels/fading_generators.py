#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1103
import math
import numpy as np
from collections import Iterable
from ..util.misc import randn_c


# noinspection PyPep8
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
    obj.generate_more_samples(NSamples)
    return obj.get_samples()


class FadingSampleGenerator(object):
    """
    Base class for fading generators.

    Parameters
    ----------
    shape : tuple (of integers) or an int

        The shape of the sample generator. Each time
        `generate_more_samples(num_samples)` method is called it will
        generate samples with this shape as the first dimensions.
    """

    def __init__(self, shape=None):
        self._shape = shape

        # Set this variable in a derived class with the next samples
        # everytime the generate_more_samples method is called. Note that
        # generate_more_samples should take the value of self._shape into
        # account.
        self._samples = None

    @property
    def shape(self):
        """
        Get the shape of the sampling generator

        This is the shape of the samples that will be generated (not
        including num_samples).
        """
        if self._shape is not None and not isinstance(self._shape, Iterable):
            shape = (self._shape, )
        else:
            shape = self._shape

        return shape

    @shape.setter
    def shape(self, value):
        """
        Set the shape of the sampling generator.

        This is the shape of the samples that will be generated (not
        including num_samples).

        Parameters
        ----------
        shape : tuple of integers, an int, or None
            The shape of the generated channel.
        """
        self._shape = value

    def get_samples(self):
        """Get the last generated sample.
        """
        return self._samples

    def generate_more_samples(self, num_samples=None):  # pragma: nocover
        """
        Generate next samples.

        When implementing this method in a subclass you must take the value
        of the self._shape attribute into account.

        Parameters
        ----------
        num_samples : int (optional)
            Number of samples (with the provided shape) to generate. If not
            provided it will be assumed to be 1.
        """
        raise NotImplementedError("Implement in a subclass")

    def get_similar_fading_generator(self):
        """
        Get a similar fading generator with the same configuration, but that
        generates independent samples.
        """
        # Note: Don't forget to copy self._shape in sublcasses, besides any
        # member attribute in the subclass
        raise NotImplementedError("Implement in a subclass")


class RayleighSampleGenerator(FadingSampleGenerator):
    """
    Class that generates fading samples from a Raleigh distribution.

    Parameters
    ----------
    shape : tuple (of integers) or an int
        The shape of the sample generator. Each time the
        `generate_jakes_samples` method is called it will generate samples
        with this shape. If not provided, then 1 will be assumed.
    """

    def __init__(self, shape=None):
        super(RayleighSampleGenerator, self).__init__(shape)

        # Generate first sample
        self.generate_more_samples()

    def generate_more_samples(self, num_samples=None):
        """
        Generate next samples.

        Parameters
        ----------
        num_samples : int (optional)
            Number of samples (with the provided shape) to generate. If not
            provided it will be assumed to be 1.
        """
        shape = self.shape

        if num_samples is None:
            if shape is None:
                self._samples = randn_c()
            else:
                self._samples = randn_c(*shape)
        elif self.shape is None:
            self._samples = randn_c(num_samples)
        else:
            shape = list(shape)
            shape.append(num_samples)
            self._samples = randn_c(*shape)

    def get_similar_fading_generator(self):
        """
        Get a similar fading generator with the same configuration, but that
        generates independent samples.
        """
        return RayleighSampleGenerator(self._shape)


# TODO: Remove the RS parameter or make it work with the
# get_similar_fading_generator method.  You could also move it to the base
# class and add it as an argument to RayleighSampleGenerator
class JakesSampleGenerator(FadingSampleGenerator):
    """
    Class that generated fading samples according to the Jakes model given
    by

    .. math:: h(t) = \\frac{1}{\\sqrt{L}}\\sum_{l=0}^{L-1}\\exp\\{j[2\\pi f_D \\cos(\\phi_l)t+\\psi_l]\\}

    Parameters
    ----------
    Fd : double
        The Doppler frequency (in Hetz).
    Ts : double
        The sample interval (in seconds).
    L : int
        The number of rays for the Jakes model.
    shape : tuple (of integers) or an int
        The shape of the sample generator. Each time the
        `generate_jakes_samples` method is called it will generate samples
        with this shape. If not provided, then 1 will be assumed. This
        could be used to generate MIMO channels. For instance, in order to
        generate channels samples for a MIMO scenario with 3 receive
        antennas and 2 transmit antennas use a shape of (3, 2).
    RS : A numpy.random.RandomState object.
        The RandomState object used to generate the random values. If not
        provided, the global RandomState in numpy will be used.

    See also
    --------
    generate_jakes_samples
    """

    def __init__(self, Fd=100, Ts=1e-3, L=8, shape=None, RS=None):
        super(JakesSampleGenerator, self).__init__(shape)

        self._Fd = Fd
        self._Ts = Ts
        self._L = L

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
        # `generate_more_samples` method.
        self._current_time = 0.0

        # Updateself._phi_l and self._psi_l according to self._shape
        self._set_phi_and_psi_according_to_shape()

        # Generate first sample
        self.generate_more_samples()

    @property
    def shape(self):
        """
        Get the shape of the sampling generator

        This is the shape of the samples that will be generated (not
        including num_samples).
        """
        return super(JakesSampleGenerator, self).shape

    @shape.setter
    def shape(self, value):
        """
        Set the shape of the sampling generator.

        This is the shape of the samples that will be generated (not
        including num_samples).

        Parameters
        ----------
        shape : tuple of integers, an int, or None
            The shape of the generated channel.
        """
        self._shape = value
        # Since phi and psi depend on the shape we need to update
        # them. Note that `_set_phi_and_psi_according_to_shape` will use
        # the value of self._shape
        self._set_phi_and_psi_according_to_shape()

    @property
    def L(self):
        return self._L

    @property
    def Ts(self):
        return self._Ts

    @property
    def Fd(self):
        return self._Fd

    def _set_phi_and_psi_according_to_shape(self):
        """
        This will update the phi and psi attributes used to generate the jakes
        samples to reflect the current value of self._shape.
        """
        if self.shape is None:
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
            new_shape.extend(self.shape)
            new_shape.append(1)
            self._phi_l = 2 * np.pi * self.RS.rand(*new_shape)
            self._psi_l = 2 * np.pi * self.RS.rand(*new_shape)

    def _generate_time_samples(self, num_samples=None):
        """
        Generate the time samples that will be used internally in
        `generate_more_samples` method.

        Parameters
        ----------
        num_samples : int
            Number of samples to be generated.

        Returns
        -------
        Numpy Array
            The numpy array with the time samples. The shape of the
            generated time variable is "(1, A, num_samples)", where 'A' has
            as many '1's as the length of self._shape.
            Ex: If self._shape is None then the shape of the returned 't'
            variable is (1, num_samples). If self._shape is (2,3) then the
            shape of the returned 't' variable is (1, 1, 1, num_samples)

        Notes
        -----
        Each time `_generate_time_samples` is called it will update
        `_current_time` to reflect the advance of the time after
        generating the new samples.
        """
        if num_samples is None:
            num_samples = 1

        # Generate a 1D numpy with the time samples
        t = np.arange(
            self._current_time,  # Start time
            num_samples * self.Ts + self._current_time,
            self.Ts * 1.0000000001)

        # Update the self._current_time variable with the value of the next
        # time sample that should be generated when _generate_time_samples
        # is called again.
        self._current_time = t[-1] + self.Ts

        # Now we will change the shape of the 't' variable to an
        # appropriated shape for later use.
        if self._shape is not None:
            # Ex: If self._shape is (2,3) then the shape of the generated
            # 't' variable should be (1,1,1,NSnum_samples). The first
            # dimension correspond to the number of taps (that is, self.L),
            # the following two dimensions correspond to the dimensions in
            # self._shape, and the last dimension corresponds to the number
            # of time samples.
            #
            # Note that we use '1' for all dimensions except the last one
            # and numpy will replicate to the correct value later thanks to
            # broadcast.
            t.shape = [1] * (len(self._shape) + 1) + [int(num_samples)]
        else:
            # Since self._shape is None, we only need one dimension for the
            # taps (that is, self.L) and another dimension for the actual
            # time samples.
            #
            # Note that we use '1' for all dimensions except the last one
            # and numpy will replicate to the correct value later thanks to
            # broadcast.
            t.shape = (1, num_samples)

        return t

    def generate_more_samples(self, num_samples=None):
        """
        Generate next samples.

        Note that any subsequent call to this method continues from the
        point where the last call stopped. That is, if you generate 10
        samples and then 15 more samples, you will get the same samples you
        would have got if you had generated 25 samples.

        Parameters
        ----------
        num_samples : int (optional)
            Number of samples (with the provided shape) to generate. If not
            provided it will be assumed to be 1.

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
        self._samples = h

    def get_similar_fading_generator(self):
        """
        Get a similar fading generator with the same configuration, but that
        generates independent samples.
        """
        return JakesSampleGenerator(self._Fd, self._Ts, self._L, self._shape)
