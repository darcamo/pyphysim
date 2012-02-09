#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of the transmission of PSK symbols through an
awgn channel."""

from traits.etsconfig.etsconfig import ETSConfig
ETSConfig.toolkit = "qt4"

from configobj import ConfigObj

import matplotlib
matplotlib.use('Gtk')
from matplotlib import pyplot as plt

import numpy as np

from simulations import SimulationResults, Result, SimulationRunner, SimulationParameters
from util import misc
from util.conversion import dB2Linear
import comm.modulators as mod
from traits.api import HasTraits, Int, Float, Array, Instance, ListFloat, Property
from traits.api import on_trait_change
from traitsui.api import View, Item, Group, ArrayEditor, ListEditor, TabularEditor, CustomEditor, Action, Handler, Controller, ModelView


# class SimplePskSimulationRunnerHandler(Controller):
#     """
#     """
#     # Handler must be able to observe and manipulate both its corresponding
#     # window and model objects. In Traits UI, this is accomplished by means
#     # of the UIInfo object.  Whenever Traits UI creates a window or panel
#     # from a View, a UIInfo object is created to act as the Handler’s
#     # reference to that window and to the objects whose trait attributes
#     # are displayed in it.
#     def start_simulation(self, UIInfo_object):
#         """Handler method for the simulate action

#         Arguments:
#         - `UIInfo_object`:
#         """
#         # print "haha"
#         # print UIInfo_object.get()
#         runner = UIInfo_object.ui.get()['context']['object']
#         runner.simulate()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SimplePskSimulationRunner - START xxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SimplePskSimulationRunner(SimulationRunner, HasTraits):
    """Implements a simulation runner for a transmission with a PSK
    modulation through an AWGN channel.

    In order to implement a simulation runner, 3 steps are required:
    - The simulation parameters must be added to the self.params variable
    - The _run_simulation funtion must be implemented. It must receive a
      single SimulationParameters object which contains the simulation
      parameters.
    - The _keep_going may be optionally implemented.

    This class also inherits from HasTraits and the simulation parameters
    are traits. Because of this, we can configure the parameters
    graphically by calling the configure_traits method. In addition,
    attributes that depend on other attributes are automatically updated
    using the on_trait_change decorator.
    """

    # Define the traits for the simulation parameters that the user may set
    SNR = Array()
    M = Int()
    NSymbs = Int()
    rep_max = Int()  # rep_max is in the base class. Does traits works?
    max_bit_errors = Int()

    # Define traits for other attributes (that depend on other traits)
    modulator = Instance(mod.PSK)
    params = Instance(SimulationParameters)

    # # Action for when the "Simulate" button is clicked
    # simulate_action = Action(name="Simulate",
    #                          action="start_simulation")

    # Define the view used when the configure_traits method is called
    parameters_view = View(Group(Item('SNR', style='simple', editor=ArrayEditor(), label='SNR'),
                                 Item('M'),
                                 Item('NSymbs'),
                                 Item('max_bit_errors'),
                                 Item('rep_max', style='readonly'),
                                 label='Simulation Parameters',
                                 show_border=False),
                           #handler=SimplePskSimulationRunnerHandler(),
                           #buttons=[simulate_action, 'Cancel', 'Revert'],
                           buttons=['OK', 'Cancel', 'Revert'],
                           resizable=True,
                           # Action if the user closes the window
                           close_result=False)

    def __init__(self):
        # Call the __init__ function of the base classes
        SimulationRunner.__init__(self)
        HasTraits.__init__(self)

        # Set the simulations parameters as attributes here, but what is
        # really necessary is to set the self.params object
        self.SNR = np.array([5, 10, 15])
        self.M = 4
        self.NSymbs = 500
        self.rep_max = 1000

        # We will stop when the number of bit errors is greater than
        # max_bit_errors
        self.max_bit_errors = 200

        # Message Exibited in the progressbar. Set to None to disable the
        # progressbar. See the comment on the SimulationRunner class.
        self.progressbar_message = "{M}-" + \
            self.modulator.__class__.__name__ + " Simulation - SNR: {SNR}"

    @on_trait_change('M')
    def _update_modulator_object(self, ):
        """Updates the modulator object whenever M changes
        """
        self.modulator = mod.PSK(self.M)

    @on_trait_change('SNR, NSymbs, M')
    def _update_params_object(self):
        """Updates the self.params object to the current values of the
        simulation aprameters"""
        # The self.params object must contain all the simulation parameters
        # that will be accessed in the 'simulate' function.
        self.params.add("description", "Parameters for the simulation of a {0}-{1} transmission through an AWGN channel ".format(
                self.M,
                self.modulator.__class__.__name__))
        self.params.add("SNR", self.SNR)
        # Modulation cardinality
        self.params.add("M", self.M)
        # Number of symbols that will be transmitted in the _run_simulation
        # function Unpack the SNR parameter
        self.params.add("NSymbs", self.NSymbs)
        self.params.set_unpack_parameter("SNR")

    def _run_simulation(self, current_parameters):
        # To make sure that this function does not modify the object state,
        # we sobrescibe self to None.
        #self = None

        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        NSymbs = current_parameters["NSymbs"]
        M = current_parameters["M"]
        SNR = current_parameters["SNR"]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        inputData = np.random.randint(0, M, NSymbs)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        modulatedData = self.modulator.modulate(inputData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noiseVar = 1 / dB2Linear(SNR)
        noise = ((np.random.randn(NSymbs) + 1j * np.random.randn(NSymbs)) *
                 np.sqrt(noiseVar / 2))
        receivedData = modulatedData + noise
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        demodulatedData = self.modulator.demodulate(receivedData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the symbol and bit error rates xxxxxxxxxxxxxxxxx
        symbolErrors = sum(inputData != demodulatedData)
        aux = misc.xor(inputData, demodulatedData)
        # Count the number of bits in aux
        bitErrors = sum(misc.bitCount(aux))
        numSymbols = inputData.size
        numBits = inputData.size * mod.level2bits(M)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Return the simulation results xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        symbolErrorsResult = Result.create(
            "symbol_errors", Result.SUMTYPE, symbolErrors)

        numSymbolsResult = Result.create(
            "num_symbols", Result.SUMTYPE, numSymbols)

        bitErrorsResult = Result.create("bit_errors", Result.SUMTYPE, bitErrors)

        numBitsResult = Result.create("num_bits", Result.SUMTYPE, numBits)

        berResult = Result.create("ber", Result.RATIOTYPE, bitErrors, numBits)

        serResult = Result.create(
            "ser", Result.RATIOTYPE, symbolErrors, numSymbols)

        simResults = SimulationResults()
        simResults.add_result(symbolErrorsResult)
        simResults.add_result(numSymbolsResult)
        simResults.add_result(bitErrorsResult)
        simResults.add_result(numBitsResult)
        simResults.add_result(berResult)
        simResults.add_result(serResult)

        return simResults

    def _keep_going(self, simulation_results):
        # Return true as long as cumulated_bit_errors is lower then
        # max_bit_errors
        cumulated_bit_errors = simulation_results['bit_errors'][-1].get_result()
        return cumulated_bit_errors < self.max_bit_errors
# xxxxxxxxxx SimplePskSimulationRunner - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx PskSimulationRunner - START xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class PskSimulationRunner(SimplePskSimulationRunner):
    """A more complete simulation runner for a transmission with a PSK
    modulation through an AWGN channel.

    Some features added to SimplePskSimulationRunner are configuration from
    a config file and plot of the results.
    """

    def __init__(self, config_file_name='psk_simulation_config.txt'):
        """
        """
        SimplePskSimulationRunner.__init__(self)

        # Read the configuration file
        conf_file_parser = ConfigObj(config_file_name, list_values=True)

        # Read the simulation parameters
        rep_max = int(conf_file_parser['Simulation']['rep_max'])
        M = int(conf_file_parser['Simulation']['M'])
        NSymbs = int(conf_file_parser['Simulation']['NSymbs'])
        SNR = conf_file_parser['Simulation']['SNR']
        SNR = [int(i) for i in SNR]
        max_bit_errors = int(conf_file_parser['Simulation']['max_bit_errors'])

        # Set the simulations parameters from the configuration file
        self.SNR = SNR
        self.M = M
        self.NSymbs = NSymbs
        self.rep_max = rep_max
        self.max_bit_errors = max_bit_errors

    def get_data_to_be_plotted(self):
        def get_result_value(index, param_name):
            return self.results[index][param_name][0].get_result()

        # xxxxx Concatenate the simulation results xxxxxxxxxxxxxxxxxxxxxxxx
        N_SNR_values = len(self.results)
        ber = np.zeros(N_SNR_values)
        ser = np.zeros(N_SNR_values)
        for i in range(0, N_SNR_values):
            ber[i] = get_result_value(i, 'ber')
            ser[i] = get_result_value(i, 'ser')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Get the SNR from the simulation parameters
        SNR = np.array(self.params['SNR'])
        # Number of bits per symbol
        k = mod.level2bits(self.modulator.M)
        Eb_over_N0 = SNR - 10 * np.log10(k)

        # Calculates the Theoretical SER and BER
        theoretical_ser = self.modulator.calcTheoreticalSER(SNR)
        theoretical_ber = self.modulator.calcTheoreticalBER(SNR)
        return (SNR, Eb_over_N0, ber, ser, theoretical_ber, theoretical_ser)

    def plot_results(self):
        """Plot the results from the simulation, as well as the
        theoretical results.
        """
        # def make_patch_spines_invisible(ax):
        #     ax.set_frame_on(True)
        #     ax.patch.set_visible(False)
        #     for sp in ax.spines.itervalues():
        #         sp.set_visible(False)
        SNR, Eb_over_N0, ber, ser, theoretical_ber, theoretical_ser = self.get_data_to_be_plotted()

        # xxxxx Plot the results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        f = plt.figure()
        ax = f.add_subplot(211)
        ax2 = f.add_subplot(212)

        line1 = ax.plot(SNR, ber, '--o', color='green', label='Simulated BER')
        line2 = ax.plot(SNR, ser, '--^', color='blue', label='Simulated SER')
        ax2.plot(Eb_over_N0, ber, '--o', color='green', label='Simulated BER')
        ax2.plot(Eb_over_N0, ser, '--^', color='blue', label='Simulated SER')

        line3 = ax.plot(SNR, theoretical_ber, color='green', label='Theoretical BER')
        line4 = ax.plot(SNR, theoretical_ser, color='blue', label='Theoretical SER')
        ax2.plot(Eb_over_N0, theoretical_ber, color='green', label='Theoretical BER')
        ax2.plot(Eb_over_N0, theoretical_ser, color='blue', label='Theoretical SER')
        # Not really necessary. Just to make flymake happy in emacs
        line1 + line2 + line3 + line4

        # xxxxx Set the properties of the ax axes xxxxxxxxxxxxxxxxxxxxxxxxx
        # Uses the label property of each line as the legend, since I'm not
        # specifying the legend here
        ax.legend()
        title_string = '{0}-{1} Simulation'.format(
            self.modulator.M,
            self.modulator.__class__.__name__)
        ax.set_title(title_string)
        ax.set_xlabel('SNR')
        ax.set_ylabel('Error Rate')
        ax.set_yscale('log')
        ax.axis('tight')
        ax.grid(True, which='both')

        # xxxxx Set the properties of the ax2 axes xxxxxxxxxxxxxxxxxxxxxxxx
        ax2.legend()
        # title_string = '{0}-{1} Simulation'.format(
        #     self.modulator.M,
        #     self.modulator.__class__.__name__)
        # ax2.set_title(title_string)
        ax2.set_xlabel('Eb/N0')
        ax2.set_ylabel('Error Rate')
        ax2.set_yscale('log')
        ax2.axis('tight')
        ax2.grid(True, which='both')

        #f.show()
        plt.show()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def plot_results_with_chaco_shell(self):
        SNR, Eb_over_N0, ber, ser, theoretical_ber, theoretical_ser = self.get_data_to_be_plotted()

        from chaco.shell import hold, semilogy, legend, tool, show
        semilogy(SNR, ber, '-r', name='BER')
        hold(True)
        semilogy(SNR, ser, name='SER')
        legend(True)
        tool()
        show()

    def plot_results_with_chaco(self):
        from plot.simulationresultsplotter import SimulationResultsPlotter
        results_plotter = SimulationResultsPlotter(self)
        results_plotter.configure_traits()
# xxxxxxxxxx PskSimulationRunner - END xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def write_config_file_template(config_file_name="psk_simulation_config.txt"):
    """Write a configuration file that can be used to run the simulate
    function of a PskSimulationRunner object.
    """
    # See http://www.voidspace.org.uk/python/configobj.html#getting-started
    configobj = ConfigObj(config_file_name)
    configobj.clear()
    configobj.initial_comment = ["Simulation Parameters"]

    # xxxxx Creates a section for the simulation parameters xxxxxxxxxxxxxxx
    configobj['Simulation'] = {}

    # xxxxx Simulation parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    configobj['Simulation']['SNR'] = range(0, 19, 3)
    configobj['Simulation']['M'] = 4
    configobj['Simulation']['NSymbs'] = 5000
    configobj['Simulation']['rep_max'] = 20000
    configobj['Simulation']['max_bit_errors'] = 200

    # xxxxx Comments for each simulation parameters xxxxxxxxxxxxxxxxxxxxxxx
    # Get the simulation section
    s = configobj['Simulation']
    # All comments are lists of lines. We just need to append new lines
    s.comments['SNR'].extend(['', "SNR Values"])
    s.comments['M'].extend(['', "Modulation Cardinality"])
    s.comments['NSymbs'].extend(['', "Number of Symbols transmitted in each iteration"])
    s.comments['rep_max'].extend(['', "Maximum Number of iterations (Simulation stops after rep_max or", "max_bit_errors is reached)"])
    s.comments['max_bit_errors'].extend(['', "Maximum Number of bit Errors (Simulation stops after rep_max or", "max_bit_errors is reached)"])

    configobj.write()


# xxxxxxxxxx MAIN xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    # UNCOMMENT THE LINE BELOW to create the configuration file. After
    # that, comment the line again and tweak the configuration file.
    #
    # write_config_file_template()

    psk_runner = PskSimulationRunner()
    psk_runner.simulate()

    print "Elapsed Time: {0}".format(psk_runner.elapsed_time)
    print "Iterations Executed: {0}".format(psk_runner.runned_reps)

    # psk_runner.plot_results()
    psk_runner.plot_results_with_chaco()
    #psk_runner.plot_results_with_chaco_shell()

    #results_plotter = psk_runner.results_plotter


if __name__ == '__main__1':
    #runner = SimplePskSimulationRunner(250)
    runner = PskSimulationRunner()
    runner.configure_traits()
