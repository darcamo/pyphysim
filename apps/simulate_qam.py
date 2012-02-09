#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of the transmission of QAM symbols through an
awgn channel."""

import simulate_psk
from simulate_psk import PskSimulationRunner
import comm.modulators as mod

from traits.api import Instance, on_trait_change


class QamSimulationRunner(PskSimulationRunner):
    """A SimulationRunner class for a transmission with a PSK modulation
    through an AWGN channel.
    """

    # Because we inherited from PskSimulationRunner, then the modulator
    # trait is originally an instance of mod.PSK. Therefore, we need to
    # change the modulation trait to be an instance of mod.QAM
    modulator = Instance(mod.QAM)

    def __init__(self, config_file_name='qam_simulation_config.txt'):
        """
        """
        PskSimulationRunner.__init__(self, config_file_name)

    @on_trait_change('M')
    def _update_modulator_object(self, ):
        """Updates the modulator object whenever M changes
        """
        self.modulator = mod.QAM(self.M)


# The configuration file is the same as used for the PSK simulation
def write_config_file_template(config_file_name="qam_simulation_config.txt"):
    return simulate_psk.write_config_file_template(config_file_name)


if __name__ == '__main__':
    # UNCOMMENT THE LINE BELOW to create the configuration file. After
    # that, comment the line again and tweak the configuration file.
    #
    # write_config_file_template()

    qam_runner = QamSimulationRunner()
    qam_runner.simulate()

    print "Elapsed Time: {0}".format(qam_runner.elapsed_time)
    print "Iterations Executed: {0}".format(qam_runner.runned_reps)

    #qam_runner.plot_results()
    qam_runner.plot_results_with_chaco()
