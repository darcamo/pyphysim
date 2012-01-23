#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of the transmission of QAM symbols through an
awgn channel."""

import simulate_psk
from simulate_psk import PskSimulationRunner
import comm.modulators as mod


class QamSimulationRunner(PskSimulationRunner):
    """A SimulationRunner class for a transmission with a PSK modulation
    through an AWGN channel.
    """

    def __init__(self, config_file_name='qam_simulation_config.txt'):
        """
        """
        PskSimulationRunner.__init__(self, config_file_name)

        M = self.params['M']
        self.modulator = mod.QAM(M)
        self.params.add("description", "Parameters for the simulation of a {0}-QAM transmission through an AWGN channel ".format(M))
        self.progressbar_message = "{M}-QAM Simulation - SNR: {SNR}"


def write_config_file_template(config_file_name="qam_simulation_config.txt"):
    """
    """
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

    qam_runner.plot_results()
