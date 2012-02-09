#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define classes to plot the results in the apps folder using the chaco
plot library"""

# Some Code to tell the traits framework to use Qt4, instead of
# Wxwidgets
# import os
# os.environ['ETS_TOOLKIT'] = 'qt4'

# import sip
# sip.setapi('QString', 2)
# sip.setapi('QVariant', 2)
#from PyQt4 import QtGui, QtCore

from traits.etsconfig.etsconfig import ETSConfig
ETSConfig.toolkit = "qt4"

# Numpy imports
import numpy as np

# Traits and traitsui imports
from traits.api import HasTraits, Instance, Int, Range
from traitsui.api import View, Item, Group

# Chaco imports
from chaco.plot_factory import _create_data_sources
from chaco.api import Plot, ArrayPlotData, marker_trait, ArrayDataSource, OverlayPlotContainer, create_line_plot
from chaco.tools.api import PanTool, ZoomTool, BroadcasterTool, LegendTool

# Enable imports
from enable.component_editor import ComponentEditor

from enable.api import MarkerTrait, black_color_trait
from traits.api import Float, Any
from chaco.api import LinePlot
from chaco.scatterplot import render_markers


# xxxxx Chaco Renderer for a line plot with markers xxxxxxxxxxxxxxxxxxxxxxx
# Renderer class for a line plot with markers
# https://github.com/sergey-miryanov/chaco/blob/master/chaco/ScatterLinePlot.py
class ScatterLinePlot(LinePlot):
    marker = MarkerTrait
    marker_size = Float(2.0)
    outline_color = black_color_trait
    custom_symbol = Any

    def _gather_points(self):
        super(ScatterLinePlot, self)._gather_points()

    def _render_markers(self, gc, points):
        render_markers(gc, points, self.marker,
                self.marker_size, self.color_, self.line_width,
                self.outline_color_,
                self.custom_symbol)

    def _render(self, gc, points, selected_points=None):
        super(ScatterLinePlot, self)._render(gc, points, selected_points)
        if len(points) > 0:
            for pts in points:
                self._render_markers(gc, pts)

    def _render_icon(self, gc, x, y, width, height):
        super(ScatterLinePlot, self)._render_icon(gc, x, y, width, height)
        point = np.array([x + width / 2, y + height / 2])
        self._render_markers(gc, [point])
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class SimulationResultsPlotter(HasTraits):
    """Class to plot the results with chaco.
    """
    plot = Instance(Plot)
    # Traits can have a description, the 'desc' argument
    marker = marker_trait(desc='Marker type of the plot')
    marker_size = Range(low=1, high=6, value=4, desc='Marker size of the plot')

    # If no view is specified when configure_traits() is called on
    # the SimulationResultsPlotter object, then the one named traits_view takes
    # preference
    traits_view = View(
        Group(
            Item('plot', editor=ComponentEditor(), show_label=False),
            Item('marker', label='Marker'),
            Item('marker_size', label='Size'),
            orientation='vertical'
        ),
        width=500,
        height=500,
        resizable=True,
        title="Chaco Plot")

    def __init__(self, simulation_runner):
        super(SimulationResultsPlotter, self).__init__()
        self.simulation_runner = simulation_runner

        # Get the data to be plotted from the simulation runner object
        SNR, Eb_over_N0, ber, ser, theoretical_ber, theoretical_ser = simulation_runner.get_data_to_be_plotted()

        plotdata = ArrayPlotData(SNR=SNR,
                                 Eb_over_N0=Eb_over_N0,
                                 ber=ber,
                                 ser=ser,
                                 theoretical_ber=theoretical_ber,
                                 theoretical_ser=theoretical_ser
                                 )

        # Create the plot object
        self.plot = Plot(plotdata)

        # Simulated values
        # self.plot.plot(
        #     ("SNR", "ber"),
        #     type="line",
        #     color="green",
        #     name='Simulated BER',
        #     value_scale='log')[0]
        self.ber_plot_renderer = self.plot.plot(
            ("SNR", "ber"),
            type="scatter",
            color="green",
            name='Simulated BER',
            value_scale='log')[0]

        # self.plot.plot(
        #     ("SNR", "ser"),
        #     type="line",
        #     color="blue",
        #     name='Simulated SER',
        #     value_scale='log')[0]
        self.ser_plot_renderer = self.plot.plot(
            ("SNR", "ser"),
            type="scatter",
            color="blue",
            name='Simulated SER',
            value_scale='log')[0]

        # Theoretical BER
        theoretical_ber_plot_renderer = self.plot.plot(
            ("SNR", "theoretical_ber"),
            type="line",
            color="green",
            name='Theoretical BER',
            value_scale='log',
            line_style='dash')[0]
        # Theoretical SER
        theoretical_ser_plot_renderer = self.plot.plot(
            ("SNR", "theoretical_ser"),
            type="line",
            color="green",
            name='Theoretical SER',
            value_scale='log',
            line_style='dash')[0]

        # Turn on the plot legend
        self.plot.legend.visible = True
        self.plot.legend.tools.append(
            LegendTool(self.plot.legend, drag_button="right"))

        # Change y scale to log
        title_string = '{0}-{1} Simulation'.format(
            self.simulation_runner.modulator.M,
            self.simulation_runner.modulator.__class__.__name__)
        self.plot.title = title_string

        # Enable grid lines
        self.plot.x_grid.set(True)
        self.plot.y_grid.set(True)

        # Set the y and x axis
        self.plot.y_axis.title = 'Error Rate'
        self.plot.x_axis.title = 'SNR'

        # Increase the size of the tick labels and axes labels
        self.plot.x_axis.tick_label_font.size = 13
        self.plot.y_axis.tick_label_font.size = 13
        self.plot.x_axis.title_font.size = 15
        self.plot.y_axis.title_font.size = 15

        def _get_logscale_tick_string(tick):
            """Function that receives the tick values and return a
            string representation suitable to a logscale axis.

            Arguments:
            - `tick`: tick numerical value
            """
            floor_power = np.floor(np.log10(tick))
            index = int(np.round(tick * (10 ** (-floor_power))))
            power = int(floor_power)
            if index == 1:
                return "10^%s" % power
            else:
                return ""

        self.plot.y_axis.tick_label_formatter = _get_logscale_tick_string

    def _marker_size_changed(self):
        self.ber_plot_renderer.marker_size = self.marker_size
        self.ser_plot_renderer.marker_size = self.marker_size

    def _marker_changed(self):
        self.ber_plot_renderer.marker = self.marker
        self.ser_plot_renderer.marker = self.marker


class DummyPlotter(HasTraits):
    """Class to plot the results with chaco.
    """
    plot = Instance(OverlayPlotContainer)
    # Traits can have a description, the 'desc' argument
    marker = marker_trait(desc='Marker type of the plot')
    marker_size = Range(low=1, high=6, value=4, desc='Marker size of the plot')

    # If no view is specified when configure_traits() is called on
    # the SimulationResultsPlotter object, then the one named traits_view takes
    # preference
    traits_view = View(
        Group(
            Item('plot', editor=ComponentEditor(), show_label=False),
            Item('marker', label='Marker'),
            Item('marker_size', label='Size'),
            orientation='vertical'
        ),
        width=500,
        height=500,
        resizable=True,
        title="Chaco Plot")

    def __init__(self, simulation_runner):
        super(DummyPlotter, self).__init__()
        self.simulation_runner = simulation_runner

        # # Get the data to be plotted from the simulation runner object
        # SNR, Eb_over_N0, ber, ser, theoretical_ber, theoretical_ser = simulation_runner.get_data_to_be_plotted()

        # plotdata = ArrayPlotData(SNR=SNR,
        #                          Eb_over_N0=Eb_over_N0,
        #                          ber=ber,
        #                          ser=ser,
        #                          theoretical_ber=theoretical_ber,
        #                          theoretical_ser=theoretical_ser
        #                          )

        # # Create the plot object
        # self.plot = Plot(plotdata)

        # # Simulated values
        # # self.plot.plot(
        # #     ("SNR", "ber"),
        # #     type="line",
        # #     color="green",
        # #     name='Simulated BER',
        # #     value_scale='log')[0]
        # self.ber_plot_renderer = self.plot.plot(
        #     ("SNR", "ber"),
        #     type="scatter",
        #     color="green",
        #     name='Simulated BER',
        #     value_scale='log')[0]

        # # self.plot.plot(
        # #     ("SNR", "ser"),
        # #     type="line",
        # #     color="blue",
        # #     name='Simulated SER',
        # #     value_scale='log')[0]
        # self.ser_plot_renderer = self.plot.plot(
        #     ("SNR", "ser"),
        #     type="scatter",
        #     color="blue",
        #     name='Simulated SER',
        #     value_scale='log')[0]

        # # Theoretical BER
        # theoretical_ber_plot_renderer = self.plot.plot(
        #     ("SNR", "theoretical_ber"),
        #     type="line",
        #     color="green",
        #     name='Theoretical BER',
        #     value_scale='log',
        #     line_style='dash')[0]
        # # Theoretical SER
        # theoretical_ser_plot_renderer = self.plot.plot(
        #     ("SNR", "theoretical_ser"),
        #     type="line",
        #     color="green",
        #     name='Theoretical SER',
        #     value_scale='log',
        #     line_style='dash')[0]

        # # Turn on the plot legend
        # self.plot.legend.visible = True

        # # Change y scale to log
        # title_string = '{0}-{1} Simulation'.format(
        #     self.simulation_runner.modulator.M,
        #     self.simulation_runner.modulator.__class__.__name__)
        # self.plot.title = title_string

        # # Enable grid lines
        # self.plot.x_grid.set(True)
        # self.plot.y_grid.set(True)

        # # Set the y and x axis
        # self.plot.y_axis.title = 'Error Rate'
        # self.plot.x_axis.title = 'SNR'

        # # Increase the size of the tick labels and axes labels
        # self.plot.x_axis.tick_label_font.size = 13
        # self.plot.y_axis.tick_label_font.size = 13
        # self.plot.x_axis.title_font.size = 15
        # self.plot.y_axis.title_font.size = 15

        # self.plot.y_axis.tick_label_formatter = self.__class__._get_logscale_tick_string

    @staticmethod
    def _get_logscale_tick_string(tick):
        """Function that receives the tick values and return a
        string representation suitable to a logscale axis.

        This function will be set as the y_axis.tick_label_formatter of the
        plot so that the y_axis tick labels follow MATLAB style.

        Arguments:
        - `tick`: tick numerical value of the tick
        """
        floor_power = np.floor(np.log10(tick))
        index = int(np.round(tick * (10 ** (-floor_power))))
        power = int(floor_power)
        if index == 1:
            return "10^%s" % power
        else:
            return ""

    def _make_curves(self):
        # SNR, Eb_over_N0, ber, ser, theoretical_ber, theoretical_ser = self.simulation_runner.get_data_to_be_plotted()

        # Get Dummy data
        from comm.modulators import PSK
        SNR = np.arange(0, 20, 3)

        psk = PSK(16)
        SER = psk.calcTheoreticalSER(SNR)
        BER = psk.calcTheoreticalBER(SNR)

        # SNR_ds = ArrayDataSource(SNR)
        # BER_ds = ArrayDataSource(BER)
        # SER_ds = ArrayDataSource(SER)

        # Create the curves here and return them
        # ber_plot_line = LinePlot(index=SNR_ds,
        #                          value=BER_ds)
        # ser_plot_line = LinePlot(index=SNR_ds,
        #                          value=SER_ds)
        ber_plot_line = create_line_plot((SNR, BER), color='green')
        ser_plot_line = create_line_plot((SNR, SER), color='blue')

        return (ber_plot_line, ser_plot_line)

    def _create_plot_component(self):
        """Creates the OverlayPlotContainer"""
        container = OverlayPlotContainer(padding=40, bgcolor="lightgray",
                                         use_backbuffer=True,
                                         border_visible=True,
                                         fill_padding=True)

        # Add the cursves to the container
        # ... ... ...
        for line in self._make_curves():
            container.add(line)

        # index_mapper = plot.index_mapper
        # value_mapper = plot.value_mapper

        return container

    def _plot_default(self):
        return self._create_plot_component()

    def _marker_size_changed(self):
        self.ber_plot_renderer.marker_size = self.marker_size
        self.ser_plot_renderer.marker_size = self.marker_size

    def _marker_changed(self):
        self.ber_plot_renderer.marker = self.marker
        self.ser_plot_renderer.marker = self.marker



# def get_dummy_plot_data():
#     from comm.modulators import PSK

#     SNR = np.arange(0, 20, 3)
#     pd=ArrayPlotData()
#     pd.set_data('SNR', SNR)

#     psk = PSK(4)
#     SER = psk.calcTheoreticalSER(SNR)
#     BER = psk.calcTheoreticalBER(SNR)

#     SNR_ds = ArrayDataSource(SNR)
#     BER_ds = ArrayDataSource(BER)
#     SER_ds = ArrayDataSource(SER)

#     pd.set_data('SER', SER)
#     pd.set_data('BER', BER)
#     return pd

if __name__ == '__main__':
    #SimulationResultsPlotter().configure_traits()

    # Get some plot data
    #pd = get_dummy_plot_data()

    plotter = DummyPlotter(None)
    plotter.configure_traits()
