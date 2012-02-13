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
from traits.api import HasTraits, Instance, Int, Range, Str, Array, Enum, Dict, on_trait_change, List, Set
from traitsui.api import View, Item, Group, VGroup

# Chaco imports
from chaco.plot_factory import _create_data_sources
from chaco.api import Plot, ArrayPlotData, marker_trait, ArrayDataSource, OverlayPlotContainer, create_line_plot, Legend, PlotLabel
from chaco.tools.api import PanTool, ZoomTool, BroadcasterTool, LegendTool, SaveTool, TraitsTool

# Enable imports
from enable.component_editor import ComponentEditor

from enable.api import MarkerTrait, black_color_trait
from traits.api import Float, Any, ListInstance
from traitsui.editors import ArrayEditor, TabularEditor, TableEditor, CompoundEditor, CustomEditor, ListEditor, TreeEditor, InstanceEditor, DropEditor
from chaco.api import LinePlot, DataRange1D, LinearMapper, ScatterPlot
from chaco.scatterplot import render_markers
from chaco.plot_factory import add_default_axes, add_default_grids


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

    def _marker_size_changed(self):
        self.request_redraw()

    def _marker_changed(self):
        self.request_redraw()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class PlotIndexData(HasTraits):
    name = Str()
    data = Array()

    traits_view = View(Item('name', style='readonly'),
                       Item('data', style='readonly', editor=ArrayEditor()),
                       buttons=['OK'])


class PlotCurveData(HasTraits):
    name = Str()
    data = Array()
    traits_view = View(Item('name', style='readonly'),
                       Item('data', style='readonly', editor=ArrayEditor()))


class PlotView(HasTraits):
    """Class to plor simulation results.
    """
    # Dictionary Trait to store the curves. The keys are used as the legend
    # for the curves
    curves_data = Dict(Str, PlotCurveData)

    # Dictionary Trait to store the index array data. If there is more then
    # one element (for instance, SNR and Es/N0) and the user will be able
    # to choose which one to use in the plot. The key of the chosen data is
    # used as the axis label.
    indexes_data = Dict(Str, PlotIndexData)

    # xxxxx Plot traits xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Tell if the value axis uses a linear or a logarithmic mapper
    value_mapper_type = Enum('linear', 'log')
    # Indicates which of the index_data curves will be used for the index
    # axis
    chosen_index = Str()

    plot_container = Instance(OverlayPlotContainer)

    # List of colors to cycle through
    auto_colors = List(["green", "blue", "red",
                        "pink", "darkgray", "silver", "lightgreen", "lightblue"])

    # Trait Definitions


    # Group for the curves_data traits
    curves_group = Instance(Group)


    traits_view = View(
        Group(Item('plot_container', editor=ComponentEditor(), show_label=False),
              Group(Item('curves_data', editor=InstanceEditor()),
                    Item('indexes_data', editor=InstanceEditor()),
                    Item('chosen_index', style='readonly'))),
        width=800, height=600,
        buttons=['OK', 'Cancel'],
        resizable=True)

    # traits_view = View(Item('plot_container', editor=ComponentEditor(), show_label=False),
    #                    Item('curves_data', editor=InstanceEditor()),
    #                    Item('indexes_data', editor=InstanceEditor()),
    #                    Item('chosen_index', style='readonly'),
    #                    width=800, height=600,
    #                    buttons=['OK', 'Cancel'],
    #                    resizable=True,
    #                    title='Window Title')

    def __init__(self, ):
        """
        """
        HasTraits.__init__(self)

    def create_the_plot(self):
        # Create the datasources
        index = ArrayDataSource(self.indexes_data[self.chosen_index].data)
        values = [ArrayDataSource(self.curves_data[key].data) for key in self.curves_data.keys()]

        # Create the data range objects
        index_range = DataRange1D()
        index_range.add(index)

        value_range = DataRange1D()
        for v in values:
            value_range.add(v)

        index_mapper = LinearMapper(range=index_range)

        # TODO: So estou adicionando a primeira curva!!!
        # Adicione todas
        value_mapper = LinearMapper(range=value_range)

        plot_container = OverlayPlotContainer(padding=60,
                                              fill_padding=True,
                                              bgcolor="lightgray",
                                              use_backbuffer=True)
        plots_dict = {}  # Used for the legend
        for v, curve_name, color in zip(values, self.curves_data.keys(), self.auto_colors):
            plot = ScatterLinePlot(index=index, value=v,
                                   index_mapper=index_mapper,
                                   value_mapper=value_mapper,
                                   orientation='h',
                                   bgcolor='white',
                                   color=color,
                                   line_width=1.0,
                                   line_style='solid',  # solid, dash, ...
                                   border_visible=False)
            plots_dict[curve_name] = plot

            y_axis, x_axis = add_default_axes(plot, vtitle='Y Axis Title', htitle='X Axis Title')
            add_default_grids(plot)
            plot_container.add(plot)

        # xxxxx Set a legend for the plot xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Create the Legend object
        legend = Legend(component=plot_container, padding=10, align="ur")
        # Fill the legend object
        legend.plots = plots_dict
        # Allow the legend to be dragged by right mouse button
        legend.tools.append(LegendTool(component=legend, drag_button="right"))
        # Set the legend as an overlay of the plot
        plot_container.overlays.append(legend)

        # xxxxx Add the title at the top of the plot xxxxxxxxxxxxxxxxxxxxxx
        plot_container.overlays.append(
            PlotLabel("My Plot title",
                      component=plot_container,
                      font="swiss 20",
                      overlay_position="top"))

        # xxxxx Add the Save tool xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Press Ctrl+S to save the plot
        plot_container.tools.append(SaveTool(component=plot_container, always_on=True, filename="test_save_tool.pdf"))

        # xxxxx Add the TraitsTool xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        #plot_container.tools.append(TraitsTool(y_axis))
        #plot_container.tools.append(TraitsTool(x_axis))

        traits_tool = TraitsTool(plot_container)
        plot_container.tools.append(traits_tool)

        # Return the plot_container
        return plot_container

    def add_index_data(self, name, index_array):
        """Add a new array as the index data.

        Arguments:
        - `name`: Name of the index axis for this data
        - `index_array`: Numpy array with the data
        """
        index_data = PlotIndexData(name=name, data=index_array)
        self.indexes_data[name] = index_data

    def add_curve_data(self, name, curve_array):
        """Add a new curve to be plotted.

        Arguments:
        - `name`: Name for the curve legend
        - `curve_array`: Numpy array with the curve data
        """
        curve_data = PlotCurveData(name=name, data=curve_array)
        self.curves_data[name] = curve_data


    # NOTE: Since curves_data is a container (a dictionary) and we want
    # this method to run even if some item in the container changes, we had
    # to add "items" to the special name convention that traits use. If the
    # name _curves_data_changed" was used then the method would run only if
    # a whole dictionary was assigned to curves_data.
    def _indexes_data_items_changed(self, old, new):
        if self.chosen_index not in self.indexes_data.keys():
            # Set the chosen_index as the first curve
            self.chosen_index = self.indexes_data.keys()[0]


# TODO: Apagar se não conseguir fazer prestar
class PlotConfigView(HasTraits):
    plot_view = Instance(PlotView)
    curves = Set()

    traits_view = View(Item('plot_view'),
                       Item('curves'))

    def __init__(self, plot_view):
        """

        Arguments:
        - `plot_view`: PlotView object
        """
        HasTraits.__init__(self)
        self.plot_view = plot_view

        self.curves = set(self.plot_view.plot_container.plot_components)



if __name__ == '__main__':
    # index_data = PlotIndexData()
    # plot_curve = PlotCurveData()
    plot_view = PlotView()

    angle_radians = np.arange(0, 2 * np.pi, 0.05 * np.pi)
    angle_degrees = 180 * angle_radians / np.pi
    sin_values = np.sin(angle_radians)
    cos_values = np.cos(angle_radians)

    plot_view.add_index_data('Angle (radians)', angle_radians)
    plot_view.add_index_data('Angle (degrees)', angle_degrees)
    plot_view.add_curve_data('Sin', sin_values)
    plot_view.add_curve_data('Cos', cos_values)

    plot_view.plot_container = plot_view.create_the_plot()
    plot_view.configure_traits()


    #plot_config_view = PlotConfigView(plot_view)
    #plot_config_view.configure_traits()
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
        #super(SimulationResultsPlotter, self).__init__()
        HasTraits.__init__(self)
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
        self.ber_plot_renderer = self.plot.add_xy_plot(
            "SNR", "ber", ScatterLinePlot,
            color="green",
            name='Simulated BER',
            value_scale='log',
            line_style='dash')[0]

        self.ser_plot_renderer = self.plot.add_xy_plot(
            "SNR", "ser", ScatterLinePlot,
            color="blue",
            name='Simulated SER',
            value_scale='log',
            line_style='dash')[0]

        # Theoretical BER
        theoretical_ber_plot_renderer = self.plot.plot(
            ("SNR", "theoretical_ber"),
            type="line",
            color="green",
            name='Theoretical BER',
            value_scale='log',
            line_style='solid')[0]
        # Theoretical SER
        theoretical_ser_plot_renderer = self.plot.plot(
            ("SNR", "theoretical_ser"),
            type="line",
            color="green",
            name='Theoretical SER',
            value_scale='log',
            line_style='solid')[0]

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

        # Update plot marker_size and marker type with the ones in the
        # SimulationResultsPlotter object
        self._marker_size_changed()
        self._marker_changed()

        # xxxxx Add a SaveTool to the plot xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.plot.tools.append(SaveTool(self.plot))

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
        # self.ber_plot_renderer.marker_size = self.marker_size
        # self.ser_plot_renderer.marker_size = self.marker_size
        for plot in self.plot.components:
            plot.marker_size = self.marker_size

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


if __name__ == '__main__1':
    #SimulationResultsPlotter().configure_traits()

    # Get some plot data
    #pd = get_dummy_plot_data()

    plotter = DummyPlotter(None)
    plotter.configure_traits()
