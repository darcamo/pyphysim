#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define classes to plot the results in the apps folder using the chaco
plot library"""

__revision__ = "$Revision$"

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
from traitsui.api import View, Item, Group, VGroup, HGroup

# Chaco imports
from chaco.plot_factory import _create_data_sources
from chaco.api import Plot, ArrayPlotData, marker_trait, ArrayDataSource, OverlayPlotContainer, create_line_plot, Legend, PlotLabel, ToolbarPlot
from chaco.tools.api import PanTool, ZoomTool, BroadcasterTool, LegendTool, SaveTool, TraitsTool, BetterZoom, BetterSelectingZoom

# Enable imports
from enable.component_editor import ComponentEditor

from enable.api import MarkerTrait, black_color_trait
from traits.api import Float, Any, ListInstance, Bool, Button, Event
from traitsui.editors import ArrayEditor, TabularEditor, TableEditor, CompoundEditor, CustomEditor, ListEditor, TreeEditor, InstanceEditor, DropEditor, RangeEditor, ButtonEditor, CheckListEditor, EnumEditor
from chaco.api import LinePlot, DataRange1D, LinearMapper, ScatterPlot, log_auto_ticks
from chaco.scatterplot import render_markers
from chaco.plot_factory import add_default_axes, add_default_grids


# xxxxx Chaco Renderer for a line plot with markers xxxxxxxxxxxxxxxxxxxxxxx
# Renderer class for a line plot with markers
# https://github.com/sergey-miryanov/chaco/blob/master/chaco/ScatterLinePlot.py
# Author of this class "sergey miryanov"
#
# I made only small changes:
# - In the _render_icon function I added a shift in the x axis of half the
#   marker size so that the marker is rendered in the middle of the line in
#   the legend
# - In the _render function I added "with gc:" and the gc.clip_to_rect (as
#   well as the save and restore state) so that the markers are not draw
#   outside the plot area (if the user moved the plot with the PanTool). I
#   got this from the _render function from the ScatterPlot class.
class ScatterLinePlot(LinePlot):
    marker = MarkerTrait
    marker_size = Float(2.0)
    outline_color = black_color_trait
    custom_symbol = Any

    traits_view = View(Item("color", style="custom"),
                       "line_width",
                       "line_style",
                       Item("marker", label="Marker type"),
                       Item("marker_size", label="Marker Size"),
                       buttons=["OK", "Cancel"])

    def _gather_points(self):
        super(ScatterLinePlot, self)._gather_points()

    def _render_markers(self, gc, points):
        render_markers(gc, points, self.marker,
                self.marker_size, self.color_, self.line_width,
                self.outline_color_,
                self.custom_symbol)

    def _render(self, gc, points, selected_points=None):
        super(ScatterLinePlot, self)._render(gc, points, selected_points)
        with gc:
            gc.save_state()
            gc.clip_to_rect(self.x, self.y, self.width, self.height)
            if len(points) > 0:
                for pts in points:
                    self._render_markers(gc, pts)
            gc.restore_state()

    def _render_icon(self, gc, x, y, width, height):
        super(ScatterLinePlot, self)._render_icon(gc, x, y, width, height)
        point = np.array([x + width / 2. - self.marker_size / 2., y + height / 2.])
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
    """Class to plot simulation results.
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

    def __init__(self, ):
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

        # TODO: Only the first curve is beeing added here.  Add all of
        # them!
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


class SimulationResultsPlotter(HasTraits):
    """Class to plot the results with chaco.
    """
    plot = Instance(Plot)
    # Traits can have a description, the 'desc' argument
    marker = marker_trait(desc='the marker type of all curves')
    marker_size = Range(low=1, high=6, value=4, desc='the marker size of all curves')

    reset_plot = Event("Reset Plot")

    # xxxxx Choose one of the index data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We can have multiple index datas, but only the selected one will be
    # used
    curves_renderers = List()

    index_data_labels = List(Str)
    chosen_index = Str()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # View used to display a single curve renderer
    _plot_curve_view = View(Item('marker_size',
                                      style='custom',
                                      editor=RangeEditor(mode='spinner', low=1, high=15)),
                                 Item('marker'),
                                 Item('color'),
                                 Item('line_style'),
                                 Item('line_width'),
                                 Item('visible'))

    # xxxxx Default Viewfor the SimulationResultsPlotter class xxxxxxxxxxxx
    # If no view is specified when configure_traits() is called on
    # the SimulationResultsPlotter object, then the one named traits_view takes
    # preference
    traits_view = View(
        Group(
            Item('plot', editor=ComponentEditor(), show_label=False, width=800, height=600, resizable=True),
            Item('_'),
            Group(
                Item('curves_renderers', style='custom', editor=ListEditor(
                        use_notebook=True,
                        view=_plot_curve_view), show_label=False),
                Group(
                    Item('marker', label='All Markers Type'),
                    Item('marker_size', label='All Markers Size'),
                    # Change the chosen_index value according to one of the
                    # values in index_data_labels
                    Item('chosen_index', style='simple', editor=EnumEditor(name='index_data_labels')),
                    Group(Item("reset_plot", editor=ButtonEditor(), show_label=False)),
                    orientation='vertical'),
                orientation='horizontal'),
            orientation='vertical'
        ),
        width=800,
        height=600,
        resizable=True,
        title="Chaco Plot")
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # TODO: Ainda nao e usado
    def add_index(self, index_label, index_array):
        """Add a new index in the plot.

        All curves will be plotted for the same index data.

        Note: You can call add_index multiple times to add several
        indexes. You will then be able to select the used on in the plot
        window.

        Arguments:
        - `index_label`: A label for the index
        - `index_array`: A numpy array with the index values.
        """
        # plot.data is the plot ArrayPlotData object. The method
        # "list_data" return a list of the names of the data managed by the
        # ArrayPlotData object
        if index_label not in self.plot.data.list_data():
            self.plot.data.set_data(index_label, index_array)
            self.index_data_labels.append(index_label)

        if index_label not in self.index_data_labels:
            self.index_data_labels.append(index_label)

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
        self.plot.padding_left = 90

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Labels for the possible index values
        self.index_data_labels.append('SNR')
        self.index_data_labels.append('Eb_over_N0')  # Must be the same
                                                     # attibute name used
                                                     # in the ArrayPlotData
        self.chosen_index = self.index_data_labels[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Simulated values
        self.plot.add_xy_plot(
            "SNR", "ber", ScatterLinePlot,
            color="green",
            name='Simulated BER',
            value_scale='log',
            line_style='dash')[0]

        self.plot.add_xy_plot(
            "SNR", "ser", ScatterLinePlot,
            color="blue",
            name='Simulated SER',
            value_scale='log',
            line_style='dash')[0]

        # Theoretical BER
        self.plot.plot(
            ("SNR", "theoretical_ber"),
            type="line",
            color="green",
            name='Theoretical BER',
            value_scale='log',
            line_style='solid')[0]
        # Theoretical SER
        self.plot.plot(
            ("SNR", "theoretical_ser"),
            type="line",
            color="blue",
            name='Theoretical SER',
            value_scale='log',
            line_style='solid')[0]

        self.curves_renderers = self.plot.components

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
        self.plot.x_axis.title = self.chosen_index

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
        if self.plot.title is '':
            filename = 'saved_plot.pdf'
        else:
            filename = '%s.pdf' % self.plot.title
        self.plot.tools.append(SaveTool(self.plot, filename=filename))

        # xxxxx Add a Zoom tool and a pan tool xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Because the zoom tool draws a visible "zoom box", we need to add
        # it to the list of overlays instead of the list of tools. (It will
        # still get events.)
        self.plot.overlays.append(BetterSelectingZoom(self.plot, tool_mode="box", always_on=True, drag_button='right'))
        # self.plot.overlays.append(ZoomTool(self.plot, tool_mode="box", always_on=True, drag_button='right'))
        #self.plot.tools.append(BetterZoom(self.plot, tool_mode="box", always_on=True, drag_button='right'))
        self.plot.tools.append(PanTool(self.plot, tool_mode="box", always_on=True))

        # Change the Y Axis labels
        self.plot.y_axis.tick_label_formatter = SimulationResultsPlotter._get_logscale_tick_string

    def _chosen_index_changed(self):
        """Change the index datasource of all renderers, as well as the x
        axis label acording to the chosen_index trait.
        """
        print 'chosen index: x%sx' % self.chosen_index
        # _get_or_create_datasource is defined in the Plot class. It
        # returns a datasource associated to the provided name (which was
        # added to the ArrayPlotData object passed in the creation of the
        # Plot object)
        index_ds = self.plot._get_or_create_datasource(self.chosen_index)

        for renderer in self.plot.components:
            renderer.index = index_ds
        self.plot.x_axis.title = self.chosen_index

    @staticmethod
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

    def _marker_size_changed(self):
        for renderer in self.plot.components:
            renderer.marker_size = self.marker_size

    def _marker_changed(self):
        for renderer in self.plot.components:
            renderer.marker = self.marker

    def _reset_plot_fired(self, a):
        """Reset the plot (pan and zoom) when the event is fired."""
        self.plot.range2d.x_range.reset()
        self.plot.range2d.y_range.reset()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__1':
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


    # RESET PAN
    # plotter.plot.range2d.x_range.reset()
    # plotter.plot.range2d.y_range.reset()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


if __name__ == '__main__':
    #SimulationResultsPlotter().configure_traits()

    # from comm.modulators import PSK
    # SNR = np.arange(0, 20, 3)

    # psk = PSK(16)
    # SER = psk.calcTheoreticalSER(SNR)
    # BER = psk.calcTheoreticalBER(SNR)

    pass
