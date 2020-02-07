.. pyphysim documentation master file, created by
   sphinx-quickstart on Thu Sep 15 16:50:43 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyPhysim's documentation!
====================================

PyPhysim is a python library implementing functions, classes and simulators
related to the physical layer of digital communications systems.


.. _list_of_hand_written_doc:

List of Documentation Articles
==============================

PyPhysim organization is described in :doc:`hand_written/description`, where a summary
of the several packages in PyPhysim is shown. For instructions on how to
implement new simulations with PyPhysim see :doc:`hand_written/simulators`.

The complete list of articles in the PyPhysim documentation is shown below.

.. toctree::
   :maxdepth: 1

   hand_written/description
   hand_written/simulators
   hand_written/monte_carlo
   hand_written/writing_documentation
   hand_written/writing_unittests
   hand_written/speedup
   hand_written/packaging
   hand_written/typing_support
   hand_written/zreferences

.. _pyphysim_api:

Packages and Modules in PyPhysim
================================

.. toctree::
   :maxdepth: 3

   pyphysim

.. The modules.rst file is created by sphinx-apidoc, but we dont want to
   include it, since we already included a toctree with the pyphysim
   subpackages above. Therefore, we include the modules.rst file as hidden here
   so that sphinx does not complain that it is not included in any toctree.
.. toctree::
   :maxdepth: 4
   :hidden:

   modules.rst


..
   Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

Todo List
=========

.. todolist::
