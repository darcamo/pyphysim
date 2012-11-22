Writing Documentation for PyPhysim
==================================

The handwritten rst files should be listed in the toctree of the
`index.rst` file, while the documentation for each package should be listed
in the toctree in the `description.rst` file.

Note that each package will have its corresponding `.rst` file, that uses
the autodoc feature of sphinx to include the docstrings of the python files
in the actual documentation. For instance, the `.rst` file of the `comm`
module is shown below.

.. literalinclude:: comm.rst
   :language: rst
