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

Notice that we use "``.. automodule:: comm``" to get the documentation of
the comm package from the source code. Then we introduce a section for the
modules in the comm package and use "``.. autosummary::``" to generate `rst`
files for each module in the comm package. These `rst` files will be
generated in the "`autosummary_folder`" folder, according to the template
`module_template.txt` (located at the "`docs/_templates/`" folder).

Therefore, you only need to create new `rst` files for any new package
implemented in PyPhysim and list them in the :file:`description.rst` file
(the :doc:`description` document). Everything else will be automatically
done by the sphinx with the help of the autodoc and autosummary extensions.
