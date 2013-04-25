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


Writing math in the documentation
---------------------------------

The sphinx documentation generation sistem is configured to allow the
inclusion of math snippets with LaTeX syntax in the documentation. Sinple
put a directive like ``:math:`x+y = \frac{1}{2}```
in the documentation and it will be rendered as :math:`x+y = \frac{1}{2}`
in the final HTML documentation.

In order to make writing math easier, a few extra LaTeX macros are also
defined. In special, we use bold to indicate matrices. Usually one would
need to write ``:math:`\mathbf{H}_{jk}``` to write :math:`\mathbf{H}_{jk}`,
but the macro "``\mtH``" is defined for a bold "H" such that we can write
the same think with ``:math:`\mtH_{jk}```. Likewise, equivalent macros are
defined for the other letters as well as ``\vtH`` and similar to represent
vectors (lower case bold letters).


.. TODO::

   Configure mathjax in sphinx to accept the ``\mtH``, ``\vtH`` as well as
   the macros for the other letters in the LaTeX math snippets.
