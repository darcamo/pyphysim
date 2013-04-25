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

The sphinx documentation generation system is configured to allow the
inclusion of math snippets with LaTeX syntax in the documentation. Simply
put a directive like ``:math:`x+y = \frac{1}{2}```
in the documentation and it will be rendered as :math:`x+y = \frac{1}{2}`
in the final HTML documentation.

In order for this to work you need to download MathJax from
`http://www.mathjax.org/download/`
and put it in the `docs/Mathjax/` folder.

Also, in order to make writing math easier, a few extra LaTeX macros should
be defined. In special, we use bold to indicate matrices. Usually one would
need to write ``:math:`\mathbf{H}_{jk}``` to write :math:`\mathbf{H}_{jk}`,
but the macro "``\mtH``" is used instead for a bold "H" such that we can
write the same think with ``:math:`\mtH_{jk}``` and this should be
configured in Mathjax. Likewise, equivalent macros must also be defined for
the other letters as well as ``\vtH`` and similar to represent vectors
(lower case bold letters).

The after downloading Mathjax to the `docs/Mathjax/` edit the
`docs/Mathjax/config/local/local.js` file and add the following macros
there.

.. code-block:: javascript
                
    // Matrices
    TEX.Macro("mtA", "\\mathbf{A}");
    TEX.Macro("mtB", "\\mathbf{B}");
    TEX.Macro("mtC", "\\mathbf{C}");
    TEX.Macro("mtD", "\\mathbf{D}");
    TEX.Macro("mtE", "\\mathbf{E}");
    TEX.Macro("mtF", "\\mathbf{F}");
    TEX.Macro("mtG", "\\mathbf{G}");
    TEX.Macro("mtH", "\\mathbf{H}");
    TEX.Macro("mtI", "\\mathbf{I}");
    TEX.Macro("mtJ", "\\mathbf{J}");
    TEX.Macro("mtK", "\\mathbf{K}");
    TEX.Macro("mtL", "\\mathbf{L}");
    TEX.Macro("mtM", "\\mathbf{M}");
    TEX.Macro("mtN", "\\mathbf{N}");
    TEX.Macro("mtO", "\\mathbf{P}");
    TEX.Macro("mtP", "\\mathbf{P}");
    TEX.Macro("mtQ", "\\mathbf{Q}");
    TEX.Macro("mtR", "\\mathbf{R}");
    TEX.Macro("mtS", "\\mathbf{S}");
    TEX.Macro("mtT", "\\mathbf{T}");
    TEX.Macro("mtU", "\\mathbf{U}");
    TEX.Macro("mtV", "\\mathbf{V}");
    TEX.Macro("mtW", "\\mathbf{W}");
    TEX.Macro("mtX", "\\mathbf{X}");
    TEX.Macro("mtY", "\\mathbf{Y}");
    TEX.Macro("mtZ", "\\mathbf{Z}");
    // Vectors
    TEX.Macro("vtB", "\\mathbf{b}");
    TEX.Macro("vtC", "\\mathbf{c}");
    TEX.Macro("vtD", "\\mathbf{d}");
    TEX.Macro("vtE", "\\mathbf{e}");
    TEX.Macro("vtF", "\\mathbf{f}");
    TEX.Macro("vtG", "\\mathbf{g}");
    TEX.Macro("vtH", "\\mathbf{h}");
    TEX.Macro("vtI", "\\mathbf{i}");
    TEX.Macro("vtJ", "\\mathbf{j}");
    TEX.Macro("vtK", "\\mathbf{k}");
    TEX.Macro("vtL", "\\mathbf{l}");
    TEX.Macro("vtM", "\\mathbf{m}");
    TEX.Macro("vtN", "\\mathbf{n}");
    TEX.Macro("vtO", "\\mathbf{p}");
    TEX.Macro("vtP", "\\mathbf{p}");
    TEX.Macro("vtQ", "\\mathbf{q}");
    TEX.Macro("vtR", "\\mathbf{r}");
    TEX.Macro("vtS", "\\mathbf{s}");
    TEX.Macro("vtT", "\\mathbf{t}");
    TEX.Macro("vtU", "\\mathbf{u}");
    TEX.Macro("vtV", "\\mathbf{v}");
    TEX.Macro("vtW", "\\mathbf{w}");
    TEX.Macro("vtX", "\\mathbf{x}");
    TEX.Macro("vtY", "\\mathbf{y}");
    TEX.Macro("vtZ", "\\mathbf{z}");

    
