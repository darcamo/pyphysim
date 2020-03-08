from setuptools import Extension


def build(setup_kwargs):
    try:
        import numpy as np
        from Cython.Build import cythonize

        cyfuncs_ext = Extension(name="pyphysim.c_extensions.misc_c",
                                sources=["pyphysim/util/misc_c.pyx"],
                                include_dirs=[np.get_include()])

        EXTENSIONS = [cyfuncs_ext]

        setup_kwargs.update({
            'ext_modules':
            cythonize(EXTENSIONS, language_level=3),
            'zip_safe':
            False,
        })
    except ModuleNotFoundError:
        import warnings
        warnings.warn(
            "numpy and cython must be installed to compyle cython extensions in pyphysim"
        )
