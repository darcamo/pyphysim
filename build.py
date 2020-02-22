# from distutils.command.build_ext import build_ext
# from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
# from setuptools import Extension
import numpy as np

# ext_modules = [
#     Extension(name="pyphysim.c_extensions.misc_c",
#               sources=["pyphysim/util/misc_c.pyx"],
#               include_dirs=[np.get_include()]),
# ]

# class BuildFailed(Exception):
#     pass

# class ExtBuilder(build_ext):
#     def run(self):
#         try:
#             build_ext.run(self)
#         except (DistutilsPlatformError, FileNotFoundError):
#             raise BuildFailed('File not found. Could not compile C extension.')

#     def build_extension(self, ext):
#         try:
#             build_ext.build_extension(self, ext)
#         except (CCompilerError, DistutilsExecError, DistutilsPlatformError,
#                 ValueError):
#             raise BuildFailed('Could not compile C extension.')

# def build(setup_kwargs):
#     """
#     This function is mandatory in order to build the extensions.
#     """
#     setup_kwargs.update({
#         "ext_modules": ext_modules,
#         "cmdclass": {
#             "build_ext": ExtBuilder
#         }
#     })

from setuptools import Extension
from Cython.Build import cythonize

cyfuncs_ext = Extension(name="pyphysim.c_extensions.misc_c",
                        sources=["pyphysim/util/misc_c.pyx"],
                        include_dirs=[np.get_include()])

EXTENSIONS = [cyfuncs_ext]


def build(setup_kwargs):
    setup_kwargs.update({
        'ext_modules': cythonize(EXTENSIONS, language_level=3),
        'zip_safe': False,
    })
