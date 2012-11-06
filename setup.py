#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages
import os


# find_packages returns a list all Python packages found within directory
packages = find_packages(where='.', exclude=[])

# Run the command "python setup.py nosetests" to run the tests.


# Utility function to read the README file. Used for the long_description.
# It's nice, because now 1) we have a top level README file and 2) it's
# easier to type in the README file than to put a raw string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    # Basic Package Information
    name="PyPhySim",
    version="0.1",

    # Metadata for PyPI
    author="Darlan Cavalcante Moreira",
    author_email="darcamo@gmail.com",
    license="GNU General Public License (GPL)",
    url="http://code.google.com/p/py-physim/",
    download_url='fillmein',
    keywords='phy QAM PSK QPSK BPSK Modulation Monte Carlo',
    description=("Implementation of a digital communication (physical layer) in python."),
    long_description=read("README"),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Telecommunications Industry",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering",
    ],
    requires=["setuptools", "numpy", "nose"],
    scripts=["bin/run_nosetests_with_coverage.sh",
             "bin/run_python_coverage.sh",
             "py-physim"],
    #py_modules=['modulators', 'simulations'],
    packages=packages,
    package_data={'': ["README", "Makefile", "LICENSE.txt"],
                  'tests': ["README"],
                  'util': ["README"]},
    #data_files=['pena_calculator.org'],
    #windows=[{"script": "gui.py"}],
    # options={"py2exe":{"includes":["sip"]}},
    #options={"py2exe": {"skip_archive": True, "includes": ["sip"]}}



    setup_requires=['nose>=1.0'],
)


# Mensagem para escrever no Fórum

# Como construir um pacote Python para rodar em outra máquina?

# Preciso de uma maneira de rodar um programa que criei em Python (com alguns pacotes e módulos em cada pacote) em outra máquina (que possui apenas uma versão antiga do python instalada e não tem também as bibliotecas que preciso. Aparentemente existem várias opções para isso como o freeze, o pyinstaller, dentre outros.
