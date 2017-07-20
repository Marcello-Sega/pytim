# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
"""A python based tool for interfacial molecules analysis
"""

# Always prefer setuptools over distutils
from   setuptools import find_packages
# To use a consistent encoding
import codecs
import os
import sys

from   distutils.core import setup
from   distutils.extension import Extension
from   Cython.Distutils import build_ext

import numpy


pytim_dbscan = Extension("pytim_dbscan", ["pytim/_dbscan_inner.pyx"], language="c++",
    include_dirs = [numpy.get_include()])

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with codecs.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# This fixes the default architecture flags of Apple's python
if sys.platform == 'darwin' and os.path.exists('/usr/bin/xcodebuild'):
    os.environ['ARCHFLAGS'] = ''

setup(
    name='pytim',
    ext_modules=[pytim_dbscan],
    cmdclass = {'build_ext': build_ext},
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.5.2',

    description='Python Tool for Interfacial Molecules Analysis',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/Marcello-Sega/pytim',

    # Author details
    author='Marcello Sega, Balazs Fabian, Gyorgy Hantal, Pal Jedlovszky',
    author_email='marcello.sega@univie.ac.at',

    # Choose your license
    license='GPLv3',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='molecular simuations analysis ',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['MDAnalysis>=0.15','PyWavelets>=0.5.2','numpy>=1.12.0',
                      'scipy>=0.18','scikit-image>=0.13.0','cython>=0.24.1',
                      'sphinx>=1.4.3','matplotlib','pytetgen>=0.1.1'],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
  ##  extras_require={
  ##      'dev': ['check-manifest'],
  ##      'test': ['coverage'],
  ##  },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'pytim': ['data/*'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
  ##  data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
  ##  entry_points={
  ##      'console_scripts': [
  ##          'sample=sample:main',
  ##      ],
  ##  },

)
