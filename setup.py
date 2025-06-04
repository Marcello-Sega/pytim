# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
"""A python based tool for interfacial molecules analysis
"""

# To use a consistent encoding
import codecs
import os
import sys

# Always prefer setuptools over distutils
try:
    from setuptools import find_packages
    from Cython.Distutils import build_ext
    import numpy
except ImportError as mod_error:
    mod_name = mod_error.message.split()[3]
    sys.stderr.write("Error : " + mod_name + " is not installed\n"
                     "Use pip install " + mod_name + "\n")
    exit(100)

from setuptools import setup
from distutils.extension import Extension

pytim_dbscan = Extension(
    "pytim_dbscan", ["pytim/dbscan_inner.pyx"],
    language="c++",
    include_dirs=[numpy.get_include()])

circumradius = Extension(
    "circumradius", ["pytim/circumradius.pyx"],
    language="c++",
    include_dirs=[numpy.get_include()])

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with codecs.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# This fixes the default architecture flags of Apple's python
if sys.platform == 'darwin' and os.path.exists('/usr/bin/xcodebuild'):
    os.environ['ARCHFLAGS'] = ''

# Get version from the file version.py
version = {}
with open("pytim/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='pytim',
    ext_modules=[pytim_dbscan, circumradius],
    cmdclass={
        'build_ext': build_ext,
    },
    
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'pytim': ['data/*'],
    },

)
