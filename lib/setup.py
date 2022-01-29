#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from setuptools import setup, find_packages
# from distutils.core import setup
import os
import stat
import shutil
import platform
import sys
import site
import glob


# -- file paths --
long_description="""PyVNLB: Python implementation of Video Non-Local Bayes"""
setup(
    name='vnlb',
    version='1.0.0',
    description='A python implementation of VNLB',
    long_description=long_description,
    url='https://github.com/gauenk/vnlb',
    author='Kent Gauen',
    author_email='gauenk@purdue.edu',
    license='MIT',
    keywords='burst denoising, non-local search, video denoising, neural network',
    install_requires=['numpy','torch','flake8'],
    packages=find_packages(),
)
