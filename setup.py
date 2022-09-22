"""
Script used to pip install src as a standalone package.
Necessary for package imports between sibling directories.

Run "pip install -e ." in the root directory (i.e. region-guided-chest-x-ray-report-generation)
to install src.
"""

from setuptools import setup, find_packages

setup(name='src', version='1.0', packages=find_packages())
