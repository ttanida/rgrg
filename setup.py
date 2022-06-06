"""
Script used to pip install src as a standalone package.
Necessary for package imports between sibling directories.
"""

from setuptools import setup, find_packages

setup(name='src', version='1.0', packages=find_packages())
