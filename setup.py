#! /usr/bin/env python
# coding: utf-8

from setuptools import setup

__version__ = '1.2'

setup(
    name='noc',
    version=__version__,
    description="Natal Optimization Code (NOC): module for calculating optimal TGEC stellar models",
    author='Matthieu Castro',
    author_email='mcastro@fisica.ufrn.br',
    scripts=['noc.py'],
    packages=['nocpkg', 'tgec'],
    license='GNU GPL'
)
