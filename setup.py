# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:57:58 2025

@author: peill
"""

from setuptools import setup

setup(name='LinearRegression',
      version='1.0.0',
      description='Perform linear regression',
      author='Clément Peillon',
      author_email='clement.peillon@sigma-clermont.fr',
      packages=['LinearRegression', 'LinearRegression.example'],
      package_dir={'LinearRegression':'LinearRegression',
                   'LinearRegression.example':'LinearRegression/example'},
      install_requires=['numpy', 'matplotlib']
      )