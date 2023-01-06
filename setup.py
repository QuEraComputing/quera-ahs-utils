#!/usr/bin/env python3
from setuptools import setup
with open('src/quera_ahs_utils/_version.py', 'r') as IO:
    exec(IO.read())
setup(version=__version__)