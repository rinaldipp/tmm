# -*- coding: utf-8 -*-
from setuptools import setup
from glob import glob

with open("README.md", "r") as f:
    long_description = f.read()

settings = {
    'name': 'tmm',
    'version': '0.0.1',
    'description': 'TMM for modeling acoustic treatments',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'url': 'https://github.com/rinaldipp/tmm',
    'author': 'Rinaldi Polese Petrolli',
    'author_email': 'rinaldipp@gmail.com',
    'license': 'MIT',
    'install_requires': ['numpy', 'scipy', 'matplotlib',
        'pandas', 'mpmath', 'xlsxwriter', 'git+https://github.com/pyttamaster/pytta@development'],
    'classifiers': [
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", ],
    'python_requires': '>=3.6, <3.9',
}

setup(**settings)
