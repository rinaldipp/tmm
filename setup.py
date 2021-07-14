# -*- coding: utf-8 -*-
from setuptools import setup
from glob import glob
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="tmm",
    version="0.0.1",
    author="Rinaldi Polese Petrolli",
    author_email="rinaldipp@gmail.com",
    description="TMM for modeling acoustic treatments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rinaldipp/tmm",
    packages=setuptools.find_packages(),
    install_requires = ['numpy', 'scipy', 'matplotlib', 'pandas', 'mpmath', 'xlsxwriter', 'h5py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
