# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tmm-rinaldipp",
    version="0.0.1",
    author="Rinaldi Polese Petrolli",
    author_email="rinaldipp@gmail.com",
    description="TMM for modeling acoustic treatments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rinaldipp/tmm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
