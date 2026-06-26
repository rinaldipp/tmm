import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="tmm",
    version="0.0.4",
    author="Rinaldi Polese Petrolli",
    author_email="rinaldipp@gmail.com",
    description="Transfer Matrix Models for modeling acoustic treatments",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rinaldipp/tmm",
    packages=setuptools.find_namespace_packages(include=["tmm", "tmm.database", "tmm.database.*"]),
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib", "pandas", "mpmath", "xlsxwriter", "h5py"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
