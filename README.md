# SLEPLET

[![PyPI](https://badge.fury.io/py/sleplet.svg)](https://pypi.org/project/sleplet)
[![Python](https://img.shields.io/pypi/pyversions/sleplet)](https://www.python.org)
[![Documentation](https://img.shields.io/badge/Documentation-SLEPLET-blueviolet.svg)](https://astro-informatics.github.io/sleplet)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.7268074.svg)](https://doi.org/10.5281/zenodo.7268074)
[![Test](https://github.com/astro-informatics/sleplet/actions/workflows/test.yml/badge.svg)](https://github.com/astro-informatics/sleplet/actions/workflows/test.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

`SLEPLET` is a Python package for the construction of Slepian wavelets in the
spherical and manifold (via meshes) settings. The API of `SLEPLET` has been
designed in an object-orientated manner and is easily extendible. Upon
installation, `SLEPLET` comes with two command line interfaces - `sphere` and
`mesh` - which allows one to easily generate plots on the sphere and a set of
meshes using `plotly`.

## Installation

The recommended way to install `SLEPLET` is via
[pip](https://pypi.org/project/pip/)

```sh
pip install sleplet
```

To install the latest development version of `SLEPLET` clone this repository
and run

```sh
pip install -e .
```

This will install two scripts `sphere` and `mesh` which can be used to generate
the figures in [the associated papers](./../src/sleplet/documentation.md#paper-figures).
