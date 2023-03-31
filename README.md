# SLEPLET

[![PyPI](https://badge.fury.io/py/sleplet.svg)](https://pypi.org/project/sleplet)
[![Python](https://img.shields.io/pypi/pyversions/sleplet)](https://www.python.org)
[![Documentation](https://img.shields.io/badge/Documentation-API-blueviolet.svg)](https://astro-informatics.github.io/sleplet)
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
the figures in [the figure section](#paper-figures).

## Bandlimit

The bandlimit is set as `L` throughout the code and the CLIs. The default value
is set to `L=16` and the figures created in [the figure section](#paper-figures)
all use `L=128`. The pre-computed data exists on
[Zenodo](https://doi.org/10.5281/zenodo.7767698) for powers of two up to `L=128`.
Other values will be computed when running the appropriate code (and saved for
future use). Note that beyond `L=32` the code can be slow due to the
difficulties of computing the Slepian matrix prior to the eigendecomposition, as
such it is recommended to stick to the powers of two up to `L=128`.

## Environment Variables

- `NCPU`: sets the number of cores to use

When it comes to selecting a Slepian region the order precedence is
[polar cap region](https://doi.org/10.1111/j.1365-246X.2006.03065.x) >
[limited latitude longitude region](https://doi.org/10.1109/TSP.2016.2646668) >
arbitrary region,
[as seen in the code](https://github.com/astro-informatics/sleplet/blob/main/src/sleplet/utils/region.py).
The default region is the `south_america` arbitrary region.

- `POLAR_GAP`
  - for a Slepian `polar cap region`, when set in conjunction with `THETA_MAX`
    but without the other `PHI`/`THETA` variables
- `THETA_MAX`
  - for a Slepian `polar cap region`, when set without the other `PHI`/`THETA`
    variables
  - for a Slepian `limited latitude longitude region`
- `THETA_MIN`
  - for a Slepian `limited latitude longitude region`
- `PHI_MAX`
  - for a Slepian `limited latitude longitude region`
- `PHI_MIN`
  - for a Slepian `limited latitude longitude region`
- `SLEPIAN_MASK`
  - for an arbitrary Slepian region, currently `africa`/`south_america` supported

##### Fig. 2.2

```sh
#!/usr/bin/env bash
# a
sphere elongated_gaussian -e -1 -1 -L 128
# b-d
for angle in 0,0,0.25 0,0.25,0.25 0.25,0.25,0.25; do
    read -r a b g <<< $(echo ${angle} | tr ',' ' ')
    sphere elongated_gaussian -e -1 -1 -L 128 -m rotate -a ${a} -b ${b} -g ${g}
done
```
