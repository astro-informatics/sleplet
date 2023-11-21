# SLEPLET

[![PyPI](https://badge.fury.io/py/sleplet.svg)](https://pypi.org/project/sleplet)
[![Python](https://img.shields.io/pypi/pyversions/sleplet)](https://www.python.org)
[![Licence](https://img.shields.io/github/license/astro-informatics/sleplet)](https://github.com/astro-informatics/sleplet/blob/main/LICENCE.txt)
[![JOSS](https://joss.theoj.org/papers/55d9cf16a27bf2d3141f0f66c676b7f2/status.svg)](https://joss.theoj.org/papers/55d9cf16a27bf2d3141f0f66c676b7f2)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.7268074.svg)](https://doi.org/10.5281/zenodo.7268074)
[![Documentation](https://img.shields.io/badge/Documentation-SLEPLET-blueviolet.svg)](https://astro-informatics.github.io/sleplet)
[![Test](https://github.com/astro-informatics/sleplet/actions/workflows/test.yml/badge.svg)](https://github.com/astro-informatics/sleplet/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/astro-informatics/sleplet/badge.svg?branch=main)](https://coveralls.io/github/astro-informatics/sleplet?branch=main)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Renovate](https://img.shields.io/badge/renovate-enabled-orange?logo=renovatebot)](https://renovatebot.com)

`SLEPLET` is a Python package for the construction of Slepian wavelets in the
spherical and manifold (via meshes) settings. The API of `SLEPLET` has been
designed in an object-orientated manner and is easily extendible. Upon
installation, `SLEPLET` comes with two command line interfaces - `sphere` and
`mesh` - which allows one to easily generate plots on the sphere and a set of
meshes using `plotly`.

To read more about Slepian wavelets please see the following publications

[![Sifting Convolution on the Sphere](https://img.shields.io/badge/DOI-10.1109/LSP.2021.3050961-pink.svg)](https://dx.doi.org/10.1109/LSP.2021.3050961)
[![Slepian Scale-Discretised Wavelets on the Sphere](https://img.shields.io/badge/DOI-10.1109/TSP.2022.3233309-pink.svg)](https://dx.doi.org/10.1109/TSP.2022.3233309)
[![Slepian Scale-Discretised Wavelets on Manifolds](https://img.shields.io/badge/DOI-10.48550/arXiv.2302.06006-pink.svg)](https://doi.org/10.48550/arXiv.2302.06006)
[![Slepian Wavelets for the Analysis of Incomplete Data on Manifolds](https://img.shields.io/badge/PhD%20Thesis-Patrick%20J.%20Roddy-pink.svg)](https://paddyroddy.github.io/thesis)

## Installation

The recommended way to install `SLEPLET` is via
[pip](https://pypi.org/project/pip)

```sh
pip install sleplet
```

To install the latest development version of `SLEPLET` clone this repository
and run

```sh
pip install -e .
```

This will install two scripts `sphere` and `mesh` which can be used to generate
the figures in
[the associated papers](https://astro-informatics.github.io/sleplet#paper-figures).

### Supported Platforms

`SLEPLET` has been tested with
[![Python](https://img.shields.io/pypi/pyversions/sleplet)](https://www.python.org).
Windows is not currently supported as `SLEPLET` relies on
[pyssht](https://pypi.org/project/pyssht) and
[pys2let](https://pypi.org/project/pys2let) which do not work on Windows.
These can hopefully be replaced with
[s2fft](https://github.com/astro-informatics/s2fft) and
[s2wav](https://github.com/astro-informatics/s2wav) in the future when they
are available on [PyPI](https://pypi.org).

## Example Usage

`SLEPLET` may be interacted with via the API or the CLIs.

### API Usage

The following demonstrates the first wavelet (ignoring the scaling function) of
the South America region on the sphere.

```python
import sleplet

B, J, J_MIN, L = 3, 0, 2, 128

region = sleplet.slepian.Region(mask_name="south_america")
f = sleplet.functions.SlepianWavelets(L, region=region, B=B, j_min=J_MIN, j=J)
f_sphere = sleplet.slepian_methods.slepian_inverse(f.coefficients, f.L, f.slepian)
sleplet.plotting.PlotSphere(
    f_sphere,
    f.L,
    f"slepian_wavelets_south_america_{B}B_{J_MIN}jmin_{J_MIN+J}j_L{L}",
    normalise=False,
    region=f.region,
).execute()
```

![Slepian Wavelet j=2](https://github.com/astro-informatics/sleplet/blob/main/documentation/slepian_wavelets_south_america_3B_2jmin_2j_L128_res512_real.png?raw=true)

### CLI Usage

The demonstrates the first wavelet (ignoring the scaling function) of the head
region of a Homer Simpson mesh for a per-vertex normals field.

```sh
mesh homer -e 3 2 0 -m slepian_wavelet_coefficients -u -z
```

![Slepian Mesh Wavelet Coefficients j=2](https://github.com/astro-informatics/sleplet/blob/main/documentation/slepian_wavelet_coefficients_homer_3B_2jmin_2j_zoom.png?raw=true)

## Documentation

See here for the [documentation](https://astro-informatics.github.io/sleplet).
This includes demonstrations of the figures from the associated papers along
with the API documentation. Further examples are included in the
[examples folder](https://github.com/astro-informatics/sleplet/tree/main/examples).

## Community Guidelines

We'd love any contributions you may have, please see the
[contributing guidelines](https://github.com/astro-informatics/sleplet/blob/main/CONTRIBUTING.md).

## Citing

If you use `SLEPLET` in your research, please cite the paper.

```bibtex
@article{Roddy2023,
  title   = {{SLEPLET: Slepian Scale-Discretised Wavelets in Python}},
  author  = {Roddy, Patrick J.},
  year    = 2023,
  journal = {Journal of Open Source Software},
  volume  = 8,
  number  = 84,
  pages   = 5221,
  doi     = {10.21105/joss.05221},
}
```

Please also cite [S2LET](https://doi.org/10.1051/0004-6361/201220729) upon which
`SLEPLET` is built, along with [SSHT](https://doi.org/10.1109/TSP.2011.2166394)
in the spherical setting or [libigl](https://doi.org/10.1145/3134472.3134497)
in the mesh setting.
