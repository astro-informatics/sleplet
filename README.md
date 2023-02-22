# SLEPLET

[![SLEPLET](https://zenodo.org/badge/DOI/10.5281/zenodo.7268074.svg)](https://doi.org/10.5281/zenodo.7268074)
[![Python](https://img.shields.io/badge/python-3.10-orange.svg)](https://www.python.org/downloads/release/python-3100/)
[![tests](https://github.com/astro-informatics/sleplet/actions/workflows/tests.yml/badge.svg)](https://github.com/astro-informatics/sleplet/actions/workflows/tests.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## Installation

Run

```sh
pip install .
```

This will install two scripts `sphere` and `mesh` which can be used to generate the figures in [the following section](#paper-figures).

### Developer Installation

Run

```sh
pip install -e .[dev]
```

then

```sh
pre-commit install
```

## Paper Figures

To recreate the figures from below papers one may use the CLI or API methods.
For those which don't use the `mesh` or `sphere` CLIs the relevant API code
isn't provided as it is contained within the source code.

### Sifting Convolution on the Sphere

[![Sifting Convolution on the Sphere](https://img.shields.io/badge/DOI-10.1109/LSP.2021.3050961-pink.svg)](https://dx.doi.org/10.1109/LSP.2021.3050961)

#### Fig. 1

```sh
for ell in $(seq 2 -1 1); do
    sphere harmonic_gaussian -a 0.75 -b 0.125 -e ${ell} 1 -L 128 -m translate -o
done
```

#### Fig. 2

```sh
sphere earth -L 128
```

#### Fig. 3

```sh
for ell in $(seq 2 -1 1); do
    sphere harmonic_gaussian -c earth -e ${ell} 1 -L 128
done
```

### Slepian Scale-Discretised Wavelets on the Sphere

[![Slepian Scale-Discretised Wavelets on the Sphere](https://img.shields.io/badge/DOI-10.1109/TSP.2022.3233309-pink.svg)](https://dx.doi.org/10.1109/TSP.2022.3233309)

#### Fig. 2

```sh
python -m sleplet.plotting.arbitrary.south_america.tiling_south_america
```

#### Fig. 3

```sh
# a
sphere earth -L 128 -s 2 -u
# b
sphere slepian_south_america -L 128 -s 2 -u
```

#### Fig. 4

```sh
for p in 0 9 24 49 99 199; do
    sphere slepian -e ${p} -L 128 -u
done
```

```python
from sleplet.functions.fp.slepian import Slepian
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

region = Region(mask_name="south_america")
for p in [0, 9, 24, 49, 99, 199]:
    f = Slepian(L=128, region=region, rank=p)
    f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
    Plot(
        f_sphere,
        f.L,
        f.name,
        normalise=False,
        region=region,
    ).execute()
```

#### Fig. 5

```sh
python -m sleplet.plotting.arbitrary.south_america.eigenvalues_south_america
```

#### Fig. 6

```sh
# a
sphere slepian_wavelets -L 128 -u
# b-f
for j in $(seq 0 4); do
    sphere slepian_wavelets -e 3 2 ${j} -L 128 -u
done
```

```python
from sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

region = Region(mask_name="south_america")
for j in [None] + list(range(5)):
    f = SlepianWavelets(L=128, region=region, B=3, j_min=2, j=j)
    f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
    Plot(
        f_sphere,
        f.L,
        f.name,
        normalise=False,
        region=region,
    ).execute()
```

#### Fig. 7

```sh
# a
sphere slepian_wavelet_coefficients_south_america -L 128 -s 2 -u
# b-f
for j in $(seq 0 4); do
    sphere slepian_wavelet_coefficients_south_america -e 3 2 ${j} -L 128 -s 2 -u
done
```

#### Fig. 8

```sh
# a
sphere slepian_south_america -L 128 -n -10 -s 2 -u
# b-d
for s in 2 3 5; do
    python -m sleplet.plotting.arbitrary.south_america.denoising_slepian_south_america -n -10 -s ${s}
done
```

#### Fig. 9

```sh
# a
sphere earth -L 128 -s 2 -u -v africa
# b
sphere slepian_africa -L 128 -s 2 -u
```

#### Fig. 10

```sh
python -m sleplet.plotting.arbitrary.africa.eigenvalues_africa
```

#### Fig. 11

```sh
sed -i 's/south_america/africa/g' src/sleplet/config/settings.toml
for p in 0 9 24 49 99 199; do
    sphere slepian -e ${p} -L 128 -u
done
```

```python
from sleplet.functions.fp.slepian import Slepian
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

region = Region(mask_name="africa")
for p in [0, 9, 24, 49, 99, 199]:
    f = Slepian(L=128, region=region, rank=p)
    f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
    Plot(
        f_sphere,
        f.L,
        f.name,
        normalise=False,
        region=region,
    ).execute()
```

#### Fig. 12

```sh
sed -i 's/south_america/africa/g' src/sleplet/config/settings.toml
# a
sphere slepian_wavelets -L 128 -u
# b
for j in $(seq 0 5); do
    sphere slepian_wavelets -e 3 2 ${j} -L 128 -u
done
```

```python
from sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

region = Region(mask_name="africa")
for j in [None] + list(range(5)):
    f = SlepianWavelets(L=128, region=region, B=3, j_min=2, j=j)
    f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
    Plot(
        f_sphere,
        f.L,
        f.name,
        normalise=False,
        region=region,
    ).execute()
```

#### Fig. 13

```sh
# a
sphere slepian_wavelet_coefficients_africa -L 128 -s 2 -u
# b
for j in $(seq 0 5); do
    sphere slepian_wavelet_coefficients_africa -e 3 2 ${j} -L 128 -s 2 -u
done
```

#### Fig. 14

```sh
# a
sphere slepian_africa -L 128 -n -10 -s 2 -u
# b-d
for s in 2 3 5; do
    python -m sleplet.plotting.arbitrary.africa.denoising_slepian_africa -n -10 -s ${s}
done
```

### Slepian Scale-Discretised Wavelets on Manifolds

[![Slepian Scale-Discretised Wavelets on Manifolds](https://img.shields.io/badge/DOI-10.48550/arXiv.2302.06006-pink.svg)](https://doi.org/10.48550/arXiv.2302.06006)

#### Fig. 2

```sh
for r in $(seq 2 9); do
    mesh homer -e ${r} -u
done
```

#### Fig. 4

```sh
python -m sleplet.plotting.mesh.mesh_tiling homer
```

#### Fig. 5

```sh
python -m sleplet.plotting.mesh.mesh_region homer
```

#### Fig. 6

```sh
for p in 0 9 24 49 99 199; do
    mesh homer -m slepian_functions -e ${p} -u -z
done
```

#### Fig. 7

```sh
python -m sleplet.plotting.mesh.mesh_slepian_eigenvalues homer
```

#### Fig. 8

```sh
# a
mesh homer -m slepian_wavelets -u -z
# b-f
for j in $(seq 0 4); do
    mesh homer -e 3 2 ${j} -m slepian_wavelets -u -z
done
```

#### Fig. 9

```sh
mesh homer -m field -u
```

#### Fig. 10

```sh
# a
mesh homer -m slepian_wavelet_coefficients -u -z
# b-f
for j in $(seq 0 4); do
    mesh homer -e 3 2 ${j} -m slepian_wavelet_coefficients -u -z
done
```

#### Fig. 11

```sh
# a
mesh homer -m slepian_field -u -z
# b
mesh homer -m slepian_field -n -5 -u -z
# c
python -m sleplet.plotting.mesh.denoising_slepian_mesh homer -n -5 -s 1
```

#### Fig. 12

```sh
for f in bird cheetah cube dragonteapot; do
    python -m sleplet.plotting.mesh.mesh_region ${f}
done
```
