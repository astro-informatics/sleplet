# SLEPLET

[![SLEPLET](https://zenodo.org/badge/DOI/10.5281/zenodo.7268074.svg)](https://doi.org/10.5281/zenodo.7268074)
[![tests](https://github.com/astro-informatics/sleplet/actions/workflows/tests.yml/badge.svg)](https://github.com/astro-informatics/sleplet/actions/workflows/tests.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## Installation

Run

```{sh}
pip install .
```

This will install two scripts `sphere` and `mesh` which can be used to generate the figures in [the following section](#paper-figures).

### Developer Installation

Run

```{sh}
pip install -e .[dev]
```

then

```{sh}
pre-commit install
```

### Testing

Install [tox](https://tox.wiki/) and then run `tox`.

## Paper Figures

### Sifting Convolution on the Sphere

[![Sifting Convolution on the Sphere](https://img.shields.io/badge/DOI-10.1109/LSP.2021.3050961-pink.svg)](https://dx.doi.org/10.1109/LSP.2021.3050961)

#### Fig. 1

```{sh}
for ell in $(seq 2 -1 1); do
    sphere harmonic_gaussian -a 0.75 -b 0.125 -e ${ell} 1 -L 128 -m translate -o
done
```

#### Fig. 2

```{sh}
sphere earth -L 128
```

#### Fig. 3

```{bash}
for ell in $(seq 2 -1 1); do
    sphere harmonic_gaussian -c earth -e ${ell} 1 -L 128
done
```

### Slepian Scale-Discretised Wavelets on the Sphere

[![Slepian Scale-Discretised Wavelets on the Sphere](https://img.shields.io/badge/DOI-10.1109/TSP.2022.3233309-pink.svg)](https://dx.doi.org/10.1109/TSP.2022.3233309)

#### Fig. 2

```{sh}
python -m sleplet.plotting.arbitrary.south_america.tiling_south_america
```

#### Fig. 3

```{sh}
sphere earth -L 128 -s 2 -u
sphere slepian_south_america -L 128 -s 2 -u
```

#### Fig. 4

```{sh}
for p in 0 9 24 49 99 199; do
    sphere slepian -e ${p} -L 128 -u
done
```

#### Fig. 5

```{sh}
python -m sleplet.plotting.arbitrary.south_america.eigenvalues_south_america
```

#### Fig. 6

```{sh}
sphere slepian_wavelets -L 128 -u
for j in $(seq 0 4); do
    sphere slepian_wavelets -e 3 2 ${j} -L 128 -u
done
```

#### Fig. 7

```{sh}
sphere slepian_wavelet_coefficients_south_america -L 128 -s 2 -u
for j in $(seq 0 4); do
    sphere slepian_wavelet_coefficients_south_america -e 3 2 ${j} -L 128 -s 2 -u
done
```

#### Fig. 8

```{sh}
sphere slepian_south_america -L 128 -n -10 -s 2 -u
for s in 2 3 5; do
    python -m sleplet.plotting.arbitrary.south_america.denoising_slepian_south_america -n -10 -s ${s}
done
```

#### Fig. 9

```{sh}
sphere earth -L 128 -s 2 -u -v africa
sphere slepian_africa -L 128 -s 2 -u
```

#### Fig. 10

```{sh}
python -m sleplet.plotting.arbitrary.africa.eigenvalues_africa
```

#### Fig. 11

```{sh}
for p in 0 9 24 49 99 199; do
    sphere slepian -e ${p} -L 128 -u
done
```

#### Fig. 12

```{sh}
sphere slepian_wavelets -L 128 -u
for j in $(seq 0 5); do
    sphere slepian_wavelets -e 3 2 ${j} -L 128 -u
done
```

#### Fig. 13

```{sh}
sphere slepian_wavelet_coefficients_africa -L 128 -s 2 -u
for j in $(seq 0 5); do
    sphere slepian_wavelet_coefficients_africa -e 3 2 ${j} -L 128 -s 2 -u
done
```

#### Fig. 14

```{sh}
sphere slepian_africa -L 128 -n -10 -s 2 -u
for s in 2 3 5; do
    python -m sleplet.plotting.arbitrary.africa.denoising_slepian_africa -n -10 -s ${s}
done
```

### Slepian Scale-Discretised Wavelets on Manifolds

[![Slepian Scale-Discretised Wavelets on Manifolds](https://img.shields.io/badge/DOI-10.48550/arXiv.2302.06006-pink.svg)](https://doi.org/10.48550/arXiv.2302.06006)

#### Fig. 2

```{sh}
for r in $(seq 2 9); do
    mesh homer -e ${r} -u
done
```

#### Fig. 4

```{sh}
python -m sleplet.plotting.mesh.mesh_tiling homer
```

#### Fig. 5

```{sh}
python -m sleplet.plotting.mesh.mesh_region homer
```

#### Fig. 6

```{sh}
for p in 0 9 24 49 99 199; do
    mesh homer -m slepian_functions -e ${p} -u -z
done
```

#### Fig. 7

```{sh}
python -m sleplet.plotting.mesh.mesh_slepian_eigenvalues homer
```

#### Fig. 8

```{sh}
mesh homer -m slepian_wavelets -u -z
for j in $(seq 0 4); do
    mesh homer -e 3 2 ${j} -m slepian_wavelets -u -z
done
```

#### Fig. 9

```{sh}
mesh homer -m field -u
```

#### Fig. 10

```{sh}
mesh homer -m slepian_wavelet_coefficients -u -z
for j in $(seq 0 4); do
    mesh homer -e 3 2 ${j} -m slepian_wavelet_coefficients -u -z
done
```

#### Fig. 11

```{sh}
mesh homer -m slepian_field -u -z
mesh homer -m slepian_field -n -5 -u -z
python -m sleplet.plotting.mesh.denoising_slepian_mesh homer -n -5 -s 1
```

#### Fig. 12

```{sh}
for f in bird cheetah cube dragonteapot; do
    python -m sleplet.plotting.mesh.mesh_region ${f}
done
```
