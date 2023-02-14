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

## Paper Figures

### Sifting Convolution on the Sphere

[![Sifting Convolution on the Sphere](https://img.shields.io/badge/DOI-10.1109/LSP.2021.3050961-pink.svg)](https://dx.doi.org/10.1109/LSP.2021.3050961)

#### Fig. 1

```{bash}
for ell in $(seq 2 -1 1); do
    sphere harmonic_gaussian -a 0.75 -b 0.125 -e ${ell} 1 -L 128 -m translate -o
done
```

#### Fig. 2

```{bash}
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

#### Fig. 1

#### Fig. 2

#### Fig. 3

#### Fig. 4

#### Fig. 5

python -m ./sleplet/plotting/south_america/eigenvalues_south_america.py

#### Fig. 6

#### Fig. 7

#### Fig. 8

#### Fig. 9

#### Fig. 10

#### Fig. 11

#### Fig. 12

#### Fig. 13

#### Fig. 14

echo figure: 2
python ${PLOTS}/south_america/tiling_south_america.py

echo figure: 3, Earth
sphere earth \
 -L ${L} \
 -s ${SMOOTHING} \
 -u
echo figure: 3, Slepian South America
sphere slepian_south_america \
 -L ${L} \
 -s ${SMOOTHING} \
 -u

for p in ${P_RANGE[@]}; do
echo figure: 4, p: ${p}
sphere slepian \
 -e ${p} \
 -L ${L} \
 -u
done

echo figure: 5
python ${PLOTS}/south_america/eigenvalues_south_america.py

echo figure: 6, scaling
sphere slepian_wavelets \
 -L ${L} \
 -u
echo figure: 7, scaling
sphere slepian_wavelet_coefficients_south_america \
 -L ${L} \
 -s ${SMOOTHING} \
 -u

for j in ${J_RANGE}; do
echo figure: 6, j: ${j}
sphere slepian_wavelets \
 -e ${B} ${J_MIN} ${j} \
 -L ${L} \
 -u

    echo figure: 7, j: ${j}
    sphere slepian_wavelet_coefficients_south_america \
        -e ${B} ${J_MIN} ${j} \
        -L ${L} \
        -s ${SMOOTHING} \
        -u

done

echo figure: 8, noised
sphere slepian_south_america \
 -L ${L} \
 -n ${SNR} \
 -s ${SMOOTHING} \
 -u

for s in ${SIGMA[@]}; do
echo figure: 8, sigma: ${s}
python \
 ${PLOTS}/south_america/denoising_slepian_south_america.py \
 -n ${SNR} \
 -s ${s}
done

echo figure: 9, Earth
sphere earth \
 -L ${L} \
 -s ${SMOOTHING} \
 -u \
 -v ${EARTH_VIEW}
echo figure: 9, Slepian Africa
sphere slepian_africa \
 -L ${L} \
 -s ${SMOOTHING} \
 -u

echo figure: 10
python ${PLOTS}/africa/eigenvalues_africa.py

for p in ${P_RANGE[@]}; do
echo figure: 11, p: ${p}
sphere slepian \
 -e ${p} \
 -L ${L} \
 -u
done

echo figure: 12, scaling
sphere slepian_wavelets \
 -L ${L} \
 -u
echo figure: 13, scaling
sphere slepian_wavelet_coefficients_africa \
 -L ${L} \
 -s ${SMOOTHING} \
 -u

for j in ${J_RANGE_AFRICA}; do
echo figure: 12, j: ${j}
sphere slepian_wavelets \
 -e ${B} ${J_MIN} ${j} \
 -L ${L} \
 -u

    echo figure: 13, j: ${j}
    sphere slepian_wavelet_coefficients_africa \
        -e ${B} ${J_MIN} ${j} \
        -L ${L} \
        -s ${SMOOTHING} \
        -u

done

echo figure: 14, noised
sphere slepian_africa \
 -L ${L} \
 -n ${SNR} \
 -s ${SMOOTHING} \
 -u

for s in ${SIGMA[@]}; do
echo figure: 14, sigma: ${s}
python \
 ${PLOTS}/africa/denoising_slepian_africa.py \
 -n ${SNR} \
 -s ${s}
done

### Slepian Scale-Discretised Wavelets on Manifolds

[![Slepian Scale-Discretised Wavelets on Manifolds](https://img.shields.io/badge/DOI-10.48550/arXiv.2302.06006-pink.svg)](https://doi.org/10.48550/arXiv.2302.06006)
