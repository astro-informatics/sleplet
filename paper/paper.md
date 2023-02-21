---
title: "SLEPLET: Slepian scale-discretised wavelets in Python"
tags:
  - manifolds
  - python
  - slepian-functions
  - sphere
  - wavelets
authors:
  - name: Patrick J. Roddy
    orcid: 0000-0002-6271-1700
    affiliation: 1
affiliations:
  - name: Advanced Research Computing, University College London, UK
    index: 1
date: 16 February 2023
bibliography: paper.bib
---

# Summary

Wavelets are widely used in various disciplines to analyse signals both in space and scale. Whilst many fields measure data on manifolds (i.e. the sphere), often data are only observed on a partial region of the manifold. Wavelets are a typical approach to data of this form, but the wavelet coefficients which overlap with the boundary become contaminated and must be removed for accurate analysis. Another approach is to estimate the region of missing data and to use existing whole-manifold methods for analysis. However, both approaches introduce uncertainty into any analysis. Slepian wavelets enable one to work directly with only the data present, thus avoiding the problems discussed above. Possible applications of Slepian wavelets to areas of research measuring data on the partial sphere include: gravitational/magnetic fields in geodesy; ground-based measurements in astronomy; measurements of whole-planet properties in planetary science; geomagnetism of the Earth; and in analyses of the cosmic microwave background.

# Statement of Need

Many fields in science and engineering measure data that inherently live on non-Euclidean geometries, such as the sphere. Techniques developed in the Euclidean setting must be extended to other geometries. Due to recent interest in geometric deep learning, analogues of Euclidean techniques must also handle general manifolds or graphs. Often, data are only observed over partial regions of manifolds, and thus standard whole-manifold techniques may not yield accurate predictions. Slepian wavelets are designed for datasets like these. Slepian wavelets are built upon the eigenfunctions of the Slepian concentration problem of the manifold [@Slepian1961; @Landau1961; @Landau1962] - a set of bandlimited functions which are maximally concentrated within a given region. Wavelets are constructed through a tiling of the Slepian harmonic line by leveraging the existing scale-discretised framework [@Wiaux2008; @Leistedt2013]. Whilst these wavelets were inspired by spherical datasets, like in cosmology, the wavelet construction may be utilised for manifold or graph data.

To the author's knowledge, there is no public software which allows one to compute Slepian wavelets (or similar approach) on the sphere or general manifolds/meshes. `SHTools` [@Wieczorek2018] and `slepian_alpha` [@Simons2020] are examples of codes which allow one to compute Slepian functions. In conjunction with `SSHT` [@McEwen2011], `S2LET` [@Leistedt2013] may be used to develop scale-discretised wavelets on the whole sphere.

`SLEPLET` is a Python package for the construction of Slepian wavelets in the spherical and manifold (via meshes) settings. The API of `SLEPLET` has been designed in an object-orientated manner and is easily extendible. Upon installation, `SLEPLET` comes with two command line interfaces `sphere` and `mesh` which allows one to easily generate plots on the sphere and a set of meshes using `plotly`. Whilst these scripts are the primary intended use, `SLEPLET` may be used directly to generate the Slepian coefficients in the spherical/manifold setting and use methods to convert these into real space for visualisation or other intended purposes. The construction of the sifting convolution [@Roddy2021] was required to create Slepian wavelets. As a result, there are also many classes in harmonic space (rather than Slepian) which were used to demonstrate its effectiveness. `SLEPLET` has been used in the development of [@Roddy2021; @Roddy2022; @Roddy2022a; @Roddy2023].

# Demonstration

The following two methods are equivalent. This example leverages `plotly` to display the first Slepian wavelet of a South America region on a dataset of the topography of the Earth - as seen in [@Roddy2022].

## Command Line Interface

```sh
sphere slepian_wavelets -e 3 2 0 -L 128 -u
```

## API

```python
from sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

B = 3  # wavelet parameter
J_MIN = 2  # minimum wavelet scale
J = 0  # wavelet scale
L = 128  # bandlimit
MASK = "south_america"  # mask name

region = Region(mask_name=MASK)
f = SlepianWavelets(L, region=region, j=J)
f_sphere = slepian_inverse(f.coefficients, L, f.slepian)
Plot(
    f_sphere,
    L,
    f"slepian_wavelets_{MASK}_{B}B_{J_MIN}jmin_{J_MIN+J}j_L{128}",
    normalise=False,
    region=region,
).execute()
```

# Conclusions

Slepian wavelets have many potential applications in analyses of manifolds where data are only observed over a partial region. One such application is in CMB analyses, where observations are inherently made on the celestial sphere, and foreground emissions mask the centre of the data. In fields such as astrophysics and cosmology, datasets are increasingly large and thus require analysis at high resolutions for accurate predictions. Whilst Slepian wavelets may be trivially computed from a set of Slepian functions, the computation of the spherical Slepian functions themselves are computationally complex, where the matrix scales as $\mathcal{O}(L^{4})$. Although symmetries of this matrix and the spherical harmonics have been exploited, filling in this matrix is inherently slow due to the many integrals performed. The matrix is filled in parallel in `Python` using `concurrent.futures`, and the spherical harmonic transforms are computed in `C` using `SSHT`. This may be sped up further by utilising the new `ducc0` backend for `SSHT`, which may allow for a multithreaded solution. Ultimately, the eigenproblem must be solved to compute the Slepian functions, requiring sophisticated algorithms to balance speed and accuracy. Therefore, to work with high-resolution data such as these, one requires HPC methods on supercomputers with massive memory and storage. To this end, Slepian wavelets may be exploited at present at low resolutions, but further work is required for them to be fully scalable.

# Acknowledgements

The author would like to thank Jason D. McEwen for their advice and guidance on the mathematics behind `SLEPLET`. Further, the author would like to thank Zubair Khalid for providing their `MATLAB` implementation to compute the Slepian functions of a polar cap region, as well as the formulation
for a limited colatitude-longitude region. `SLEPLET` makes use of several libraries the author would like to acknowledge, in particular `SSHT` [@McEwen2011], `S2LET` [@Leistedt2013], and `libigl` [@Libigl2017].

# References
