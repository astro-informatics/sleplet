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

Wavelets are also widely used in various disciplines to analyse signals both in space and scale. Whilst many fields measure data on manifolds (i.e. the sphere), often data are only observed on a partial region of the manifold. Wavelets are a typical approach to data of this form, but the wavelet coefficients which overlap with the boundary become contaminated and must be removed for accurate analysis. Another approach is to estimate the region of missing data and to use existing whole-manifold methods for analysis. However, both approaches introduce uncertainty into any analysis. Slepian wavelets enable one to work directly with only the data present, thus avoiding the problems discussed above. Possible applications of Slepian wavelets to areas of research measuring data on the partial sphere include: gravitational/magnetic fields in geodesy; ground-based measurements in astronomy; measurements of whole-planet properties in planetary science; geomagnetism of the Earth; and in analyses of the cosmic microwave background.

# Statement of Need

Both `SHTools` [@Wieczorek2018] and `slepian_alpha` [@Simons2020] are examples of codes which allow one to compute Slepian functions on the sphere. In conjunction with `SSHT` [@McEwen2011], `S2LET` [@Leistedt2013] may be used to develop scale-discretised wavelets on the sphere. To the author's knowledge there is no known software which allows one to compute Slepian wavelets on the sphere or general manifolds/meshes.

# Research Based on SLEPLET

`SLEPLET` has been used in the development of [@Roddy2021], [@Roddy2022] and [@Roddy2023].

# Acknowledgements

The author would like to thank Jason D. McEwen for their advice and guidance on the mathematics behind `SLEPLET`. Further, the author would like to thank Zubair Khalid for providing their `MATLAB` implementation to compute the Slepian functions of a polar cap region, as well as the formulation
for a limited colatitude-longitude region. `SLEPLET` makes use of several libraries the author would like to acknowledge, in particular `SSHT` [@McEwen2011], `S2LET` [@Leistedt2013], and `libigl` [@Libigl2017].

# References
