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

## Paper Figures

To recreate the figures from the below papers, one may use the CLI or the API.
For those which don't use the `mesh` or `sphere` CLIs, the relevant API code
isn't provided as it is contained within the
[examples folder](https://github.com/astro-informatics/sleplet/tree/main/examples).

### Sifting Convolution on the Sphere

[![Sifting Convolution on the Sphere](https://img.shields.io/badge/DOI-10.1109/LSP.2021.3050961-pink.svg)](https://dx.doi.org/10.1109/LSP.2021.3050961)

#### Fig. 1

```sh
for ell in $(seq 2 -1 1); do
    sphere harmonic_gaussian -a 0.75 -b 0.125 -e ${ell} 1 -L 128 -m translate -o
done
```

```python
import numpy as np
import pyssht as ssht

from sleplet.functions.flm.harmonic_gaussian import HarmonicGaussian
from sleplet.plotting.create_plot_sphere import Plot

for ell in range(2, 0, -1):
    f = HarmonicGaussian(L=128, l_sigma=10**ell, m_sigma=10)
    flm = f.translate(alpha=0.75 * np.pi, beta=0.125 * np.pi)
    f_sphere = ssht.inverse(flm, f.L, Method="MWSS")
    Plot(f_sphere, f.L, f"fig_1_ell_{ell}", annotations=[]).execute()
```

#### Fig. 2

```sh
sphere earth -L 128
```

```python
import pyssht as ssht

from sleplet.functions.flm.earth import Earth
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.harmonic_methods import rotate_earth_to_south_america

f = Earth(L=128)
flm = rotate_earth_to_south_america(f.coefficients, f.L)
f_sphere = ssht.inverse(flm, f.L, Method="MWSS")
Plot(f_sphere, f.L, "fig_2").execute()
```

#### Fig. 3

```sh
for ell in $(seq 2 -1 1); do
    sphere harmonic_gaussian -c earth -e ${ell} 1 -L 128
done
```

```python
import pyssht as ssht

from sleplet.functions.flm.earth import Earth
from sleplet.functions.flm.harmonic_gaussian import HarmonicGaussian
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.harmonic_methods import rotate_earth_to_south_america

for ell in range(2, 0, -1):
    f = HarmonicGaussian(L=128, l_sigma=10**ell, m_sigma=10)
    g = Earth(L=128)
    flm = f.convolve(f.coefficients, g.coefficients.conj())
    flm_rot = rotate_earth_to_south_america(flm, f.L)
    f_sphere = ssht.inverse(flm_rot, f.L, Method="MWSS")
    Plot(f_sphere, f.L, f"fig_3_ell_{ell}").execute()
```

### Slepian Scale-Discretised Wavelets on the Sphere

[![Slepian Scale-Discretised Wavelets on the Sphere](https://img.shields.io/badge/DOI-10.1109/TSP.2022.3233309-pink.svg)](https://dx.doi.org/10.1109/TSP.2022.3233309)

#### Fig. 2

```sh
python -m examples.arbitrary.south_america.tiling_south_america
```

#### Fig. 3

```sh
export SLEPIAN_MASK="south_america"
# a
sphere earth -L 128 -s 2 -u
# b
sphere slepian_south_america -L 128 -s 2 -u
```

```python
import pyssht as ssht

from sleplet.functions.flm.earth import Earth
from sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.harmonic_methods import rotate_earth_to_south_america
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

# a
f = Earth(L=128, smoothing=2)
flm = rotate_earth_to_south_america(f.coefficients, f.L)
f_sphere = ssht.inverse(flm, f.L, Method="MWSS")
Plot(f_sphere, f.L, "fig_3_a", normalise=False).execute()
# b
region = Region(mask_name="south_america")
g = SlepianSouthAmerica(L=128, region=region, smoothing=2)
g_sphere = slepian_inverse(g.coefficients, g.L, g.slepian)
Plot(g_sphere, g.L, "fig_3_b", normalise=False, region=g.region).execute()
```

#### Fig. 4

```sh
export SLEPIAN_MASK="south_america"
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
    Plot(f_sphere, f.L, f"fig_4_p_{p}", normalise=False, region=f.region).execute()
```

#### Fig. 5

```sh
python -m examples.arbitrary.south_america.eigenvalues_south_america
```

#### Fig. 6

```sh
export SLEPIAN_MASK="south_america"
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
for j in [None, *list(range(5))]:
    f = SlepianWavelets(L=128, region=region, B=3, j_min=2, j=j)
    f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
    Plot(f_sphere, f.L, f"fig_6_j_{j}", normalise=False, region=f.region).execute()
```

#### Fig. 7

```sh
export SLEPIAN_MASK="south_america"
# a
sphere slepian_wavelet_coefficients_south_america -L 128 -s 2 -u
# b-f
for j in $(seq 0 4); do
    sphere slepian_wavelet_coefficients_south_america -e 3 2 ${j} -L 128 -s 2 -u
done
```

```python
from sleplet.functions.fp.slepian_wavelet_coefficients_south_america import (
    SlepianWaveletCoefficientsSouthAmerica,
)
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

region = Region(mask_name="south_america")
for j in [None, *list(range(5))]:
    f = SlepianWaveletCoefficientsSouthAmerica(
        L=128, region=region, B=3, j_min=2, j=j, smoothing=2
    )
    f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
    Plot(f_sphere, f.L, f"fig_7_j_{j}", normalise=False, region=f.region).execute()
```

#### Fig. 8

```sh
export SLEPIAN_MASK="south_america"
# a
sphere slepian_south_america -L 128 -n -10 -s 2 -u
# b-d
for s in 2 3 5; do
    python -m examples.arbitrary.south_america.denoising_slepian_south_america -n -10 -s ${s}
done
```

```python
from sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.scripts.plotting_on_sphere import compute_amplitude_for_noisy_plots
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

# a
region = Region(mask_name="south_america")
f = SlepianSouthAmerica(L=128, region=region, noise=-10, smoothing=2)
f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
amplitude = compute_amplitude_for_noisy_plots(f)
Plot(
    f_sphere, f.L, "fig_8_a", amplitude=amplitude, normalise=False, region=f.region
).execute()
```

#### Fig. 9

```sh
export SLEPIAN_MASK="africa"
# a
sphere earth -L 128 -s 2 -u -v africa
# b
sphere slepian_africa -L 128 -s 2 -u
```

```python
import pyssht as ssht

from sleplet.functions.flm.earth import Earth
from sleplet.functions.fp.slepian_africa import SlepianAfrica
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.harmonic_methods import rotate_earth_to_africa
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

# a
f = Earth(L=128, smoothing=2)
flm = rotate_earth_to_africa(f.coefficients, f.L)
f_sphere = ssht.inverse(flm, f.L, Method="MWSS")
Plot(f_sphere, f.L, "fig_9_a", normalise=False).execute()
# b
region = Region(mask_name="africa")
g = SlepianAfrica(L=128, region=region, smoothing=2)
g_sphere = slepian_inverse(g.coefficients, g.L, g.slepian)
Plot(g_sphere, g.L, "fig_9_b", normalise=False, region=g.region).execute()
```

#### Fig. 10

```sh
python -m examples.arbitrary.africa.eigenvalues_africa
```

#### Fig. 11

```sh
export SLEPIAN_MASK="africa"
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
    Plot(f_sphere, f.L, f"fig_11_p{p}", normalise=False, region=f.region).execute()
```

#### Fig. 12

```sh
export SLEPIAN_MASK="africa"
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
for j in [None, *list(range(6))]:
    f = SlepianWavelets(L=128, region=region, B=3, j_min=2, j=j)
    f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
    Plot(f_sphere, f.L, f"fig_12_j_{j}", normalise=False, region=f.region).execute()
```

#### Fig. 13

```sh
export SLEPIAN_MASK="africa"
# a
sphere slepian_wavelet_coefficients_africa -L 128 -s 2 -u
# b
for j in $(seq 0 5); do
    sphere slepian_wavelet_coefficients_africa -e 3 2 ${j} -L 128 -s 2 -u
done
```

```python
from sleplet.functions.fp.slepian_wavelet_coefficients_africa import (
    SlepianWaveletCoefficientsAfrica,
)
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

region = Region(mask_name="africa")
for j in [None, *list(range(6))]:
    f = SlepianWaveletCoefficientsAfrica(
        L=128, region=region, B=3, j_min=2, j=j, smoothing=2
    )
    f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
    Plot(f_sphere, f.L, f"fig_13_j_{j}", normalise=False, region=f.region).execute()
```

#### Fig. 14

```sh
export SLEPIAN_MASK="africa"
# a
sphere slepian_africa -L 128 -n -10 -s 2 -u
# b-d
for s in 2 3 5; do
    python -m examples.arbitrary.africa.denoising_slepian_africa -n -10 -s ${s}
done
```

```python
from sleplet.functions.fp.slepian_africa import SlepianAfrica
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.scripts.plotting_on_sphere import compute_amplitude_for_noisy_plots
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import slepian_inverse

# a
region = Region(mask_name="africa")
f = SlepianAfrica(L=128, region=region, noise=-10, smoothing=2)
f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
amplitude = compute_amplitude_for_noisy_plots(f)
Plot(
    f_sphere, f.L, "fig_14_a", amplitude=amplitude, normalise=False, region=f.region
).execute()
```

### Slepian Scale-Discretised Wavelets on Manifolds

[![Slepian Scale-Discretised Wavelets on Manifolds](https://img.shields.io/badge/DOI-10.48550/arXiv.2302.06006-pink.svg)](https://doi.org/10.48550/arXiv.2302.06006)

#### Fig. 2

```sh
for r in $(seq 2 9); do
    mesh homer -e ${r} -u
done
```

```python
from sleplet.meshes.classes.mesh import Mesh
from sleplet.meshes.harmonic_coefficients.mesh_basis_functions import MeshBasisFunctions
from sleplet.plotting.create_plot_mesh import Plot
from sleplet.utils.harmonic_methods import mesh_inverse

mesh = Mesh("homer")
for r in range(2, 10):
    f = MeshBasisFunctions(mesh, rank=r)
    f_mesh = mesh_inverse(f.mesh, f.coefficients)
    Plot(mesh, f"fig_2_r_{r}", f_mesh, normalise=False).execute()
```

#### Fig. 4

```sh
python -m examples.mesh.mesh_tiling homer
```

#### Fig. 5

```sh
python -m examples.mesh.mesh_region homer
```

#### Fig. 6

```sh
for p in 0 9 24 49 99 199; do
    mesh homer -m slepian_functions -e ${p} -u -z
done
```

```python
from sleplet.meshes.classes.mesh import Mesh
from sleplet.meshes.slepian_coefficients.mesh_slepian_functions import (
    MeshSlepianFunctions,
)
from sleplet.plotting.create_plot_mesh import Plot
from sleplet.utils.slepian_methods import slepian_mesh_inverse

mesh = Mesh("homer", zoom=True)
for p in [0, 9, 24, 49, 99, 199]:
    f = MeshSlepianFunctions(mesh, rank=p)
    f_mesh = slepian_mesh_inverse(f.mesh_slepian, f.coefficients)
    Plot(mesh, f"fig_6_p_{p}", f_mesh, normalise=False, region=True).execute()
```

#### Fig. 7

```sh
python -m examples.mesh.mesh_slepian_eigenvalues homer
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

```python
from sleplet.meshes.classes.mesh import Mesh
from sleplet.meshes.slepian_coefficients.mesh_slepian_wavelets import (
    MeshSlepianWavelets,
)
from sleplet.plotting.create_plot_mesh import Plot
from sleplet.utils.slepian_methods import slepian_mesh_inverse

mesh = Mesh("homer", zoom=True)
for j in [None, *list(range(5))]:
    f = MeshSlepianWavelets(mesh, B=3, j_min=2, j=j)
    f_mesh = slepian_mesh_inverse(f.mesh_slepian, f.coefficients)
    Plot(mesh, f"fig_8_j_{j}", f_mesh, normalise=False, region=True).execute()
```

#### Fig. 9

```sh
mesh homer -m field -u
```

```python
from sleplet.meshes.classes.mesh import Mesh
from sleplet.meshes.harmonic_coefficients.mesh_field import MeshField
from sleplet.plotting.create_plot_mesh import Plot
from sleplet.utils.harmonic_methods import mesh_inverse

mesh = Mesh("homer")
f = MeshField(mesh)
f_mesh = mesh_inverse(f.mesh, f.coefficients)
Plot(mesh, "fig_9", f_mesh, normalise=False).execute()
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

```python
from sleplet.meshes.classes.mesh import Mesh
from sleplet.meshes.slepian_coefficients.mesh_slepian_wavelet_coefficients import (
    MeshSlepianWaveletCoefficients,
)
from sleplet.plotting.create_plot_mesh import Plot
from sleplet.utils.slepian_methods import slepian_mesh_inverse

mesh = Mesh("homer", zoom=True)
for j in [None, *list(range(5))]:
    f = MeshSlepianWaveletCoefficients(mesh, B=3, j_min=2, j=j)
    f_mesh = slepian_mesh_inverse(f.mesh_slepian, f.coefficients)
    Plot(mesh, f"fig_10_j_{j}", f_mesh, normalise=False, region=True).execute()
```

#### Fig. 11

```sh
# a
mesh homer -m slepian_field -u -z
# b
mesh homer -m slepian_field -n -5 -u -z
# c
python -m examples.mesh.denoising_slepian_mesh homer -n -5 -s 2
```

```python
from sleplet.meshes.classes.mesh import Mesh
from sleplet.meshes.slepian_coefficients.mesh_slepian_field import (
    MeshSlepianField,
)
from sleplet.plotting.create_plot_mesh import Plot
from sleplet.scripts.plotting_on_mesh import compute_amplitude_for_noisy_plots
from sleplet.utils.slepian_methods import slepian_mesh_inverse

mesh = Mesh("homer", zoom=True)
# a
f = MeshSlepianField(mesh)
f_mesh = slepian_mesh_inverse(f.mesh_slepian, f.coefficients)
Plot(mesh, "fig_11_a", f_mesh, normalise=False, region=True).execute()
# b
g = MeshSlepianField(mesh, noise=-5)
g_mesh = slepian_mesh_inverse(g.mesh_slepian, g.coefficients)
amplitude = compute_amplitude_for_noisy_plots(g)
Plot(
    mesh, "fig_11_b", g_mesh, amplitude=amplitude, normalise=False, region=True
).execute()
```

#### Fig. 12

```sh
for f in cheetah dragon bird teapot cube; do
    python -m examples.mesh.mesh_region ${f}
done
```

#### Tab. 1

```sh
python -m examples.mesh.produce_table
```
