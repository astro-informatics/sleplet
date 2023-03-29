from sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from sleplet.plot_methods import compute_amplitude_for_noisy_sphere_plots
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.region import Region
from sleplet.slepian_methods import slepian_inverse

# a
region = Region(mask_name="south_america")
f = SlepianSouthAmerica(L=128, region=region, noise=-10, smoothing=2)
f_sphere = slepian_inverse(f.coefficients, f.L, f.slepian)
amplitude = compute_amplitude_for_noisy_sphere_plots(f)
Plot(
    f_sphere, f.L, "fig_8_a", amplitude=amplitude, normalise=False, region=f.region
).execute()
