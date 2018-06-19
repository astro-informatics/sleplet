import sys, os
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import scipy.io as sio
from matplotlib import cm, colors, colorbar, gridspec
import plotly.offline as py
from plotly.graph_objs import *

sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht


class ShiftingConvolution:
    def __init__(self, L_comp, L_plot, m, gamma, beta, alpha):
        self.L_comp = L_comp
        self.L_plot = L_plot
        self.m = m
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    # Generate spherical harmonics.
    def dirac_delta(self, L):
        fun = lambda l, m: 1
        flm = self.north_pole(L, fun)

        return flm

    def gaussian(self, L, sig=1):
        fun = lambda l, m: np.exp(-l * (l + 1)) / (2 * sig * sig)
        flm = self.north_pole(L, fun)

        return flm

    def squashed_gaussian(self, L, sig=1):
        fun = lambda l, m: np.sin(m) * np.exp(-l * (l + 1)) / (2 * sig * sig)
        flm = self.north_pole(L, fun)

        return flm

    def north_pole(self, L, fun):
        flm = np.zeros((L * L), dtype=complex)

        for el in range(L):
            for em in range(-el, el + 1):
                ind = ssht.elm2ind(el, m=0)
                factor = np.sqrt((2 * el + 1) / (4 * np.pi))
                flm[ind] = factor * fun(el, m=3) * (1.0 + 1j * 0.0)

        return flm

    def earth(self):
        matfile = os.path.join(
            os.environ['SSHT'], 'src', 'matlab', 'data',
            'EGM2008_Topography_flms_L0128')
        mat_contents = sio.loadmat(matfile)
        flm = np.ascontiguousarray(mat_contents['flm'][:, 0])

        return flm

    # Compute function on the sphere.
    def func_on_sphere(self, flm, L):
        f = ssht.inverse(flm, L)

        return f

    # Rotate spherical harmonic
    def rotate(self, flm, L, alpha, beta, gamma):
        flm_rot = ssht.rotate_flms(
            flm, alpha, beta, gamma, L)

        return flm_rot

    def sift_conv(self, flm, L):
        flm_conv = np.zeros((L * L), dtype=complex)

        for el in range(L):
            for m in range(-el, el + 1):
                ind = ssht.elm2ind(el, m)
        #         flm[ind] = f * 1j
        # flm_conv = flm * 1j
        return flm_conv

    def test_plot(self, f, L, method='MW', close=True, parametric=False,
                  parametric_scaling=[0.0, 0.5], output_file=None, show=True,
                  color_bar=True, units=None, color_range=None, axis=True):
        # add ability to choose color bar min max
        # and sort out shapes of the plots

        if method == 'MW_pole':
            if len(f) == 2:
                f, f_sp = f
            else:
                f, f_sp, phi_sp = f

        (thetas, phis) = ssht.sample_positions(L, Method=method, Grid=True);

        if (thetas.size != f.size):
            raise Exception('Band limit L deos not match that of f')

        f_plot = f.copy()

        f_max = f_plot.max()
        f_min = f_plot.min()

        if color_range is None:
            vmin = f_min
            vmax = f_max
        else:
            vmin = color_range[0]
            vmax = color_range[1]
            f_plot[f_plot < color_range[0]] = color_range[0]
            f_plot[f_plot > color_range[1]] = color_range[1]
            f_plot[f_plot == -1.56E30] = np.nan

        # % Compute position scaling for parametric plot.
        if parametric:
            f_normalised = (f_plot - vmin / (vmax - vmin)) * parametric_scaling[1] + parametric_scaling[0]

        # % Close plot.
        if close:
            (n_theta, n_phi) = ssht.sample_shape(L, Method=method)
            f_plot = np.insert(f_plot, n_phi, f[:, 0], axis=1)
            if parametric:
                f_normalised = np.insert(f_normalised, n_phi, f_normalised[:, 0], axis=1)
            thetas = np.insert(thetas, n_phi, thetas[:, 0], axis=1)
            phis = np.insert(phis, n_phi, phis[:, 0], axis=1)

        # % Compute location of vertices.
        if parametric:
            (x, y, z) = ssht.spherical_to_cart(f_normalised, thetas, phis)
        else:
            (x, y, z) = ssht.s2_to_cart(thetas, phis)

        # % Plot.
        fig = plt.figure(figsize=plt.figaspect(1.1))
        gs = gridspec.GridSpec(2, 1, height_ratios=[10, 0.5])
        ax = fig.add_subplot(gs[0], projection='3d')
        ax_cbar = fig.add_subplot(gs[1])
        norm = colors.Normalize()
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.jet(norm(f_plot)))
        if not axis:
            ax.set_axis_off()

        if color_bar:
            cmap = cm.jet
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cb1 = colorbar.ColorbarBase(ax_cbar, cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
            if units != None:
                cb1.set_label(units)

        # output (to file and screan)
        if output_file != None:
            plt.savefig(output_file)
        if show:
            plt.show()

        data = Data([
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                colorscale='Jet',
                cmin=vmin,
                cmax=vmax,
            )
        ])

        axis = dict(title='')

        layout = Layout(
            scene=Scene(
                camera=dict(
                    eye=dict(x=1.25, y=-1.25, z=1.25)
                ),
                xaxis=XAxis(axis),
                yaxis=YAxis(axis),
                zaxis=ZAxis(axis)
            )
        )

        fig = Figure(data=data, layout=layout)

        py.plot(fig)

    def dirac_delta_plot(self):
        flm = self.dirac_delta(self.L_comp)
        f = self.func_on_sphere(flm, self.L_comp)
        flm_rot = self.rotate(flm, self.L_comp, self.alpha, self.beta, self.gamma)
        f_rot = self.func_on_sphere(flm_rot, self.L_comp)
        # flm_conv = self.sift_conv(flm, self.L_comp, self.alpha, self.beta, self.gamma)
        # f_conv = self.func_on_spher(flm_conv, self.L_comp)
        # ssht.plot_sphere(f.real, self.L, Output_File='diracdelta_north.png')
        # ssht.plot_sphere(f_rot.real, 64, Output_File='diracdelta_rot.png')
        self.test_plot(f.real, self.L_plot, parametric=False, output_file='diracdelta_northpole.png')
        # self.test_plot(f_rot.real, self.L_plot, close=False, output_file='diracdelta_rotated.png')

    def gaussian_plot(self):
        flm = self.gaussian(self.L_comp)
        f = self.func_on_sphere(flm, self.L_comp)
        flm_rot = self.rotate(flm, self.L_comp, self.alpha, self.beta, self.gamma)
        f_rot = self.func_on_sphere(flm_rot, self.L_comp)
        self.test_plot(f.real, self.L_plot)
        self.test_plot(f_rot.real, self.L_plot)

    def squashed_gaussian_plot(self):
        flm = self.squashed_gaussian(self.L_comp)
        f = self.func_on_sphere(flm, self.L_comp)
        flm_rot = self.rotate(flm, self.L_comp, self.alpha, self.beta, self.gamma)
        f_rot = self.func_on_sphere(flm_rot, self.L_comp)
        self.test_plot(f.real, self.L_plot)
        self.test_plot(f_rot.real, self.L_plot)

    def earth_plot(self):
        flm = self.earth()
        f = self.func_on_sphere(flm, self.L_comp)
        flm_rot = self.rotate(flm, self.L_comp, self.alpha, self.beta, self.gamma)
        f_rot = self.func_on_sphere(flm_rot, self.L_comp)
        self.test_plot(f.real, self.L_plot)
        self.test_plot(f_rot.real, self.L_plot)


if __name__ == '__main__':
    # Define parameters.
    L_comp = 64
    L_plot = 64
    m = 0
    gamma = 0
    beta = -np.pi / 4  # theta
    alpha = np.pi / 4  # phi

    sc = ShiftingConvolution(L_comp, L_plot, m, gamma, beta, alpha)
    # sc.dirac_delta_plot()
    # sc.gaussian_plot()
    # sc.squashed_gaussian_plot()
    sc.earth_plot()
