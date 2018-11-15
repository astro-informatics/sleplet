import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib import cm
import plotly.offline as py
from plotly.graph_objs import Figure, Surface, Layout
from plotly.graph_objs.layout import Scene, Margin
from plotly.graph_objs.layout.scene import XAxis, YAxis, ZAxis
import plotly.io as pio
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht


class SiftingConvolution(object):
    def __init__(self, L, resolution, fun=None):
        self.L = L
        self.resolution = resolution
        self.fun = fun

    @staticmethod
    def matplotlib_to_plotly(colour, pl_entries=255):
        '''
        converts matplotlib colourscale to a plotly colourscale

        Arguments:
            colour {string} -- matplotlib colour

        Keyword Arguments:
            pl_entries {bits} -- colour type (default: {255})

        Returns:
            plotly colour -- used in plotly plots
        '''

        cmap = cm.get_cmap(colour)

        h = 1.0 / (pl_entries - 1)
        pl_colorscale = []

        for k in range(pl_entries):
            C = map(np.uint8, np.array(cmap(k * h)[:3]) * 255)
            pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

        return pl_colorscale

    def place_on_sphere(self, north_pole=False):
        '''
        calculates a given function on the the sphere

        Arguments:
            fun {function} -- the function to go on the sphere

        Keyword Arguments:
            north_pole {bool} -- whether to place on north pole (default: {False})

        Returns:
            array -- new flm on the sphere
        '''

        # initiliase flm
        flm = np.zeros((self.resolution * self.resolution), dtype=complex)

        for ell in range(self.L):
            # place on north pole
            if north_pole:
                m = 0
                factor = np.sqrt((2 * ell + 1) / (4 * np.pi))
                # calc flm at index
                ind = ssht.elm2ind(ell, m)
                flm[ind] = factor * self.fun(ell, m)
            # place on the sphere
            else:
                for m in range(-ell, ell + 1):
                    # calc flm at index
                    ind = ssht.elm2ind(ell, m)
                    flm[ind] = self.fun(ell, m)

        return flm

    def translation(self, flm, alpha, beta):
        '''
        applies the translation operator to a given flm

        Arguments:
            flm {array} -- square array shape: res x res
            alpha {float} -- the phi angle direction
            beta {float} -- the theta angle direction

        Returns:
            array -- the new flm after the translation
        '''

        flm_trans = flm.copy()
        pix_i = ssht.theta_to_index(beta, self.L)
        pix_j = ssht.phi_to_index(-alpha, self.L)

        for ell in range(self.L):
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                ylm_harm = np.zeros((self.L * self.L), dtype=complex)
                ylm_harm[ind] = 1
                ylm_real = ssht.inverse(ylm_harm, self.L)
                flm_trans[ind] = flm[ind] * ylm_real[pix_i, pix_j]

        return flm_trans

    def setup_plot(self, f, method='MW', close=True, parametric=False,
                   parametric_scaling=[0.0, 0.5], color_range=None):
        '''
        function which creates the data for the matplotlib/plotly plot

        Arguments:
            f {function} -- inverse of flm

        Keyword Arguments:
            method {str} -- sampling scheme (default: {'MW'})
            close {bool} -- if true the full sphere is plotted without a gap (default: {True})
            parametric {bool} -- the radius of the object at a certain point is defined by the function (default: {False})
            parametric_scaling {list} -- used if Parametric=True, defines the radius of the shape at a particular angle (default: {[0.0, 0.5]})
            color_range {list} -- if set saturates the color bar in that range, else the function min and max is used (default: {None})

        Raises:
            Exception -- if band limit L is not the same size as function f

        Returns:
            tuple -- values for the plotting
        '''

        # add ability to choose color bar min max
        # and sort out shapes of the plots

        if method == 'MW_pole':
            if len(f) == 2:
                f, f_sp = f
            else:
                f, f_sp, phi_sp = f

        (thetas, phis) = ssht.sample_positions(
            self.resolution, Method=method, Grid=True)

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
            f_normalised = (f_plot - vmin / (vmax - vmin)) * \
                parametric_scaling[1] + parametric_scaling[0]

        # % Close plot.
        if close:
            (n_theta, n_phi) = ssht.sample_shape(
                self.resolution, Method=method)
            f_plot = np.insert(f_plot, n_phi, f[:, 0], axis=1)
            if parametric:
                f_normalised = np.insert(
                    f_normalised, n_phi, f_normalised[:, 0], axis=1)
            thetas = np.insert(thetas, n_phi, thetas[:, 0], axis=1)
            phis = np.insert(phis, n_phi, phis[:, 0], axis=1)

        # % Compute location of vertices.
        if parametric:
            (x, y, z) = ssht.spherical_to_cart(f_normalised, thetas, phis)
        else:
            (x, y, z) = ssht.s2_to_cart(thetas, phis)

        return x, y, z, f_plot, vmin, vmax

    def plotly_plot(self, f, auto_open, save_fig, dir='figures', filename='temp-plot'):
        '''
        creates basic plotly plot rather than matplotlib

        Arguments:
            f {function} -- inverse flm
        '''

        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self.setup_plot(f)

        data = [
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                colorscale=self.matplotlib_to_plotly('viridis'),
                cmin=vmin,
                cmax=vmax
            )]

        axis = dict(
            title='',
            showgrid=False,
            zeroline=False,
            ticks='',
            showticklabels=False
        )

        zoom = 1.4
        layout = Layout(
            scene=Scene(
                camera=dict(
                    eye=dict(x=1 / zoom, y=-1 / zoom, z=1 / zoom)
                ),
                xaxis=XAxis(axis),
                yaxis=YAxis(axis),
                zaxis=ZAxis(axis)
            ),
            margin=Margin(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )

        fig = Figure(data=data, layout=layout)

        if save_fig:
            png_filename = os.path.join(dir, 'png', filename + '.png')
            if not os.path.isfile(png_filename):
                pio.write_image(fig, png_filename)
            png_filename = os.path.join(dir, 'pdf', filename + '.pdf')
            if not os.path.isfile(png_filename):
                pio.write_image(fig, png_filename)

        py.plot(fig, filename=os.path.join(dir, filename + '.html'), auto_open=auto_open)

    @staticmethod
    def filename_angle(alpha, beta):
        # get fraction for filename
        alpha_num, alpha_den = (alpha / np.pi).as_integer_ratio()
        beta_num, beta_den = (beta / np.pi).as_integer_ratio()

        def helper(numerator, denominator):
            # if whole number
            if denominator == 1:
                # if 1 * pi
                if numerator == 1:
                    filename = 'pi'
                else:
                    filename = str(numerator) + 'pi'
            else:
                filename = str(numerator) + 'pi' + str(denominator)
            return filename

        if not alpha_num and not beta_num:
            filename = '_alpha-0_beta-0_'
        elif not alpha_num:
            filename = '_alpha-0_beta-'
            filename += helper(beta_num, beta_den)
            filename += '_'
        elif not beta_num:
            filename = '_alpha-'
            filename += helper(alpha_num, alpha_den)
            filename += '_beta-0_'
        else:
            filename = '_alpha-'
            filename += helper(alpha_num, alpha_den)
            filename += '_beta-'
            filename += helper(beta_num, beta_den)
            filename += '_'
        return filename

    def plot(self, dir, alpha, beta, plotting_type='real', auto_open=True, save_fig=False, method='north', gamma=0):
        filename = self.fun.func_name + '_' + method

        # test for plotting method
        if method == 'north':
            # adjust filename
            filename += '_'
            # place on north pole
            flm = self.place_on_sphere(north_pole=True)
            # inverse & plot
            f = ssht.inverse(flm, self.resolution)
        elif method == 'rotate':
            # adjust filename
            filename += self.filename_angle(alpha, beta)
            # place on north pole
            flm = self.place_on_sphere(north_pole=True)
            # rotate by alpha, beta, gamma
            flm_rot = ssht.rotate_flms(
                flm, alpha, beta, gamma, self.resolution)
            # inverse & plot
            f = ssht.inverse(flm_rot, self.resolution)
        elif method == 'translate':
            # adjust filename
            filename += self.filename_angle(alpha, beta)
            # place on sphere
            flm = self.place_on_sphere(north_pole=False)
            # translate by alpha, beta
            flm_trans = self.translation(flm, alpha, beta)
            # inverse & plot
            f = ssht.inverse(flm_trans, self.resolution)

        # check for plotting type
        if plotting_type == 'real':
            plot = f.real
        elif plotting_type == 'imag':
            plot = f.imag
        elif plotting_type == 'abs':
            plot = abs(f)

        # do plot
        filename += plotting_type
        self.plotly_plot(plot, auto_open, save_fig, dir, filename)
