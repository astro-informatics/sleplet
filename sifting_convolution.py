import sys
import os
import numpy as np
from matplotlib import cm
import plotly.offline as py
from plotly.graph_objs import Figure, Surface, Layout
from plotly.graph_objs.layout import Scene, Margin
from plotly.graph_objs.layout.scene import XAxis, YAxis, ZAxis
import plotly.io as pio
from fractions import Fraction
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht


class SiftingConvolution(object):
    def __init__(self, flm, config_dict, extra_filename=None):
        '''
        initialise class

        Arguments:
            flm {array} -- harmonic representation of function
            config_dict {dictionary} -- config options for class
        '''

        self.func_name = flm.func_name
        self.flm = flm()
        self.L = config_dict['L']
        self.res = config_dict['resolution']
        self.gamma_pi_fraction = config_dict['gamma_pi_fraction']
        self.inverted = config_dict['inverted']
        self.plotting_type = config_dict['plotting_type']
        self.method = config_dict['method']
        self.auto_open = config_dict['auto_open']
        self.save_fig = config_dict['save_fig']
        self.fig_directory = config_dict['fig_directory']
        self.extra_filename = extra_filename

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

    def place_flm_on_north_pole(self, flm):
        '''
        place given flm on the north pole of the sphere

        Arguments:
            flm {array} -- harmonic representation of function

        Returns:
            array -- flm on the north pole
        '''

        for ell in range(self.L):
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                if m == 0:
                    flm[ind] *= np.sqrt((2 * ell + 1) / (4 * np.pi))
                else:
                    flm[ind] = 0
        return flm

    def rotation(self, flm, alpha, beta, gamma):
        '''
        rotates given flm on the sphere by alpha/beta/gamma

        Arguments:
            flm {array} -- harmonic representation of function

        Returns:
            array -- rotated flm
        '''

        flm = ssht.rotate_flms(
            flm, alpha, beta, gamma, self.res)
        return flm

    def translation(self, flm, alpha, beta):
        '''
        tranlsates given flm on the sphere by alpha/beta

        Arguments:
            flm {array} -- harmonic representation of function

        Returns:
            array -- translated flm
        '''

        # compute pixels of \omega'
        pix_i = ssht.theta_to_index(beta, self.L)
        pix_j = ssht.phi_to_index(alpha, self.L)

        def helper(ind):
            '''
            computes the value of the spherical harmonic at omega'

            Arguments:
                flm {array} -- harmonic representation of function
                ind {int} -- index of l & m

            Returns:
                float -- Y_{\ell m}(\omega')
            '''

            ylm_harmonic = np.zeros((self.L * self.L), dtype=complex)
            ylm_harmonic[ind] = 1
            ylm_pixel = ssht.inverse(ylm_harmonic, self.L)
            ylm_omega = ylm_pixel[pix_i, pix_j]
            if self.func_name == 'dirac_delta':
                return np.conj(ylm_omega)
            else:
                return ylm_omega

        for ell in range(self.L):
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                if self.func_name == 'dirac_delta':
                    flm[ind] = helper(ind)
                else:
                    if m == 0:
                        flm[ind] *= helper(ind)
                    else:
                        flm[ind] = helper(ind)
        return flm

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

        if method == 'MW_pole':
            if len(f) == 2:
                f, f_sp = f
            else:
                f, f_sp, phi_sp = f

        (thetas, phis) = ssht.sample_positions(
            self.res, Method=method, Grid=True)

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
                self.res, Method=method)
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

    def plotly_plot(self, f, filename):
        '''
        creates basic plotly plot rather than matplotlib

        Arguments:
            f {function} -- inverse flm
            filename {str} -- filename for html/png/pdf plot
        '''

        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self.setup_plot(f)

        zoom = 1.4
        camera = dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=1 / zoom, y=-1 / zoom, z=1 / zoom)
        )

        # some flm are inverted i.e. topography map of the Earth
        if self.inverted:
            # invert axis
            x *= -1
            y *= -1
            z *= -1
            # invert camera view
            camera['up']['z'] = -1
            for coord in camera['eye']:
                camera['eye'][coord] *= -1

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

        layout = Layout(
            scene=Scene(
                dragmode='orbit',
                camera=camera,
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
        directory = os.path.join(os.pardir, os.pardir, self.fig_directory)

        # if save_fig is true then print as png and pdf in their directories
        if self.save_fig:
            png_filename = os.path.join(directory, 'png', filename + '.png')
            pio.write_image(fig, png_filename)
            pdf_filename = os.path.join(directory, 'pdf', filename + '.pdf')
            pio.write_image(fig, pdf_filename)

        # create html and open if auto_open is true
        html_filename = os.path.join(directory, 'html', filename + '.html')
        py.plot(fig, filename=html_filename, auto_open=self.auto_open)

    def filename_angle(self, alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction):
        '''
        middle part of filename

        Returns:
            str -- filename
        '''

        # get numerator/denominator for filename
        alpha = Fraction(alpha_pi_fraction).limit_denominator()
        beta = Fraction(beta_pi_fraction).limit_denominator()
        gamma = Fraction(gamma_pi_fraction).limit_denominator()
        alpha_num, alpha_den = alpha.numerator, alpha.denominator
        beta_num, beta_den = beta.numerator, beta.denominator
        gamma_num, gamma_den = gamma.numerator, gamma.denominator

        def helper(numerator, denominator):
            '''
            create filename for alpha/beta as multiple of pi

            Arguments:
                numerator {int} -- alpha/beta numerator
                denominator {int} -- alpha/beta denominator

            Returns:
                str -- middle of filename
            '''

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

        # if alpha = beta = 0
        if not alpha_num and not beta_num:
            filename = 'alpha-0_beta-0_'
        # if alpha = 0
        elif not alpha_num:
            filename = 'alpha-0_beta-'
            filename += helper(beta_num, beta_den)
            filename += '_'
        # if beta = 0
        elif not beta_num:
            filename = 'alpha-'
            filename += helper(alpha_num, alpha_den)
            filename += '_beta-0_'
        # if alpha != 0 && beta !=0
        else:
            filename = 'alpha-'
            filename += helper(alpha_num, alpha_den)
            filename += '_beta-'
            filename += helper(beta_num, beta_den)
            filename += '_'

        # if rotation with gamma != 0
        if gamma_num:
            filename += 'gamma-'
            filename += helper(gamma_num, gamma_den)
            filename += '_'
        return filename

    def plot(self, alpha_pi_fraction=0.0, beta_pi_fraction=0.0, reality=False):
        '''
        master plotting method

        Keyword Arguments:
            alpha_pi_fraction {float} -- fraction of pi i.e. 1.75 (default: {0.0})
            beta_pi_fraction {float} -- fraction of pi i.e. 0.25 (default: {0.0})
        '''

        alpha = alpha_pi_fraction * np.pi
        beta = beta_pi_fraction * np.pi
        gamma = self.gamma_pi_fraction * np.pi
        filename = self.func_name + '_'
        if self.extra_filename:
            filename += self.extra_filename

        # test for plotting method
        if self.method == 'north':
            # Dirac delta not defined on sphere
            if self.func_name == 'dirac_delta':
                flm = self.place_flm_on_north_pole(self.flm)
                # adjust filename
                filename += self.method + '_'
            else:
                flm = self.flm
        elif self.method == 'rotate':
            # adjust filename
            filename += self.method + '_'
            filename += self.filename_angle(alpha_pi_fraction,
                                            beta_pi_fraction, self.gamma_pi_fraction)
            # Dirac delta not defined on sphere
            if self.func_name == 'dirac_delta':
                flm = self.place_flm_on_north_pole(self.flm)
            else:
                flm = self.flm
            # rotate by alpha, beta
            flm = self.rotation(flm, alpha, beta, gamma)
        elif self.method == 'translate':
            # adjust filename
            filename += self.method + '_'
            # don't add gamma if translation
            filename += self.filename_angle(alpha_pi_fraction,
                                            beta_pi_fraction, gamma_pi_fraction=0.0)
            # translate by alpha, beta
            flm = self.translation(self.flm, alpha, beta)

        # inverse & plot
        f = ssht.inverse(flm, self.res, Reality=reality)
        filename += 'L-' + str(self.L) + '_'
        filename += 'res-' + str(self.res) + '_'

        # check for plotting type
        if self.plotting_type == 'real':
            plot = f.real
        elif self.plotting_type == 'imag':
            plot = f.imag
        elif self.plotting_type == 'abs':
            plot = abs(f)
        elif self.plotting_type == 'sum':
            plot = f.real + f.imag

        # do plot
        filename += self.plotting_type
        self.plotly_plot(plot, filename)
