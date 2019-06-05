import sys
import os
import numpy as np
from matplotlib import cm
import plotly.offline as py
from plotly.graph_objs import Figure, Surface, Layout
from plotly.graph_objs.layout import Margin, Scene
from plotly.graph_objs.layout.scene import XAxis, YAxis, ZAxis
import plotly.io as pio
from fractions import Fraction
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
import scipy.special as sp
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht


class SiftingConvolution:
    def __init__(self, flm, flm_name, config, glm=None, glm_name=None):
        '''
        initialise class

        Arguments:
            flm {array} -- harmonic representation of function
            flm_name {str} -- function name of flm
            config {dictionary} -- config options for class

        Keyword Arguments:
            glm {array} -- kernel to convolve with (default: {None})
            glm_name {array} -- function name of glm (default: {None})
        '''

        self.auto_open = config['auto_open']
        self.flm_name = flm_name
        self.flm = flm
        self.glm = glm
        self.L = config['L']
        self.ncpu = config['ncpu']
        self.L_scipy_max = 86
        self.location = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.method = config['sampling']
        self.reality = config['reality']
        self.resolution = self.calc_resolution(config)
        self.routine = config['routine']
        self.save_fig = config['save_fig']
        self.type = config['type']
        if self.glm is not None:
            self.glm_name = glm_name

    # -----------------------------------
    # ---------- flm functions ----------
    # -----------------------------------

    def rotation(self, flm, alpha, beta, gamma):
        '''
        rotates given flm on the sphere by alpha/beta/gamma

        Arguments:
            flm {array} -- harmonic representation of function
            alpha {float} -- phi angle
            beta {float} -- theta angle
            gamma {float} -- gamma angle

        Returns:
            array -- rotated flm
        '''

        flm_rot = ssht.rotate_flms(
            flm, alpha, beta, gamma, self.L)
        return flm_rot

    def translation(self, flm):
        '''
        translates given flm on the sphere by alpha/beta

        Arguments:
            flm {array} -- harmonic representation of function

        Returns:
            array -- translated flm
        '''

        # numpy binary filename
        filename = os.path.join(self.location, 'npy', 'trans_dd_L-' + str(self.L) + '_' + self.filename_angle(
            self.alpha_pi_fraction, self.beta_pi_fraction) + 'samp-' + str(self.method) + '.npy')

        # check if file of translated dirac delta already
        # exists otherwise calculate translated dirac delta
        # translation is slow for large L
        if os.path.exists(filename):
            glm = np.load(filename)
        else:
            glm = self.translate_dirac_delta(filename)

        # convolve with flm
        if self.flm_name == 'dirac_delta':
            flm_conv = glm
        else:
            flm_conv = self.convolution(flm, glm)
        return flm_conv

    def translate_dirac_delta(self, filename):
        '''
        translates the dirac delta on the sphere to alpha/beta

        Keyword Arguments:
            filename {str} -- filename to save array (default: {None})

        Returns:
            array -- translated flm
        '''

        # initialise array
        flm_trans = np.zeros((self.L * self.L), dtype=complex)

        # scipy fails above L = 86
        if self.L < self.L_scipy_max + 1:
            L_max = self.L
        else:
            L_max = self.L_scipy_max

        # scipy method
        flm_trans = self.translate_dd_scipy(flm_trans, L_max)

        # choose method based on number of cores
        if self.ncpu == 1:
            flm_trans = self.translate_dd_serial(flm_trans)
        else:
            flm_trans = self.translate_dd_parallel(flm_trans)

        # save to speed up for future
        if filename is not None:
            np.save(filename, flm_trans)

        return flm_trans

    def convolution(self, flm, glm):
        '''
        computes the sifting convolution of two arrays

        Arguments:
            flm {array} -- input flm of the class
            glm {array} -- kernal map to convolve with

        Returns:
            array -- convolved output
        '''

        # translation/convolution are not real for general
        # function so turn off reality except for Dirac delta
        self.reality = False

        return flm * np.conj(glm)

    # ---------------------------------
    # ---------- translation ----------
    # ---------------------------------

    def translate_dd_scipy(self, flm, L):
        '''
        scipy method to translate dirac delta up to L=86

        Arguments:
            flm {array} -- harmonic representation of function
            L {int} -- value of L<=86

        Returns:
            array -- translated dirac delta
        '''
        alpha, beta = self.calc_nearest_grid_point(
            self.alpha_pi_fraction, self.beta_pi_fraction)

        for ell in range(L):
            m = 0
            ind = ssht.elm2ind(ell, m)
            flm[ind] = sp.sph_harm(
                m, ell, -alpha, beta)
            for m in range(1, ell + 1):
                ind_pm = ssht.elm2ind(ell, m)
                ind_nm = ssht.elm2ind(ell, -m)
                flm[ind_pm] = sp.sph_harm(
                    m, ell, -alpha, beta)
                flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])
        return flm

    def translate_dd_serial(self, flm):
        '''
        serial method to translate dirac delta - faster locally

        Arguments:
            flm {array} -- translated function up to L_scipy_max

        Returns:
            array -- translated dirac delta
        '''
        for ell in range(self.L_scipy_max, self.L):
            ind = ssht.elm2ind(ell, m=0)
            conj_pixel_val = self.calc_pixel_value(ind)
            flm[ind] = conj_pixel_val
            for m in range(1, ell + 1):
                print(ell, m)
                ind_pm = ssht.elm2ind(ell, m)
                ind_nm = ssht.elm2ind(ell, -m)
                conj_pixel_val = self.calc_pixel_value(ind_pm)
                flm[ind_pm] = conj_pixel_val
                flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])
        return flm

    def translate_dd_parallel(self, flm):
        '''
        parallel method to translate dirac delta

        Arguments:
            flm {array} -- translated function up to L_scipy_max

        Returns:
            array -- translated dirac delta
        '''
        # avoid strided arrays
        real = flm.real.copy()
        imag = flm.imag.copy()

        # create arrays to store final and intermediate steps
        result_r = np.ctypeslib.as_ctypes(real)
        result_i = np.ctypeslib.as_ctypes(imag)
        shared_array_r = sct.RawArray(result_r._type_, result_r)
        shared_array_i = sct.RawArray(result_i._type_, result_i)

        # ensure function declared before multiprocessing pool
        global func

        def func(chunk):
            '''
            perform translation for real function using
            the conjugate symmetry for real signals
            Arguments:
                ell {int} -- multipole
            '''

            # store real and imag parts separately
            tmp_r = np.ctypeslib.as_array(shared_array_r)
            tmp_i = np.ctypeslib.as_array(shared_array_i)

            # deal with chunk
            for ell in chunk:
                # m = 0 components
                ind = ssht.elm2ind(ell, m=0)
                conj_pixel_val = self.calc_pixel_value(ind)
                tmp_r[ind] = conj_pixel_val.real
                tmp_i[ind] = conj_pixel_val.imag

                # odd/even numbers
                for m in range(1, ell + 1):
                    print(ell, m)
                    ind_pm = ssht.elm2ind(ell, m)
                    ind_nm = ssht.elm2ind(ell, -m)
                    conj_pixel_val = self.calc_pixel_value(ind_pm)
                    # conjugate symmetry for real signals
                    tmp_r[ind_pm] = conj_pixel_val.real
                    tmp_i[ind_pm] = conj_pixel_val.imag
                    tmp_r[ind_nm] = (-1) ** m * tmp_r[ind_pm]
                    tmp_i[ind_nm] = (-1) ** (m + 1) * tmp_i[ind_pm]

        # split up L range to maximise effiency
        arr = np.arange(self.L_scipy_max, self.L)
        size = len(arr)
        arr[size // 2:size] = arr[size // 2:size][::-1]
        chunks = [np.sort(arr[i::self.ncpu]) for i in range(self.ncpu)]

        # initialise pool and apply function
        with mp.Pool(processes=self.ncpu) as p:
            p.map(func, chunks)

        # retrieve real and imag components
        result_r = np.ctypeslib.as_array(shared_array_r)
        result_i = np.ctypeslib.as_array(shared_array_i)

        # combine results
        flm_trans = result_r + 1j * result_i

        return flm_trans

    # ----------------------------------------
    # ---------- plotting functions ----------
    # ----------------------------------------

    def plotly_plot(self, f, filename, save_figure):
        '''
        creates basic plotly plot rather than matplotlib

        Arguments:
            f {function} -- inverse flm
            filename {str} -- filename for html/png/pdf plot
            save_figure {bool} -- flag to save figure
        '''

        # get values from the setup
        x, y, z, f_plot, vmin, vmax = self.setup_plot(f)

        zoom = 1.15
        camera = dict(
            eye=dict(x=-1 / zoom, y=-0.1 / zoom, z=1 / zoom)
        )

        data = [
            Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=f_plot,
                colorscale=self.matplotlib_to_plotly('magma'),
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(
                    x=0.92,
                    len=0.98,
                    nticks=5,
                    tickfont=dict(size=32)
                )
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
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        fig = Figure(data=data, layout=layout)

        # if save_fig is true then print as png and pdf in their directories
        if save_figure:
            png_filename = os.path.join(
                self.location, 'figures', 'png', filename + '.png')
            pio.write_image(fig, png_filename)
            pdf_filename = os.path.join(
                self.location, 'figures', 'pdf', filename + '.pdf')
            pio.write_image(fig, pdf_filename)

        # create html and open if auto_open is true
        html_filename = os.path.join(
            self.location, 'figures', 'html', filename + '.html')
        py.plot(fig, filename=html_filename, auto_open=self.auto_open)

    def plot(self, alpha_pi_fraction=0.75, beta_pi_fraction=0.25, gamma_pi_fraction=0):
        '''
        master plotting method

        Keyword Arguments:
            alpha_pi_fraction {float} -- fraction of pi (default: {0.75})
            beta_pi_fraction {float} -- fraction of pi (default: {0.25})
            gamma_pi_fraction {float} -- fraction of pi (default: {0.0})
        '''

        # setup
        gamma = gamma_pi_fraction * np.pi
        filename = self.flm_name + '_'
        filename += 'L-' + str(self.L) + '_'

        # calculate nearest index of alpha/beta for translation
        alpha, beta = self.calc_nearest_grid_point(
            alpha_pi_fraction, beta_pi_fraction)

        # test for plotting routine
        if self.routine == 'north':
            flm = self.flm
        elif self.routine == 'rotate':
            # adjust filename
            filename += self.routine + '_'
            filename += self.filename_angle(
                alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)
            # rotate by alpha, beta
            flm = self.rotation(self.flm, alpha, beta, gamma)
        elif self.routine == 'translate':
            # adjust filename
            filename += self.routine + '_'
            # don't add gamma if translation
            filename += self.filename_angle(
                alpha_pi_fraction, beta_pi_fraction)
            # translate by alpha, beta
            flm = self.translation(self.flm)

        if self.glm is not None:
            # perform convolution
            flm = self.convolution(flm, self.glm)
            # adjust filename
            filename += 'convolved_' + self.glm_name + '_'
            filename += 'L-' + str(self.L) + '_'

        # boost resolution
        if self.resolution != self.L:
            flm = self.resolution_boost(flm)

        # add sampling/resolution to filename
        filename += 'samp-' + str(self.method) + '_'
        filename += 'res-' + str(self.resolution) + '_'

        # inverse & plot
        f = ssht.inverse(flm, self.resolution,
                         Method=self.method, Reality=self.reality)

        # check for plotting type
        if self.type == 'real':
            plot = f.real
        elif self.type == 'imag':
            plot = f.imag
        elif self.type == 'abs':
            plot = abs(f)
        elif self.type == 'sum':
            plot = f.real + f.imag

        # do plot
        filename += self.type
        self.plotly_plot(plot, filename, self.save_fig)

    # --------------------------------------------------
    # ---------- translation helper functions ----------
    # --------------------------------------------------

    def calc_pixel_value(self, ind):
        '''
        calculate the ylm(omega') which defines the translation

        Arguments:
            ind {int} -- index in array

        Returns:
            complex float -- the value of ylm(omega')
        '''

        # create Ylm corresponding to index
        ylm_harmonic = np.zeros(self.L * self.L, dtype=complex)
        ylm_harmonic[ind] = 1

        # convert Ylm from pixel to harmonic space
        ylm_pixel = ssht.inverse(ylm_harmonic, self.L, Method=self.method)

        # get value at pixel (i, j)
        ylm_omega = np.conj(ylm_pixel[self.pix_i, self.pix_j])

        return ylm_omega

    def calc_nearest_grid_point(self, alpha_pi_fraction, beta_pi_fraction):
        '''
        calculate nearest index of alpha/beta for translation
        this is due to calculating \omega' through the pixel
        values - the translation needs to be at the same position
        as the rotation such that the difference error is small

        Arguments:
            alpha_pi_fraction {float} -- fraction of pi (phi)
            beta_pi_fraction {float} -- fraction of pi (theta)

        Returns:
            (float, float) -- value nearest given fraction
        '''

        thetas, phis = ssht.sample_positions(self.L, Method=self.method)
        self.pix_j = (np.abs(phis - alpha_pi_fraction * np.pi)).argmin()
        self.pix_i = (np.abs(thetas - beta_pi_fraction * np.pi)).argmin()
        alpha = phis[self.pix_j]
        beta = thetas[self.pix_i]

        # to be used outside of class i.e. tests
        self.alpha_pi_fraction = alpha_pi_fraction
        self.beta_pi_fraction = beta_pi_fraction

        return alpha, beta

    # -----------------------------------------------
    # ---------- plotting helper functions ----------
    # -----------------------------------------------

    @staticmethod
    def pi_in_filename(numerator, denominator):
        '''
        create filename for angle as multiple of pi

        Arguments:
            numerator {int} -- angle numerator
            denominator {int} -- angle denominator

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

    @staticmethod
    def get_angle_num_dem(angle_fraction):
        '''
        ger numerator and denominator for a given decimal

        Arguments:
            angle_fraction {float} -- fraction of pi

        Returns:
            (int, int) -- (fraction numerator, fraction denominator)
        '''

        angle = Fraction(angle_fraction).limit_denominator()
        return angle.numerator, angle.denominator

    @staticmethod
    def calc_resolution(config):
        '''
        calculate appropriate resolution for given L

        Arguments:
            config {dict} -- config dictionary

        Returns:
            int -- resolution
        '''

        if 'pow2_res2L' in config:
            exponent = config['pow2_res2L']
        else:
            if config['L'] == 1:
                exponent = 6
            elif config['L'] < 4:
                exponent = 5
            elif config['L'] < 8:
                exponent = 4
            elif config['L'] < 128:
                exponent = 3
            elif config['L'] < 512:
                exponent = 2
            elif config['L'] < 1024:
                exponent = 1
            else:
                exponent = 0
        resolution = config['L'] * 2 ** exponent

        return resolution

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

        h = 1 / (pl_entries - 1)
        pl_colorscale = []

        for k in range(pl_entries):
            C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
            pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

        return pl_colorscale

    def resolution_boost(self, flm):
        '''
        calculates a boost in resoltion for given flm

        Arguments:
            flm {array} -- flm to be boosted in res

        Returns:
            array -- boosted resolution flm
        '''

        boost = self.resolution * self.resolution - self.L * self.L
        flm_boost = np.pad(flm, (0, boost), 'constant')
        return flm_boost

    def filename_angle(self, alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction=0):
        '''
        middle part of filename

        Arguments:
            alpha_pi_fraction {float} -- fraction of pi
            beta_pi_fraction {float} -- fraction of pi
            gamma_pi_fraction {float} -- fraction of pi

        Returns:
            str -- filename
        '''

        # get numerator/denominator for filename
        alpha_num, alpha_den = self.get_angle_num_dem(alpha_pi_fraction)
        beta_num, beta_den = self.get_angle_num_dem(beta_pi_fraction)
        gamma_num, gamma_den = self.get_angle_num_dem(gamma_pi_fraction)

        # if alpha = beta = 0
        if not alpha_num and not beta_num:
            filename = 'alpha-0_beta-0_'
        # if alpha = 0
        elif not alpha_num:
            filename = 'alpha-0_beta-'
            filename += self.pi_in_filename(beta_num, beta_den)
            filename += '_'
        # if beta = 0
        elif not beta_num:
            filename = 'alpha-'
            filename += self.pi_in_filename(alpha_num, alpha_den)
            filename += '_beta-0_'
        # if alpha != 0 && beta !=0
        else:
            filename = 'alpha-'
            filename += self.pi_in_filename(alpha_num, alpha_den)
            filename += '_beta-'
            filename += self.pi_in_filename(beta_num, beta_den)
            filename += '_'

        # if rotation with gamma != 0
        if gamma_num:
            filename += 'gamma-'
            filename += self.pi_in_filename(gamma_num, gamma_den)
            filename += '_'
        return filename

    def setup_plot(self, f, close=True, parametric=False,
                   parametric_scaling=[0.0, 0.5], color_range=None):
        '''
        function which creates the data for the matplotlib/plotly plot

        Arguments:
            f {function} -- inverse of flm

        Keyword Arguments:
            close {bool} -- if true the full sphere is plotted without a gap (default: {True})
            parametric {bool} -- the radius of the object at a certain point is defined by the function (default: {False})
            parametric_scaling {list} -- used if Parametric=True, defines the radius of the shape at a particular angle (default: {[0.0, 0.5]})
            color_range {list} -- if set saturates the color bar in that range, else the function min and max is used (default: {None})

        Raises:
            Exception -- if band limit L is not the same size as function f

        Returns:
            tuple -- values for the plotting
        '''

        if self.method == 'MW_pole':
            if len(f) == 2:
                f, f_sp = f
            else:
                f, f_sp, phi_sp = f

        (thetas, phis) = ssht.sample_positions(
            self.resolution, Method=self.method, Grid=True)

        if (thetas.size != f.size):
            raise Exception('Band-limit L deos not match that of f')

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
                self.resolution, Method=self.method)
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
