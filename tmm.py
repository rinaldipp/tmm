import collections
import numpy as np
import numpy.matlib
import os
import mpmath as mp
from matplotlib import ticker, gridspec, style, rcParams
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import style
import matplotlib as mpl
import pandas
import xlsxwriter
from scipy.special import jv
import time
import scipy.interpolate
from scipy import integrate
from scipy.optimize import minimize
import pytta

outputs = os.getcwd()
style.use('seaborn-colorblind')


class TMM:
    """"
    Transfer Matrix Model for multi-layered acoustic treatments.
    """

    def __init__(self, fmin=20, fmax=5000, df=0.5, incidence='diffuse', incidence_angle=[0, 78, 1],
                 project_folder=None):

        self.fmin = fmin
        self.fmax = fmax
        self.df = df
        self.freq = np.linspace(self.fmin, self.fmax, int((self.fmax - self.fmin) / self.df) + 1)
        self.z = np.zeros_like(self.freq, dtype='complex')
        self.z_normal = None  # Only used for plotting when incidence == 'diffuse'
        self.s0 = 1  # Device front termination area
        self.srad = 1  # Device rear termination area
        self.matrix = {}
        self.air_prop = self.air_properties()
        self.project_folder = project_folder
        self.incidence = incidence
        if self.incidence == 'diffuse':
            self.incidence_angle = np.linspace(incidence_angle[0] + 0.01, incidence_angle[1] - 0.01,
                                               int((incidence_angle[1] - incidence_angle[0]) / incidence_angle[2]))
        elif self.incidence == 'normal':
            self.incidence_angle = np.linspace(0, 1, 1)
        elif self.incidence == 'angle':
            self.incidence_angle = np.linspace(incidence_angle[0], incidence_angle[0] + 1, 1)

    @property
    def rho0(self):
        return self.air_prop['air_density']

    @property
    def c0(self):
        return self.air_prop['speed_of_sound']

    @property
    def z0(self):
        return self.rho0 * self.c0

    @property
    def z_norm(self):
        return self.z / self.z0

    @property
    def w0(self):
        return 2 * np.pi * self.freq

    @property
    def k0(self):
        return self.w0 / self.c0

    @property
    def alpha(self):
        R, alpha = self.reflection_and_absorption_coefficient(self.z)
        return alpha.reshape((len(alpha),))

    @property
    def alpha_normal(self):
        R, alpha = self.reflection_and_absorption_coefficient(self.z_normal)
        return alpha

    @property
    def y(self):
        return 1 / self.z

    @property
    def y_norm(self):
        return 1 / self.z_norm

    def plot(self, figsize=(15, 5), plots=['alpha', 'z', 'y'], saveFig=False, filename='TMM', timestamp=True,
             ext='.png', max_mode='all'):
        """
        Displays device information.
        """

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, len(plots))

        i = 0
        if 'z' in plots or 'Z' in plots:
            ax_z = plt.subplot(gs[0, i])
            ax_z.set_title(r'Impedance ($Z$)')
            ax_z.set_xlabel('Frequency [Hz]')
            ax_z.set_ylabel('Normalized Surface Impedance [Z/Z0]')
            ax_z.semilogx(self.freq, np.real(self.z_norm), linewidth=2, label='Real')
            ax_z.semilogx(self.freq, np.imag(self.z_norm), linewidth=2, label='Imag')
            ax_z.set_xlim([(np.min(self.freq)), (np.max(self.freq))])
            ax_z.axhline(y=0, color='k', linewidth=0.5)
            ax_z.axhline(y=1, linestyle='--', color='gray')
            ax_z.legend(loc='best')
            ax_z.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_z.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_z.tick_params(which='minor', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_z.tick_params(which='major', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_z.minorticks_on()  # Set major and minor ticks to same length
            ax_z.grid('minor')
            i += 1

        if 'y' in plots or 'Y' in plots:
            ax_y = plt.subplot(gs[0, i])
            ax_y.set_title(r'Admittance ($Y$)')
            ax_y.set_xlabel('Frequency [Hz]')
            ax_y.set_ylabel('Normalized Surface Admittance [Z0/Z]')
            ax_y.semilogx(self.freq, np.real(self.y_norm), linewidth=2, label='Real')
            ax_y.semilogx(self.freq, np.imag(self.y_norm), linewidth=2, label='Imag')
            ax_y.set_xlim([(np.min(self.freq)), (np.max(self.freq))])
            ax_y.axhline(y=0, color='k', linewidth=0.5)
            ax_y.axhline(y=1, linestyle='--', color='gray')
            ax_y.legend(loc='best')
            ax_y.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_y.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_y.tick_params(which='minor', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_y.tick_params(which='major', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_y.minorticks_on()  # Set major and minor ticks to same length
            ax_y.grid('minor')
            i += 1

        if 'alpha' in plots or 'abs' in plots:
            ax_a = plt.subplot(gs[0, i])
            ax_a.set_title(r'Absorption Coefficient ($\alpha$)')
            ax_a.set_xlabel('Frequency [Hz]')
            ax_a.set_ylabel('Absorption Coefficient [-]')
            if self.incidence == 'diffuse':
                ax_a.semilogx(self.freq, self.alpha, linewidth=2,
                              label=f'Diffuse Incidence ' +
                                    f'({min(self.incidence_angle):0.0f}° - {max(self.incidence_angle):0.0f}°)')
                ax_a.semilogx(self.freq, self.alpha_normal, linewidth=2, label='Normal Incidence', linestyle=':')
            else:
                ax_a.semilogx(self.freq, self.alpha, linewidth=2)
            if max_mode == 'all':
                abs_value, idx = find_nearest(self.alpha, max(self.alpha))
            elif max_mode is not None:
                max_mode_val, idx_max_mode = find_nearest(self.freq, max_mode)
                abs_value, idx = find_nearest(self.alpha[0:idx_max_mode], max(self.alpha[0:idx_max_mode]))
            if max_mode is not None:
                ax_a.axvline(x=self.freq[idx], label=f'Resonance at {self.freq[idx]} Hz', linestyle='--', color='green')
            ax_a.set_xlim([(np.min(self.freq)), (np.max(self.freq))])
            ax_a.set_ylim([-0.1, 1.1])
            ax_a.legend(loc='best')
            ax_a.axhline(y=0, color='k', linewidth=0.5)
            ax_a.axhline(y=1, linestyle='--', color='gray')
            ax_a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_a.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_a.tick_params(which='minor', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_a.tick_params(which='major', length=5, rotation=-90,
                             axis='x')  # Set major and minor ticks to same length
            ax_a.minorticks_on()  # Set major and minor ticks to same length
            ax_a.grid('minor')
            i += 1

        gs.tight_layout(fig, pad=4, w_pad=1, h_pad=1)

        if saveFig:
            timestr = time.strftime("%Y%m%d-%H%M_")
            if self.project_folder is None:
                full_path = outputs + '\\' + filename + ext
                if timestamp is True:
                    full_path = outputs + '\\' + timestr + filename + ext
            else:
                folderCheck = os.path.exists(self.project_folder + '\\Treatments')
                if folderCheck is False:
                    os.mkdir(self.project_folder + '\\Treatments')
                full_path = self.project_folder + '\\Treatments\\' + filename + ext
                if timestamp is True:
                    full_path = self.project_folder + '\\Treatments\\' + timestr + filename + ext

            plt.savefig(full_path, dpi=100)
            print('Image saved to ', full_path)
        plt.show()

    def delany_bazley(self, sigma, warnings=1):
        """
        Calculates the wavenumber (k) and characteristic impedance (zc) using the Delaney and Bazley formulations.
        Acousic Absorbers and Diffusers by Trevor Cox and Peter D'Antonio
        Eqs 5.7-5.9, 3rd Ed

        Inputs:
         - sigma: flow resistivity [rayls/m]
        - warnings: display (1) or don't display (0) warnings about limitations of empirical formulation
        """

        X = self.rho0 * self.freq / sigma # Dimensionless quantity for Delany and Bazley
        c = [0.0571, -0.754, -0.087, -0.732, 0.0978, -0.700, -0.189, -0.595]  # db
        zc = self.z0 * (1 + c[0] * X ** c[1] + 1j * c[2] * X ** c[3])  # Characteristic impedance
        kc = self.k0 * (1 + c[4] * X ** c[5] + 1j * c[6] * X ** c[7])  # Wavenumber

        # Warnings
        if warnings == 1:
            if min(X) < 0.01:
                print(f'X = {min(X)} too small')
            if max(X) > 1:
                print(f'X = {max(X)} too large')
            if sigma < 1000:
                print(f'Flow resistivity = {sigma} too small')
            if sigma > 50000:
                print(f'Flow resistivity = {sigma} too large')

        return kc, zc

    def miki(self, sigma):
        """
        Calculates the wavenumber (k) and characteristic impedance (zc) using Miki's model.
        Guideline for Adopting the Local Reaction Assumption for Porous Absorbers
        in Terms of Random Incidence Absorption Coefficients by Cheol-Ho Jeong
        Eqs. 5a and 5b

        Inputs:
         - sigma: flow resistivity [rayls/m]
        """

        zc = self.z0 * (1 + 0.070 * (self.freq / sigma) ** -0.632 - 1j * 0.107 * (self.freq / sigma) ** -0.632)
        kc = self.k0 * (1 + 0.109 * (self.freq / sigma) ** -0.618 - 1j * 0.160 * (self.freq / sigma) ** -0.618)

        return kc, zc

    def mechel_grundmann(self, sigma, fibre_type, warnings=1):
        '''
        Calculates the wavenumber (kc) and characteristic impedance (zc) using the Mechel and Grundmann formulations
        Acoustic Absorbers and Diffusers by Trevor Cox and Peter D'Antonio
        Eqs. 5.10, 3rd edition

        Inputs:
         - sigma: flow resistivity [rayls/m]
         - fibre_type: int, chosses bewtween available fibre types
                       1 = basalt or rock wool; 2 = glass fibre
        - warnings: display (1) or don't display (0) warnings about limitations of empirical formulation
        '''

        if fibre_type == 1:
            betak = [-0.00355757 - 1j * 0.0000164897, 0.421329 + 1j * 0.342011, -0.507733 + 1j * 0.086655,
                     -0.142339 + 1j * 1.25986, 1.29048 - 1j * 0.0820811, -0.771857 - 1j * 0.668050]

            betaz = [0.0026786 + 1j * 0.00385761, 0.135298 - 1j * 0.394160, 0.946702 + 1j * 1.47653,
                     -1.45202 - 1j * 4.56233, 4.03171 + 1j * 7.56031, -2.86993 - 1j * 4.90437]

        elif fibre_type == 2:
            betak = [-0.00451836 + 1j * 0.000541333, 0.421987 + 1j * 0.376270, -0.383809 - 1j * 0.353780,
                     -0.610867 + 1j * 2.59922, 1.13341 - 1j * 1.74819, 0]

            betaz = [-0.00171387 + 1j * 0.00119489, 0.283876 - 1j * 0.292168, -0.463860 + 1j * 0.188081,
                     3.12736 + 1j * 0.941600, -2.10920 - 1j * 1.32398, 0]
        else:
            print('Choose fibre type  between 1 (basalt or rock wool) or  2 (glass fibre).')

        # Calculate impedance of porous material
        X = self.rho0 * self.freq / sigma  # Dimensionless quantity

        # Mechel Grundmann
        kc = -1j * self.k0 * (betak[0] * X ** -1 + betak[1] * X ** -0.5 + betak[2] + betak[3] * X ** 0.5 +
                              betak[4] * X + betak[5] * X ** 1.5)
        zc = self.z0 * (betaz[0] * X ** -1 + betaz[1] * X ** -0.5 + betaz[2] + betaz[3] * X ** 0.5 +
                        betaz[4] * X + betaz[5] * X ** 1.5)

        # Warnings
        if warnings == 1:
            if min(X) < 0.003:
                print(f'X = {min(X)} too small')
            if max(X) > 0.4:
                print(f'X = {max(X)} too large')

        return kc, zc

    def wilson(self, sigma):
        '''
        Calculates the wavenumber (kc) and characteristic impedance (zc) using the Wilson's equivalent to
        Delaney and Bazley model from relaxation model.
        Acoustic Absorbers and Diffusers by Trevor Cox and Peter D'Antonio
        Eqs. 5.22-5.33 using simplifications detailed below, 3rd edition

        Inputs:
         - sigma: flow resistivity [rayls/m]
        '''

        X = self.rho0 * self.freq / sigma  # Dimensionless quantity
        omega = 1
        gamma = 1.4  # Ratio of specific heats
        q = 1
        zc = self.z0 * (q / omega) / np.sqrt(
            (1 + (gamma - 1) / np.sqrt(1 + 1j * 19 * X)) * (1 - 1 / np.sqrt(1 + 1j * 13 * X)))
        kc = (q * self.k0) * np.sqrt((1 + (gamma - 1) / np.sqrt(1 + 1j * 19 * X)) / (1 - 1 / np.sqrt(1 + 1j * 13 * X)))

        return kc, zc

    def allard_champoux(self, sigma, warnings=1):
        """
        Calculates the wavenumber (k) and characteristic impedance (zc) using Modiefied Allard & Champoux model.
        Sound absorption of porous materials – Accuracy of prediction methods
        by David Oliva & Valtteri Hongisto (2013)
        New empirical equations for sound propagation in rigid frame fibrous materials
        by Jean-F. Allard & Yvan Champoux (1992)
        Eqs. 5,6,7,8

        Inputs:
         - sigma: flow resistivity [rayls/m]
        """

        X = self.rho0 * self.freq / sigma
        c = [0.0982, 0.685, 0.288, 0.526, 0.0729, 0.66228, 0.187, 0.5379]  # allard_champoux_modified
        cc = self.c0 / (1 + c[0] * X ** c[1] - 1j * c[2] * X ** -c[3])
        pc = self.z0 / cc * (1 + c[4] * X ** -c[5] - 1j * c[6] * X ** -c[7])

        kc = 2 * np.pi * self.freq / cc
        zc = cc * pc

        # Warnings
        if warnings == 1:
            if min(self.freq) < 45:
                print(f'X = {min(self.freq)} too small')
            if max(self.freq) > 11e3:
                print(f'X = {max(self.freq)} too large')

        return kc, zc

    def reflection_and_absorption_coefficient(self, zs):
        """
        Calculate reflection coefficient (R) and absorption coefficient (alpha)
        for a given surface impedance zs.
        Acoustic Absorbers and Diffusers by Trevor Cox and Peter D'Antonio
        Eqs. from Chapter 1
        """

        R = (zs - self.z0) / (zs + self.z0)  # Reflection coefficient
        alpha = 1 - np.abs(R) ** 2  # Absorption coefficient

        return R, alpha

    def porous_layer(self, sigma=27, t=5, model='ac', layer=None,
                     model_params={'warnings': 0, 'fibre_type': 2}):
        """
        Adds a layer of porous material to the existing device.

        Inputs:
         - layer: int or None, determines the layer number; If None is passed the layer
                  will be determined from the existing layers
         - t: layer thickness [mm]
         - sigma: flow resistivity [k*Pa*s/m²]
         - model: str, chooses between the available equivalent homogenous fluid models
         - model_params: dict containing extra parameters for each model (if needed)
         - incidence_angle: not implemented yet
        """

        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        sigma_k = sigma * 1000  # Convert to kilo rayls/m

        if model == 'miki':
            kc, zc = self.miki(sigma_k)
        elif model == 'db':
            kc, zc = self.delany_bazley(sigma_k, warnings=model_params['warnings'])
        elif model == 'ac' or model == 'mac':
            kc, zc = self.allard_champoux(sigma_k, warnings=model_params['warnings'])
        elif model == 'wilson':
            kc, zc = self.wilson(sigma_k)
        elif model == 'mg' or model == 'mechel':
            kc, zc = self.mechel_grundmann(sigma_k, fibre_type=model_params['fibre_type'],
                                           warnings=model_params['warnings'])

        kc = numpy.matlib.repmat(kc, len(self.incidence_angle), 1).T
        zc = numpy.matlib.repmat(zc, len(self.incidence_angle), 1).T

        Tp = np.array([[np.cos(kc * t_meters), 1j * zc / self.s0 * np.sin(kc * t_meters)],
                       [1j * self.s0 / zc * np.sin(kc * t_meters), np.cos(kc * t_meters)]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {'type': 'porous_layer',
                              'flow_resistivity [k*Pa*s/m²]': sigma,
                              'thickness [mm]': t,
                              'model': model,
                              'matrix': Tp,
                              }

    def air_layer(self, layer=None, t=5):
        """
        Adds an air layer to the existing device.

        Inputs:
         - layer: int or None, determines the layer number; If None is passed the layer
                  will be determined from the existing layers
         - t: layer thickness [mm]
        """

        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters

        k0 = numpy.matlib.repmat(self.k0, len(self.incidence_angle), 1).T
        k0 = np.sqrt(k0 ** 2 - k0 ** 2 * np.sin(np.deg2rad(self.incidence_angle)))
        z0 = numpy.matlib.repmat(self.z0, len(self.incidence_angle), 1).T

        Ta = np.array([[np.cos(k0 * t_meters), 1j * z0 / self.s0 * np.sin(k0 * t_meters)],
                       [1j * self.s0 / z0 * np.sin(k0 * t_meters), np.cos(k0 * t_meters)]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {'type': 'air_layer',
                              'thickness [mm]': t,
                              'matrix': Ta,
                              }

    def membrane_layer(self, layer=None, t=1, rho=8050):
        """
        Adds a stiff membrane to the existing device.

        Acústica de Salas - projeto e modelagem
        by Eric Brandão (2016)

        Inputs:
         - layer: int or None, determines the layer number; If None is passed the layer
                  will be determined from the existing layers
         - t: layer thickness [mm]
         - rho: material density [kg/m³]

        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters

        k0_angle = numpy.matlib.repmat(self.k0, len(self.incidence_angle), 1).T
        w0_angle = numpy.matlib.repmat(self.w0, len(self.incidence_angle), 1).T
        kt = k0_angle * np.sin(np.deg2rad(self.incidence_angle))
        zc_elastic = 1j * w0_angle * rho * (1 - (t * kt ** 4) / (w0_angle ** 2 * rho))

        zc = 1j * self.w0 * rho * t_meters / self.s0
        zc = numpy.matlib.repmat(zc, len(self.incidence_angle), 1).T

        ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

        Tm = np.array([[ones, zc],
                       [zeros, ones]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {'type': 'membrane_layer',
                              'thickness [mm]': t,
                              'density [kg/m³]': rho,
                              'matrix': Tm,
                              }

    def perforated_panel_layer(self, layer=None, t=19, d=8, s=16, end_correction='nesterov', method='barrier_mpp'):
        """
        Adds a plate with circular perforations to the existing device.

        Inputs:
         - layer: int or None, determines the layer number; If None is passed the layer
                  will be determined from the existing layers
         - t: layer thickness [mm]
         - d: hole diameter [mm]
         - s: hole spacing from the center of one hole to the next [mm]
         - end_correction: str, chooses between the available end corrections for the tube length
        """

        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        d_meters = d / 1000
        s_meters = s / 1000

        if d < 2 / s:
            print(f'WARNING: Hole spacing too small for {d} [mm] hole diameter.')

        open_area = np.pi / (s_meters / (d_meters / 2)) ** 2

        if end_correction == 'nesterov':
            # Acoustic Absorbers and Diffusers by Trevor Cox and Peter D'Antonio
            delta = 0.8 * (1 - 1.47 * open_area ** 0.5 + 0.47 * open_area ** 1.5)
            t_corr = 2 * delta * d_meters / 2 + t_meters
        elif end_correction == 'beranek':
            # Leo Beranek - Acoustics
            t_corr = t_meters + 0.85 * d_meters

        if method == 'barrier':
            # Impedance from Acoustic Absorbers and Diffusers by Trevor Cox and Peter D'Antonio
            vis = self.air_prop['air_viscosity']
            rm = (self.rho0 / open_area) * np.sqrt(8 * vis * self.w0) * (
                        1 + t_meters / (d_meters))  # Surface resistance
            zpp = (1j / open_area) * t_corr * self.w0 * self.rho0 + rm  # Impedance of perforated plate
            zpp = numpy.matlib.repmat(zpp, len(self.incidence_angle), 1).T

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Tp = np.array([[ones, zpp],
                           [zeros, ones]])

        if method == 'barrier_mpp':
            # Impedance for micro perforated plate by MAA
            vis = self.air_prop['air_viscosity']
            cis = d_meters * np.sqrt(self.w0 * self.rho0 / (4 * vis))
            rm = (32 * vis / open_area) * (t_meters / d_meters ** 2) * (np.sqrt(1 + cis ** 2 / 32) +
                                                                        (np.sqrt(2) / 32) * cis * d_meters / t_meters)
            m = (self.rho0 * t_meters / open_area) * (1 + 1 / (np.sqrt(9 + cis ** 2 / 2)) + 0.85 * d_meters / t_meters)
            zpp = rm + 1j * self.w0 * m
            zpp = numpy.matlib.repmat(zpp, len(self.incidence_angle), 1).T

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Tp = np.array([[ones, zpp],
                           [zeros, ones]])

        elif method == 'eq_fluid':
            # Impedance calculated through Zwikker and Kosten's model
            kc, zc = self.viscothermal_circular(d_meters, open_area)

            kc = numpy.matlib.repmat(kc, len(self.incidence_angle), 1).T
            zc = numpy.matlib.repmat(zc, len(self.incidence_angle), 1).T

            Tp = np.array([[np.cos(kc * t_corr), 1j * zc / self.s0 * np.sin(kc * t_corr)],
                           [1j * self.s0 / zc * np.sin(kc * t_corr), np.cos(kc * t_corr)]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {'type': 'perforated_panel_layer',
                              'thickness [mm]': t,
                              'hole_diameter [mm]': d,
                              'hole_spacing [mm]': s,
                              'open_area [%]': open_area * 100,
                              'end_correction': end_correction,
                              'matrix': Tp,
                              }

    def slotted_panel_layer(self, layer=None, t=19, w=8, s=16, method='barrier'):
        """
        Adds a plate with rectangular slits to the existing device.

        Inputs:
         - layer: int or None, determines the layer number; If None is passed the layer
                  will be determined from the existing layers
         - t: layer thickness [mm]
         - w: slit width [mm]
         - s: slit spacing from the center of one slit to the next [mm]
        """

        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        w_meters = w / 1000
        s_meters = s / 1000

        open_area = w_meters / s_meters

        t_corr = t_meters + 2 * w_meters * (-1 / np.pi) * np.log(np.sin(0.5 * np.pi * open_area))

        if method == 'barrier':
            # Impedance from "On the design of resonant absorbers using a slotted plate" by Kristiansen and Vigran
            vis = self.air_prop['air_viscosity']
            Rp = 0.5 * np.sqrt(2 * vis * self.rho0 * self.w0) * (4 + (2 * t) / w)
            Xp = self.w0 * self.rho0 * (t_corr)
            zs = (Rp + 1j * Xp) / open_area
            zs = numpy.matlib.repmat(zs, len(self.incidence_angle), 1).T

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Ts = np.array([[ones, zs],
                           [zeros, ones]])

        elif method == 'eq_fluid':

            kc, zc = self.viscothermal_slit(w_meters, open_area)
            kc = numpy.matlib.repmat(kc, len(self.incidence_angle), 1).T
            zc = numpy.matlib.repmat(zc, len(self.incidence_angle), 1).T

            Ts = np.array([[np.cos(kc * t_corr), 1j * zc / self.s0 * np.sin(kc * t_corr)],
                           [1j * self.s0 / zc * np.sin(kc * t_corr), np.cos(kc * t_corr)]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {'type': 'slotted_panel_layer',
                              'thickness [mm]': t,
                              'slot_width [mm]': w,
                              'slot_spacing [mm]': s,
                              'open_area [%]': open_area * 100,
                              'matrix': Ts,
                              }

    def air_properties(self, t0=20, p0=101320, rh=30):
        """
        Computes properties of humid air.

        Input parameters:
          - t0: temperature in Celsius [C]
          - p0: atmospheric pressure in Pascal [Pa]
          - rh: relative humidity in percentage [%]

        Output parameters:
          - rho0: volume density [kg/m³]
          - c0: sound speed [m/s]
          - vis: absolute (or dynamic) viscosity [Ns/m²]
          - gam: specific heat ratio [-]
          - pn: Prandtl number [-]
          - Cp: Constant Pressure Specific Heat [J/kg*K]
        """

        kappla = 0.026  # Air thermal conductivity [W/m*k]
        t = t0 + 273.16  # Temperature in Kelvin
        R = 287.031  # Gas constant for air [J/K/kg]
        Rvp = 461.521  # Gas constant for water vapor [J/K/kg]
        Pvp = 0.0658 * t ** 3 - 53.7558 * t ** 2 + 14703.8127 * t - 1345485.0465  # Pierce(Acoustics, 1991) page 555
        vis = 7.72488e-8 * t - 5.95238e-11 * t ** 2 + 2.71368e-14 * t ** 3
        Cp = 4168.8 * (0.249679 - 7.55179e-5 * t + 1.69194e-7 * t ** 2 - 6.46128e-11 * t ** 3)
        Cv = Cp - R  # Constant Volume Specific Heat [J/kg/K] for 260 K < T < 600 K
        pn = vis * Cp / kappla  # Prandtl number (fewly varies at typical air conditions (0°C=0.715; 60°C=0.709)
        gam = Cp / Cv  # Specific heat ratio [-]
        rho0 = p0 / (R * t) - (1 / R - 1 / Rvp) * rh / 100 * Pvp / t  # Density of air [kg/m³]
        c0 = (gam * p0 / rho0) ** 0.5

        air_properties = {'temperature_in_celsius': t0,
                          'relative_humidity': rh,
                          'atmospheric_pressure': p0,
                          'prandtl_number': pn,
                          'specific_heat_ratio': gam,
                          'air_density': rho0,
                          'speed_of_sound': c0,
                          'air_viscosity': vis,
                          'air_thermal_conductivity': kappla,
                          'constant_pressure_specific_heat': Cp}

        # return rho0, c0, vis, gam, pn, Cp, kappla
        return air_properties

    def viscothermal_circular(self, d, open_area):
        """
        Zwikker and Kosten viscothermal model for circular cross-section.

        Input parameters:
         - d: hole diameter [m]
         - open_area: plate porosity (% open area) [-]
        """

        vis = self.air_prop['air_viscosity']
        #         vis = 15e-6
        gam = self.air_prop['specific_heat_ratio']
        pn = self.air_prop['prandtl_number']
        p0 = self.air_prop['atmospheric_pressure']

        beta = d / 2 * np.sqrt(self.w0 * self.rho0 / vis)
        rhoef = self.rho0 / open_area * 1 / (1 - 2 / (beta * np.sqrt(-1j)) * jv(1, beta * np.sqrt(-1j)) /
                                             jv(0, beta * np.sqrt(-1j)))
        kef = p0 / open_area * gam / (1 + (gam - 1) * 2 / (beta * np.sqrt(-1j * pn)) * jv(1, beta * np.sqrt(-1j * pn)) /
                                      jv(0, beta * np.sqrt(-1j * pn)))
        kc = self.w0 * np.sqrt(rhoef / kef)
        zc = np.sqrt(rhoef * kef)

        return kc, zc

    def viscothermal_slit(self, w, open_area):
        """
        M. Biot viscothermal model for circular slit cross-section.

        Input parameters:
         - w: slot width [m]
         - open_area: plate porosity (% open area) [-]
        """

        vis = self.air_prop['air_viscosity']
        gam = self.air_prop['specific_heat_ratio']
        pn = self.air_prop['prandtl_number']
        p0 = self.air_prop['atmospheric_pressure']

        beta = w / 2 * np.sqrt(self.w0 * self.rho0 / vis)

        rhoef = self.rho0 / open_area * 1 / (1 - np.tanh(beta * np.sqrt(1j)) / (beta * np.sqrt(1j)))
        kef = p0 / open_area * gam / (1 + (gam - 1) * np.tanh(beta * np.sqrt(1j * pn) / (beta * np.sqrt(1j * pn))))
        kc = self.w0 * np.sqrt(rhoef / kef)
        zc = np.sqrt(rhoef * kef)

        return kc, zc

    def show_layers(self, conversion=[0.0393701, '[inches]']):
        print('Device properties:')
        print('\t(1 - Front face)')
        print(f'\t({len(self.matrix)} - Rear Face)')
        print(f'\tSound incidence: {self.incidence}')
        if self.incidence == 'diffuse':
            print(f'\tAngle: {min(self.incidence_angle):0.0f}° - {max(self.incidence_angle):0.0f}°\n')
        else:
            print(f'\tAngle: {(self.incidence_angle[0]):0.0f}°\n')

        total_depth = 0
        for i in range(1, len(self.matrix) + 1):
            print(f"Layer {i}:")
            for key, value in self.matrix[i - 1].items():
                if key != 'matrix':
                    if isinstance(value, str) or isinstance(value, bool):
                        print(f'\t{key}: ', value)
                    else:
                        if '[mm]' in key:
                            converted = key.replace('[mm]', conversion[1])
                            print(f'\t{key}: {value:0.2f} | {converted}: {value * conversion[0]:0.2f}')
                        else:
                            print(f'\t{key}: {value:0.2f}')
                        if 'thickness' in key:
                            total_depth += value
        print(f'\nTotal treatment depth [mm]: {total_depth:0.2f} | ' +
              f'Total treatment depth {conversion[1]}: {total_depth * conversion[0]:0.2f}')

    def save2sheet(self, filename='TMM', timestamp=True, conversion=[0.0393701, '[inches]'],
                   ext='.xlsx', chart_styles=[35, 36], nthOct=3):

        timestr = time.strftime("%Y%m%d-%H%M_")
        if self.project_folder is None:
            full_path = outputs + '\\' + filename + ext
            if timestamp is True:
                full_path = outputs + '\\' + timestr + filename + ext
        else:
            folderCheck = os.path.exists(self.project_folder + '\\Treatments')
            if folderCheck is False:
                os.mkdir(self.project_folder + '\\Treatments')
            full_path = self.project_folder + '\\Treatments\\' + filename + ext
            if timestamp is True:
                full_path = self.project_folder + '\\Treatments\\' + timestr + filename + ext

        workbook = xlsxwriter.Workbook(full_path)
        worksheet = workbook.add_worksheet()

        # Setting formats
        bold = workbook.add_format({'bold': True, 'font_color': 'black', 'align': 'center', 'border': 2})
        regular = workbook.add_format({'bold': False, 'font_color': 'black', 'align': 'center', 'border': 1})
        regular_left_bold = workbook.add_format({'bold': True, 'font_color': 'black', 'align': 'right', 'border': 1,
                                                 })
        regular_left = workbook.add_format({'bold': False, 'font_color': 'black', 'align': 'left', 'border': 1,
                                            })

        # Adding frequency related data
        worksheet.write(0, 0, 'Frequency', bold)
        worksheet.write(0, 1, 'Real Z', bold)
        worksheet.write(0, 2, 'Img Z', bold)
        worksheet.write(0, 3, 'Absorption', bold)
        for i in range(len(self.freq)):
            worksheet.write(1 + i, 0, self.freq[i], regular)
            worksheet.write(1 + i, 1, np.real(self.z_norm[i]), regular)
            worksheet.write(1 + i, 2, np.imag(self.z_norm[i]), regular)
            worksheet.write(1 + i, 3, self.alpha[i], regular)

        # Absorption coefficient plot
        chart_abs = workbook.add_chart({'type': 'line'})
        chart_abs.add_series({'name': ['Sheet1', 0, 3],
                              'categories': ['Sheet1', 1, 0, len(self.freq) + 1, 0],
                              'values': ['Sheet1', 1, 3, len(self.freq) + 1, 3], })
        chart_abs.set_title({'name': 'Absorption Coefficient'})
        chart_abs.set_x_axis({'name': 'Frequency [Hz]'})
        chart_abs.set_y_axis({'name': 'Alpha [-]'})
        chart_abs.set_style(chart_styles[0])
        worksheet.insert_chart('G1', chart_abs, {'x_offset': 0, 'y_offset': 0, 'x_scale': 1.334, 'y_scale': 1.11})

        # Impedance plot
        chart_z = workbook.add_chart({'type': 'line'})
        chart_z.add_series({'name': ['Sheet1', 0, 1],
                            'categories': ['Sheet1', 1, 0, len(self.freq) + 1, 0],
                            'values': ['Sheet1', 1, 1, len(self.freq) + 1, 1], })
        chart_z.add_series({'name': ['Sheet1', 0, 2],
                            'categories': ['Sheet1', 1, 0, len(self.freq) + 1, 0],
                            'values': ['Sheet1', 1, 2, len(self.freq) + 1, 2], })
        chart_z.set_title({'name': 'Normalized Surface Impedance'})
        chart_z.set_x_axis({'name': 'Frequency [Hz]'})
        chart_z.set_y_axis({'name': 'Z [Pa*s/m]'})
        chart_z.set_style(chart_styles[1])
        worksheet.insert_chart('G17', chart_z, {'x_offset': 0, 'y_offset': 0, 'x_scale': 1.334, 'y_scale': 1.11})

        # Adding nthOct band absorption coeffiecients
        line = 0
        idx = 4
        worksheet.merge_range(line, idx, line, idx + 1, f'1/{nthOct} octave band absorption coefficients', bold)
        line += 1
        worksheet.write(line, idx, 'Frequency Band [Hz]', bold)
        worksheet.write(line, idx + 1, 'Absorption Coeffiecient [-]', bold)
        line += 1
        _, _, octValues = self.filter_alpha(nthOct=nthOct, plot=False, returnValues=True)
        lists = sorted(octValues.items())  # sorted by key, return a list of tuples
        xOct, yOct = zip(*lists)  # unpack a list of pairs into two tuples
        for x, y in zip(xOct, yOct):
            worksheet.write(line, idx, x, regular)
            worksheet.write(line, idx + 1, y, regular)
            line += 1

        # Adding device properties
        total_depth = 0
        worksheet.merge_range(line, idx, line, idx + 1, 'Device Properties', bold)
        line += 1
        worksheet.write(line, idx, '(1 - Front face)', regular)
        worksheet.write(line, idx + 1, f'({len(self.matrix)} - Rear face)', regular)
        line += 1
        worksheet.write(line, idx, 'Sound incidence:', regular_left_bold)
        worksheet.write(line, idx + 1, self.incidence, regular_left)
        line += 1
        worksheet.write(line, idx, 'Angle [°]:', regular_left_bold)
        if self.incidence == 'diffuse':
            worksheet.write(line, idx + 1,
                            f'{min(self.incidence_angle):0.0f} - {max(self.incidence_angle):0.0f}',
                            regular_left)
        else:
            worksheet.write(line, idx + 1,
                            f'{(self.incidence_angle[0]):0.0f}',
                            regular_left)
        line -= 1
        for i in range(1, len(self.matrix) + 1):
            if i > 1:
                line -= 1
            worksheet.merge_range(1 + i + line, idx, 1 + i + line, idx + 1, f'Layer {i}', bold)
            line += 1
            for key, value in self.matrix[i - 1].items():
                if key != 'matrix':
                    if isinstance(value, str) or isinstance(value, bool):
                        worksheet.write(1 + i + line, idx, f'{key}:', regular_left_bold)
                        worksheet.write(1 + i + line, idx + 1, f'{value}', regular_left)
                        line += 1
                    else:
                        if '[mm]' in key:
                            converted = key.replace('[mm]', conversion[1])
                            worksheet.write(1 + i + line, idx, f'{key}:', regular_left_bold)
                            worksheet.write(1 + i + line, idx + 1, value, regular_left)
                            line += 1
                            worksheet.write(1 + i + line, idx, f'{converted}:', regular_left_bold)
                            worksheet.write(1 + i + line, idx + 1, value * conversion[0], regular_left)
                            line += 1
                        else:
                            worksheet.write(1 + i + line, idx, f'{key}:', regular_left_bold)
                            worksheet.write(1 + i + line, idx + 1, value, regular_left)
                            line += 1
                        if 'thickness' in key:
                            total_depth += value

        worksheet.merge_range(1 + i + line, idx, 1 + i + line, idx + 1, 'Total', bold)
        line += 1
        worksheet.write(1 + i + line, idx, f'total treatment depth [mm]:', regular_left_bold)
        worksheet.write(1 + i + line, idx + 1, total_depth, regular_left)
        line += 1
        worksheet.write(1 + i + line, idx, f'total treatment depth {conversion[1]}:', regular_left_bold)
        worksheet.write(1 + i + line, idx + 1, total_depth * conversion[0], regular_left)
        line += 1

        # Setting column widths
        worksheet.set_column('A:D', 12)
        worksheet.set_column('E:F', 28)

        workbook.close()

        print(f'Sheet saved to ', full_path)

    def filter_alpha(self, nthOct=1, plot='available', warning=False, returnValues=False, show=False, figsize=(15, 5)):

        bands = pytta.utils.fractional_octave_frequencies(nthOct=nthOct)
        result = np.array([0], float)
        available_data = {}
        """

        Parameters
        ----------
        freq : array of int
            The frequency values.
        alpha : array of float
            The sound absorption coefficient you would like to filter.
        nthOct : int
            How many bands per octave. Default is 3 = 1/3 octave band.

        Returns
        -------
        array of float
            The center frequency for each band.
        array of float
            The sound absorption coefficient filtered.
        """
        # Compute the acoustic absorption coefficient per octave band
        for a in np.arange(1, len(bands)):
            result = np.append(result, 0)  # band[a] = 0
            idx = np.argwhere((self.freq >= bands[a, 0]) & (self.freq < bands[a, 2]))
            # If we have no 'alpha' point in this band
            if (len(idx) == 0):
                if warning:
                    print(f'Warning: no point found in band centered at {bands[a, 1]} Hz')
            # If we have only 1 'alpha' point in this band
            elif (len(idx) == 1):
                if warning:
                    print(f'Warning: only one point found in band centered at {bands[a, 1]} Hz')
                result[a] = self.alpha[idx]
            # If we have more than 1 'alpha' point in this band
            elif (len(idx) > 1):
                for b in np.arange(len(idx) - 1):
                    result[a] = result[a] + (self.freq[idx[0] + b] - self.freq[idx[0] + b - 1]) * abs(
                        self.alpha[idx[1] + b] + self.alpha[idx[0] + b - 1]) / 2
                result[a] = result[a] / (self.freq[idx[len(idx) - 1]] - self.freq[idx[0]])
                available_data[bands[a, 1]] = result[a]

        # Plot
        if plot:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax1.semilogx(self.freq, self.alpha, label='Narrowband')
            ax2 = ax1.twiny()
            ax2.set_xscale('log')
            if plot == 'available':
                lists = sorted(available_data.items())  # sorted by key, return a list of tuples
                x, y = zip(*lists)  # unpack a list of pairs into two tuples
                ax1.semilogx(x, y, 'o-', label=f'1/{nthOct} octave band')
                ax1.set_xlim([min(self.freq), max(self.freq)])
            elif plot == 'all':
                ax1.semilogx(bands[:, 1], result, 'o-', label=f'1/{nthOct} octave band')
                x = bands[:, 1].tolist()
            ax2.set_xticks([freq for freq in x])
            ax2.set_xticklabels([f'{freq:0.1f}' for freq in x])
            ax2.set_xlim(ax1.get_xlim())
            ax1.set_ylabel('Absorption Coefficient [-]')
            ax1.set_xlabel('Narrowband Frequency [Hz]')
            ax2.set_xlabel(f'1/{nthOct} Octave Bands [Hz]')
            ax1.set_ylim([-0.1, 1.1])
            ax1.legend(loc='best')
            ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax1.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax1.tick_params(which='minor', length=5, rotation=-90,
                            axis='x')  # Set major and minor ticks to same length
            ax1.tick_params(which='major', length=5, rotation=-90,
                            axis='x')  # Set major and minor ticks to same length
            ax2.tick_params(which='major', length=5, rotation=-90,
                            axis='x')  # Set major and minor ticks to same length
            ax1.minorticks_on()  # Set major and minor ticks to same length
            ax2.minorticks_off()
            ax1.grid('minor')
            plt.show()

        if show:
            pandas.set_option("display.precision", 2)
            freq_bands = []
            absorption = []
            for key, value in available_data.items():
                freq_bands.append(float(f'{key:0.2}'))
                absorption.append(float(f'{value:0.2}'))
            data = {'Bands [Hz]': freq_bands, 'Absorption [-]': absorption}
            df = pandas.DataFrame(data=data).set_index('Bands [Hz]').T
            df = df.style.set_caption(f'1/{nthOct} Octave Absorption Data')

            return df

        if returnValues:
            return bands[:, 1], result, available_data

    def field_impedance(self, z):

        A = 1 / z

        self.z_normal = z[:, 0]

        Af1 = A * np.sin(np.deg2rad(self.incidence_angle))
        Af2 = np.sin(np.deg2rad(self.incidence_angle))
        Af_div = Af1 / Af2

        Af = integrate.simps(Af_div, np.deg2rad(self.incidence_angle))

        return 1 / Af

    def compute(self, rigid_backing=True, conj=False, show_layers=True):
        """
        Calculates the final transfer matrix for the existing layers.

        Input:
         - rigid_backing: bool, if True adds a rigid layer to the end of the device
        """

        self.matrix = dict(collections.OrderedDict(sorted(self.matrix.items())))

        Tg = self.matrix[0]['matrix']
        for matrix in range(len(self.matrix) - 1):
            Tg = np.einsum('ijna,jkna->ikna', Tg, self.matrix[matrix + 1]['matrix'])

        Ag = Tg[0, 0]
        Bg = Tg[0, 1]
        Cg = Tg[1, 0]
        Dg = Tg[1, 1]

        if rigid_backing:
            zrad = 0

        else:
            # Radiation impedance for an unflanged circular tube in an infinite baffle
            zrad = self.z0 * (0.25 * (self.w0 * self.srad) ** 2 + 1j * 0.61 * self.w0 * self.srad)
            zrad = zrad.reshape((len(zrad), 1))
        zc = self.s0 * (Ag + (Bg * zrad / self.srad)) / (Cg + (Dg * zrad / self.srad))

        if self.incidence == 'diffuse':
            zc = self.field_impedance(zc)

        if not conj:
            self.z = zc
        else:
            self.z = np.conj(zc)

        self.matrix[len(self.matrix)] = {'rigid_backing': rigid_backing,
                                         'impedance_conjugate': conj}

        if show_layers:
            self.show_layers()


def find_nearest(array, value):
    """
    Function to find closest frequency in frequency array. Returns closest value and position index.
    """
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx