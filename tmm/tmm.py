"""
Toolbox for design and prediction of multilayered acoustic treatments. 
Also contains a material model based on the GRAS database.

Developed by Rinaldi Petrolli. 
For questions about usage, bugs, licensing and/or contributions contact me at rinaldipp@gmail.com.

References
----------
[1] R. Petrolli, A. Zorzo and P. D'Antonio, " Comparison of measurement and prediction for acoustical treatments 
    designed with Transfer Matrix Models ", in Euronoise, October 2021.

For further information check the function specific documentation.
"""
import collections
import os
import time

import numpy as np
import numpy.matlib as matlib
import pandas
import xlsxwriter
from matplotlib import pyplot as plt
from matplotlib import style
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.signal import butter, freqz, savgol_filter
from scipy.special import jv

from tmm import _h5utils as h5utils
from tmm import _plot as plot
from tmm import _utils as utils
from tmm.database.path import path as database_path

plot.set_style()


class TMM:
    """"
    Transfer Matrix Method for design and prediction of multilayered acoustic treatments.
    """
    def __init__(self, fmin=20, fmax=5000, df=1, incidence="diffuse", incidence_angle=None, project_folder=None,
                 filename=None, color=None, x_scale="lin"):
        """
        Parameters
        ----------
        fmin : int, optional
            Minimum frequency of interest.
        fmax : int, optional
            Maximum frequency of interest.
        df : int, optional
            Frequency resolution.
        incidence : string, optional
            String containing the desired type of incidence. 'normal' for normal incidence and 'diffuse' for and  field
            incidence approximation. 'diffuse' incidence might not be realistic for all treatment types.
        incidence_angle : list of ints, optional
            List containing the minimum and maximum incidence angles and the step size.
        project_folder : string, optional
            Path to which files will be saved. If None is passed the current directory will be used.
        filename : string, optional
            Filename that will be used to save data and image files.
        color : string, optional
            String representing Matplolib color - used for plotting only.
        x_scale : string, optional
            X axis scale - 'lin' for linear or 'log' for logarithmic.
        """
        if incidence_angle is None:
            incidence_angle = [0, 78, 1]

        self._fmin = fmin
        self._fmax = fmax
        self._freq = None
        self._df = df
        self._s0 = 1
        self._srad = 1
        self._air_prop = utils.AirProperties().standardized_c0_rho0()
        self._incidence = incidence
        self._incidence_angle = incidence_angle
        self._z = None
        self._z_angle = None
        self._scat = None
        self._matrix = {}
        self._project_folder = project_folder
        self._filename = filename
        self._display_name = None
        self._color = color
        self._params = {}
        self._x_scale = x_scale

    def __repr__(self):
        return f"TMM_{self.filename}_{len(self.matrix) - 1}layers_{self.first_peak[0]:0.0f}Hz"

    @property
    def fmin(self):
        """Return minimum frequency of analysis."""
        return self._fmin

    @fmin.setter
    def fmin(self, new_fmin):
        """Set new minimum frequency value."""
        self._fmin = new_fmin

    @property
    def fmax(self):
        """Return maximum frequency of analysis."""
        return self._fmax

    @fmax.setter
    def fmax(self, new_fmax):
        """Set new maximum frequency value."""
        self._fmax = new_fmax

    @property
    def df(self):
        """Return frequency resolution."""
        return self._df

    @df.setter
    def df(self, new_df):
        """Set new frequency resolution value."""
        self._df = new_df

    @property
    def freq(self):
        """Return frequency values."""
        if self._freq is None:
            if self._x_scale == "lin":
                return np.linspace(self.fmin,
                                   self.fmax,
                                   int((self.fmax - self.fmin) / self.df) + 1).round(1)
            elif self._x_scale == "log":
                return np.logspace(np.log10(self.fmin),
                                   np.log10(self.fmax),
                                   int((self.fmax - self.fmin) / self.df) + 1).round(1)
        else:
            return self._freq

    @freq.setter
    def freq(self, new_freq):
        """Sets frequency values."""
        self._freq = new_freq

    @property
    def air_prop(self):
        """Return air properties dictionary."""
        return self._air_prop

    @property
    def rho0(self):
        """Return air density."""
        return self.air_prop["air_density"]

    @property
    def c0(self):
        """Return speed of sound."""
        return self.air_prop["speed_of_sound"]

    @property
    def w0(self):
        """Return angular frequency values."""
        return 2 * np.pi * self.freq

    @property
    def k0(self):
        """Return wavenumber of air."""
        return self.w0 / self.c0

    @property
    def s0(self):
        """Return device front termination area."""
        return self._s0

    @s0.setter
    def s0(self, new_s0):
        """Set device front termination area."""
        self._s0 = new_s0

    @property
    def srad(self):
        """Return device rear termination area."""
        return self._srad

    @srad.setter
    def srad(self, new_srad):
        """Set device rear termination area."""
        self._srad = new_srad

    @property
    def z0(self):
        """Return air impedance."""
        return self.rho0 * self.c0

    @property
    def z(self):
        """Return surface impedance."""
        if self._z is not None:
            return self._z
        else:
            return np.zeros_like(self.freq, dtype="complex")

    @z.setter
    def z(self, new_z):
        """Set surface impedance."""
        self._z = new_z

    @property
    def z_angle(self):
        """Return angle-dependent surface impedance."""
        if self._z_angle is not None:
            return self._z_angle
        else:
            return np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

    @z_angle.setter
    def z_angle(self, new_z_angle):
        """Set angle-dependent surface impedance."""
        self._z_angle = new_z_angle

    @property
    def y(self):
        """Return admittance."""
        return 1 / self.z

    @property
    def alpha(self):
        """Return absorption coefficient."""
        _, alpha = self.reflection_and_absorption_coefficient(self.z)
        return alpha.reshape((len(alpha),))

    @property
    def first_peak(self):
        """Return the frequency in Hz and the absorption coefficient of the first absorption peak."""
        idx_array = np.diff(np.sign(np.diff(self.alpha))).nonzero()[0]
        if idx_array.size == 0:
            return self.freq[np.argmax(self.alpha)], np.amax(self.alpha)
        else:
            if self.alpha[idx_array[0] + 1] >= 0.3:
                return self.freq[idx_array[0] + 1], self.alpha[idx_array[0] + 1]
            else:
                return self.freq[idx_array[-1] + 1], self.alpha[idx_array[-1] + 1]

    @property
    def scat(self):
        """Return scattering coefficient (given by material_model only)."""
        if self._scat is not None:
            return self._scat
        else:
            return np.zeros_like(self.freq)

    @scat.setter
    def scat(self, new_scat):
        """Set scattering coefficient."""
        self._scat = new_scat

    @property
    def z_norm(self):
        """Return normalized surface impedance."""
        return self.z / self.z0

    @property
    def y_norm(self):
        """Return normalized surface admittance."""
        return 1 / self.z_norm

    @property
    def incidence(self):
        """Return incidence."""
        return self._incidence

    @property
    def incidence_angle(self):
        """Return incidence angle values."""
        if self.incidence == "diffuse":
            return np.linspace(self._incidence_angle[0] + 0.01, self._incidence_angle[1] - 0.01,
                               int((self._incidence_angle[1] - self._incidence_angle[0]) / self._incidence_angle[2]))
        elif self.incidence == "normal":
            return np.linspace(0, 1, 1)
        elif self.incidence == "angle":
            return np.linspace(self._incidence_angle[0], self._incidence_angle[0] + 1, 1)

    @property
    def matrix(self):
        """Return transfer matrix dictionary."""
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix):
        """Set transfer matrix dictionary."""
        self._matrix = new_matrix

    @property
    def depth(self):
        """Returns the treatment depth in millimeters."""
        return sum(value["thickness [mm]"] for value in self.matrix.values() if "thickness [mm]" in value)

    @property
    def color(self):
        """Return color string."""
        return self._color

    @color.setter
    def color(self, new_color):
        """Set color string."""
        self._color = new_color

    @property
    def project_folder(self):
        """Return project folder."""
        if self._project_folder is not None:
            return self._project_folder
        else:
            return os.getcwd()

    @project_folder.setter
    def project_folder(self, new_folder):
        """Set project folder."""
        self._project_folder = new_folder

    @property
    def filename(self):
        """Return filename string."""
        if self._filename is not None:
            return self._filename
        else:
            return "TMM"

    @property
    def display_name(self):
        """Return optional display name string."""
        return self._display_name

    @filename.setter
    def filename(self, new_filename):
        """Set filename."""
        self._filename = new_filename

    @property
    def params(self):
        """Return dictionary with extra parameters."""
        return self._params

    @params.setter
    def params(self, new_params):
        """Set extra params dictionary."""
        if isinstance(new_params, dict):
            self._params = new_params
        else:
            raise TypeError("Extra parameters have to be inside a dictionary.")
        
    def reflection_and_absorption_coefficient(self, zs):
        """
        Calculate reflection coefficient and absorption coefficient for a given surface impedance.

        Parameters
        ----------
        zs : array
            Surface impedance.

        Returns
        -------
        Reflection coefficient and absorption coefficient arrays.
        """
        r = (zs - self.z0) / (zs + self.z0)
        alpha = 1 - np.abs(r) ** 2

        return r, alpha

    def alpha_angle(self, angle_idx=0):
        """
        Return angle-dependent absorption coefficient.

        Parameters
        ----------
        angle_idx : int, optional
            Positional index of the desired angle in 'self.incidence_angle'.

        Returns
        -------
        Angle-dependent absorption coefficient.
        """
        _, alpha = self.reflection_and_absorption_coefficient(self.z_angle[:, angle_idx])

        return alpha

    def equivalent_fluid_model(self, sigma, model="mac", fibre_type=1):
        """
        Calculates the wavenumber (kc) and characteristic impedance (zc) using models of equivalent fluids that
        represent porous materials. For limitations and applications see the references at each section below.

        Parameters
        ----------
        sigma : int
            Flow resistivity of the porous material [k*Pa*s/m²]
        model : string, optional
            Name of the empirical model.
        fibre_type : int, optional
            Fibre type for Mechel and Grundmann model. 1 for basalt or rock wool and  2  for glass fibre.

        Returns
        -------
        Propagation constant array and characteristic acoustic impedance array.
        """
        coefficients = {  # List of coefficients for each available model
            "db": [0.0978, 0.7, 0.189, 0.595, 0.0571, 0.754, 0.087, 0.732],  # Delaney-Bazley
            "miki": [0.122, 0.618, 0.18, 0.618, 0.079, 0.632, 0.12, 0.632],  # Miki
            "qunli": [0.188, 0.554, 0.163, 0.592, 0.209, 0.548, 0.105, 0.607],  # Qunli
            "mechel_gf_lowX": [0.135, 0.646, 0.396, 0.458, 0.0688, 0.707, 0.196, 0.549],  # Mechel, glass fiber, low X
            "mechel_gf_highX": [0.102, 0.705, 0.179, 0.674, 0.0235, 0.887, 0.0875, 0.77],  # Mechel, glass fiber, high X
            "mechel_rf_lowX": [0.136, 0.641, 0.322, 0.502, 0.081, 0.699, 0.191, 0.556],  # Mechel, rock fiber, low X
            "mechel_rf_highX": [0.103, 0.716, 0.179, 0.663, 0.0563, 0.725, 0.127, 0.655],  # Mechel, rock fiber, high X
            "komatsu": [0.0004, -6.2, 0.0069, -4.1, 0.00027, -6.2, 0.0047, -4.1],  # Komatsu
            "mac": [0.0982, 0.685, 0.288, 0.526, 0.0729, 0.66228, 0.187, 0.5379],  # Modified Allard and Champoux
        }

        if model in coefficients.keys():
            """
            Empirical models based on linear regressions. For information about the applicability of each model see
            Table 2-10 at https://doc.comsol.com/5.5/doc/com.comsol.help.aco/aco_ug_pressure.05.144.html.

            Based on the findings in Sound absorption of porous materials – Accuracy of prediction methods
            by David Oliva & Valtteri Hongisto (2013) and implemented with the Equations 6.9 and 6.10 of 
            chapter 6.5.1 in the book Acoustic Absorbers and Diffusers (3rd Edition) by Trevor Cox and Peter D'Antonio.
            """
            c = coefficients[model]
            X = self.rho0 * self.freq / sigma
            kc = self.k0 * (1 + c[0] * X ** -c[1] - 1j * c[2] * X ** -c[3])  # Wavenumber
            zc = self.z0 * (1 + c[4] * X ** -c[5] - 1j * c[6] * X ** -c[7])  # Characteristic impedance

        elif model == "wilson":
            """
            Wilson's equivalent to Delaney and Bazley model from relaxation model available in 
            Acoustic Absorbers and Diffusers by Trevor Cox and Peter D'Antonio (3rd Edition).
            Equations 6.35 and 6.36 using simplifications detailed below.
            """
            X = self.rho0 * self.freq / sigma  # Dimensionless quantity
            omega = 1
            gamma = 1.4  # Ratio of specific heats
            q = 1
            zc = self.z0 * (q / omega) / np.sqrt(
                (1 + (gamma - 1) / np.sqrt(1 + 1j * 19 * X)) * (1 - 1 / np.sqrt(1 + 1j * 13 * X)))
            kc = (q * self.k0) * np.sqrt(
                (1 + (gamma - 1) / np.sqrt(1 + 1j * 19 * X)) / (1 - 1 / np.sqrt(1 + 1j * 13 * X)))

        elif model == "mechel_grundmann" or model == "mg":
            """
            Mechel and Grundmann formulations - Equation 6.12 in  Acoustic Absorbers and Diffusers by 
            Trevor Cox and Peter D'Antonio (3rd Edition).
            """
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
                betaz = []
                print("Choose fibre type  between 1 (basalt or rock wool) or  2 (glass fibre).")

            # Calculate impedance of porous material
            X = self.rho0 * self.freq / sigma  # Dimensionless quantity

            # Mechel Grundmann
            kc = -1j * self.k0 * (betak[0] * X ** -1 + betak[1] * X ** -0.5 + betak[2] + betak[3] * X ** 0.5 +
                                  betak[4] * X + betak[5] * X ** 1.5)
            zc = self.z0 * (betaz[0] * X ** -1 + betaz[1] * X ** -0.5 + betaz[2] + betaz[3] * X ** 0.5 +
                            betaz[4] * X + betaz[5] * X ** 1.5)
        else:
            available_models = [key for key in coefficients.keys()]
            available_models.append("wilson")
            available_models.append("mechel_grundmann")
            raise NameError("Unidentified model. Choose between the available models: ", available_models)

        return kc, zc

    def porous_layer(self, sigma=27, t=5, model="mac", fibre_type=1, layer=None):
        """
        Adds a layer of porous material to the existing device.

        Parameters
        ----------
        sigma : float or int, optional
            Flow resistivity of the porous material [k*Pa*s/m²]
        t : float or int, optional
            Thickness of the porous material [mm]
        model : string, optional
            Name of the empirical model.
        fibre_type : int , optional
            Fibre type for Mechel and Grundmann model. 1 for basalt or rock wool and  2  for glass fibre.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        sigma_k = sigma * 1000  # Convert to kilo rayls/m

        kc, zc = self.equivalent_fluid_model(sigma_k, model=model, fibre_type=fibre_type)

        kc = matlib.repmat(kc, len(self.incidence_angle), 1).T
        zc = matlib.repmat(zc, len(self.incidence_angle), 1).T

        Tp = np.array([[np.cos(kc * t_meters), 1j * zc / self.s0 * np.sin(kc * t_meters)],
                       [1j * self.s0 / zc * np.sin(kc * t_meters), np.cos(kc * t_meters)]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "porous_layer",
                              "flow_resistivity [k*Pa*s/m²]": sigma,
                              "thickness [mm]": t,
                              "model": model,
                              # "kc": kc,
                              # "zc": zc,
                              "matrix": Tp,
                              }

    def air_layer(self, t=5, layer=None):
        """
        Adds an air layer to the existing device.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the air layer [mm]
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters

        k0 = matlib.repmat(self.k0, len(self.incidence_angle), 1).T
        k0 = np.sqrt(k0 ** 2 - k0 ** 2 * np.sin(np.deg2rad(self.incidence_angle)))
        z0 = matlib.repmat(self.z0, len(self.incidence_angle), 1).T

        Ta = np.array([[np.cos(k0 * t_meters), 1j * z0 / self.s0 * np.sin(k0 * t_meters)],
                       [1j * self.s0 / z0 * np.sin(k0 * t_meters), np.cos(k0 * t_meters)]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "air_layer",
                              "thickness [mm]": t,
                              "matrix": Ta,
                              }

    def constant_z(self, z=85):
        """
        Sets a constant impedance value across all the frequencies. Useful to define, rigid, semi-rigid or anechoic
        impedance/admittance/absorption values.

        A fully rigid surface has an impedance value close to infinity, while a fully anechoic surface is equivalent to
        the impedance of the medium, i.e. the air impedance.

        Parameters
        ----------
        z : int, optional
            Normalized impedance value.
        """
        self.z = np.full_like(self.freq, z)
        self.z_angle = np.full_like(self.freq, self.z[0], shape=(len(self.freq), len(self.incidence_angle)))
        self.matrix = {"termination": {"type": "constant_z"}}

    def membrane_layer(self, t=1, rho=8050, layer=None):
        """
        Adds a membrane to the existing device.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the membrane [mm]
        rho : float or int, optional
            Material density [kg/m³]
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters

        zc = 1j * self.w0 * rho * t_meters / self.s0
        zc = matlib.repmat(zc, len(self.incidence_angle), 1).T

        ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

        Tm = np.array([[ones, zc],
                       [zeros, ones]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "membrane_layer",
                              "thickness [mm]": t,
                              "density [kg/m³]": rho,
                              "matrix": Tm,
                              }

    def perforated_panel_layer(self, t=19, d=8, s=16, open_area=None, rho=None, end_correction="jb", method="barrier",
                               layer=None):
        """
        Adds a plate with circular perforations to the existing device.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the perforated plate [mm]
        d : float or int, optional
            Hole diameter [mm]
        s : float or int, optional
            Hole spacing from the center of one hole to the next [mm]
        open_area : float, optional
            Ration of open area. If set to 'None' it will be calculated with the hole spacing 's'. If set to a value
            the equivalent spacing will be calculated and overwrite the existing value.
        rho : float, int or None, optional
            Plate density [kg/m3] - if 'None' is passed than a fully rigid plate is assumed.
        end_correction : string, optional
            Chooses between the available end corrections for the tube length.
        method : string, optional
            Chooses between the available methods to calculate the perforated plate impedance.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        d_meters = d / 1000
        s_meters = s / 1000

        if d < 2 / s:
            if method != "barrier_mpp":
                print(f"WARNING: Hole spacing might be too small for {d} [mm] hole diameter.")

        if open_area is None:
            open_area = np.pi / ((2 * s_meters / d_meters) ** 2)
        else:
            s = d / 2 * np.sqrt(np.pi / open_area)

        if open_area > 1:
            raise ValueError("Open area cannot be greater than 1.")

        t_corr = None
        if end_correction == "nesterov":
            # Acoustic Absorbers and Diffusers by Trevor Cox and Peter D'Antonio
            # Circular holes in circular pattern
            delta = 0.8 * (1 - 1.47 * open_area ** (1/2) + 0.47 * open_area ** 1.5)
            t_corr = 2 * delta * d_meters / 2 + t_meters
        elif end_correction == "jaouen_becot" or end_correction == "jb":
            # Acoustic Absorbers and Diffusers by Trevor Cox and Peter D'Antonio
            # Circular holes in square pattern
            delta = 0.8 * (1 - 1.13 * open_area ** (1/2) - 0.09 * open_area + 0.27 * open_area ** (3/2))
            t_corr = 2 * delta * d_meters / 2 + t_meters
        elif end_correction == "beranek":
            # Leo Beranek - Acoustics
            t_corr = t_meters + 0.85 * d_meters
        vis = self.air_prop["air_viscosity"]

        Tp = None
        if method == "barrier":
            """
            Impedance from section 7.3.1 of  Acoustic Absorbers and Diffusers 
            by Trevor Cox and Peter D'Antonio (3rd Edition).
            """
            rm = (self.rho0 / open_area) * np.sqrt(8 * vis * self.w0) * (
                        1 + t_meters / (d_meters))  # Surface resistance
            zpp = (1j / open_area) * t_corr * self.w0 * self.rho0 + rm  # Impedance of perforated plate
            if rho:
                mip = 1j * self.w0 * rho * t_meters * (1 - open_area) / self.s0  # Mass impedance of the plate
                zpp = 1 / (1 / zpp + 1 / mip)
            zpp = matlib.repmat(zpp, len(self.incidence_angle), 1).T

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Tp = np.array([[ones, zpp],
                           [zeros, ones]])

        if method == "barrier_mpp":
            """
            Impedance for micro perforated plate by MAA in Potential of microperforated panel absorber (1998).

            Not normalized by air impedance, that's why the rm is not divided by rho0*c0 and m is multiplied by rho0 
            instead being divided by c0.
            """
            cis = d_meters * np.sqrt(self.w0 * self.rho0 / (4 * vis))
            kr = np.sqrt(1 + cis ** 2 / 32) + np.sqrt(2) / 32 * cis * d_meters / t_meters
            rm = 32 * vis / open_area * t_meters / d_meters ** 2 * kr
            km = 1 + 1 / np.sqrt(1 + cis ** 2 / 2) + 0.85 * d_meters / t_meters
            m = self.rho0 * t_meters / open_area * km

            zpp = rm + 1j * self.w0 * m
            if rho:
                mip = 1j * self.w0 * rho * t_meters * (1 - open_area) / self.s0  # Mass impedance of the plate
                zpp = 1 / (1 / zpp + 1 / mip)
            zpp = matlib.repmat(zpp, len(self.incidence_angle), 1).T

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Tp = np.array([[ones, zpp],
                           [zeros, ones]])

        elif method == "eq_fluid":
            """
            Impedance calculated through Zwikker and Kosten's viscothermal model.
            """
            kc, zc = self.viscothermal_circular(d_meters, open_area)
            kc = matlib.repmat(kc, len(self.incidence_angle), 1).T
            zc = matlib.repmat(zc, len(self.incidence_angle), 1).T

            Tp = np.array([[np.cos(kc * t_corr), 1j * zc / self.s0 * np.sin(kc * t_corr)],
                           [1j * self.s0 / zc * np.sin(kc * t_corr), np.cos(kc * t_corr)]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "perforated_panel_layer",
                              "thickness [mm]": t,
                              "hole_diameter [mm]": d,
                              "hole_spacing [mm]": s,
                              "open_area [%]": open_area * 100,
                              "end_correction": end_correction,
                              "rho [kg/m3]": rho,
                              "method": method,
                              "matrix": Tp,
                              }

    def slotted_panel_layer(self, t=19, w=8, s=16, open_area=None, rho=None, method="barrier", layer=None):
        """
        Adds a plate with rectangular slits to the existing device.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the slotted plate [mm]
        w: float or int, optional
            Slit width [mm]
        s : float or int, optional
            Slit spacing from the center of one slit to the next [mm]
        open_area : float, optional
            Ration of open area. If set to 'None' it will be calculated with the hole spacing 's'. If set to a value
            the equivalent spacing will be calculated and overwrite the existing value.
        rho : float, int or None, optional
            Plate density [kg/m3] - if 'None' is passed than a fully rigid plate is assumed.
        method : string, optional
            Chooses between the available methods to calculate the perforated plate impedance.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.
        """
        if open_area is None:
            open_area = w / s
        else:
            s = w / open_area

        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        w_meters = w / 1000
        s_meters = s / 1000

        if open_area > 1:
            raise ValueError("Slit spacing must be larger than slit width.")

        t_corr = t_meters + 2 * w_meters * (-1 / np.pi) * np.log(np.sin(0.5 * np.pi * open_area))
        Ts = None
        if method == "barrier":
            """
            Impedance from "On the design of resonant absorbers using a slotted plate" by Kristiansen and Vigran.
            """
            vis = self.air_prop["air_viscosity"]
            Rp = 0.5 * np.sqrt(2 * vis * self.rho0 * self.w0) * (4 + (2 * t) / w)
            Xp = self.rho0 * t_corr
            zs = (Rp + 1j * self.w0 *  Xp) / open_area
            if rho:
                mip = 1j * self.w0 * rho * t_meters * (1 - open_area) / self.s0  # Mass impedance of the plate
                zs = 1 / (1 / zs + 1 / mip)

            zs = matlib.repmat(zs, len(self.incidence_angle), 1).T

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Ts = np.array([[ones, zs],
                           [zeros, ones]])

        elif method == "eq_fluid":
            """
            Impedance calculated through M. Biot's viscothermal model.
            """
            kc, zc = self.viscothermal_slit(w_meters, open_area)
            kc = matlib.repmat(kc, len(self.incidence_angle), 1).T
            zc = matlib.repmat(zc, len(self.incidence_angle), 1).T

            Ts = np.array([[np.cos(kc * t_corr), 1j * zc / self.s0 * np.sin(kc * t_corr)],
                           [1j * self.s0 / zc * np.sin(kc * t_corr), np.cos(kc * t_corr)]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "slotted_panel_layer",
                              "thickness [mm]": t,
                              "slot_width [mm]": w,
                              "slot_spacing [mm]": s,
                              "open_area [%]": open_area * 100,
                              "rho [kg/m3]": rho,
                              "method": method,
                              "matrix": Ts,
                              }

    def viscothermal_circular(self, d, open_area):
        """
        Zwikker and Kosten viscothermal model for circular cross-section.

        Parameters
        ----------
        d : int or float
            Hole diameter [m]
        open_area : int or float
            Plate porosity (% open area) [-]

        Returns
        -------
        Propagation constant array and characteristic acoustic impedance array.
        """

        vis = self.air_prop["air_viscosity"]
        gam = self.air_prop["specific_heat_ratio"]
        pn = self.air_prop["prandtl_number"]
        p0 = self.air_prop["atmospheric_pressure"]

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

        Parameters
        ----------
        w : int or float
            Slit width [m]
        open_area : int or float
            Plate porosity (% open area) [-]

        Returns
        -------
        Propagation constant array and characteristic acoustic impedance array.
        """
        vis = self.air_prop["air_viscosity"]
        gam = self.air_prop["specific_heat_ratio"]
        pn = self.air_prop["prandtl_number"]
        p0 = self.air_prop["atmospheric_pressure"]

        beta = w / 2 * np.sqrt(self.w0 * self.rho0 / vis)

        rhoef = self.rho0 / open_area * 1 / (1 - np.tanh(beta * np.sqrt(1j)) / (beta * np.sqrt(1j)))
        kef = p0 / open_area * gam / (1 + (gam - 1) * np.tanh(beta * np.sqrt(1j * pn) / (beta * np.sqrt(1j * pn))))
        kc = self.w0 * np.sqrt(rhoef / kef)
        zc = np.sqrt(rhoef * kef)

        return kc, zc

    def material_model(self, type="door", params=None):
        """
        Models for different surfaces in the GRAS database published in the supplemental data of
        "area framework for auralization of boundary element method simulations including source and receiver
        directivity" by Jonathan area. Hargreaves, Luke R. Rendell, and Yiu W. Lam.

        GRAS database: https://depositonce.tu-berlin.de//handle/11303/7506
        Supplemental data: https://asa.scitation.org/doi/suppl/10.1121/1.5096171

        Available materials:
        -------------------
         - Floor
         - Ceiling
         - Door
         - Concrete
         - Plaster
         - MDF
         - Window

        Parameters
        ----------
        type : str, optional
            String descriptor of the desired material available in the database.
        params : dict, optional
            Dictionary containing calculation parameters for 'door' and 'window' materials. See the docstrings below.
        """
        if type == "floor":
            """
            This is a model of the floor material defined in Scene 9 of the GRAS database. 
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_floor.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = CubicSpline(fMeas, YsMeas, bc_type="natural")
            Sf = CubicSpline(fMeas, sMeas, bc_type="natural")
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "ceiling":
            """
            This is a model of the ceiling material defined in Scene 9 of the GRAS database. 
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_ceiling.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = CubicSpline(fMeas, YsMeas, bc_type="natural")
            Sf = CubicSpline(fMeas, sMeas, bc_type="natural")
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "door":
            """
            This is a model of the door material defined in Scene 9 of the GRAS database. It is entirely fabricated 
            from other datasets, since no data was given for this component in the GRAS database. It attempts to define 
            realistic boundary conditions in a case where insufficient data exists, and is included in this work to 
            illustrate the sort of compromises that are often necessary, rather than to propose a specific model for 
            these materials. The reader is asked to consider it with these caveats in mind.

            It comprises two approaches:

            1) area purely resistive fit to octave-band summed absorption and transmission coefficient data. 
               Both absorption and transmission coefficients were used since the former did not rise at low frequencies, 
               indicating that the data in the dataset use was most likely measured for doors on the floor of a 
               reverberation room, hence transmission would be zero. From the perspective of this application, 
               transmission is another mechanism by which energy is lost and should be included in absorption, 
               hence the coefficients are summed.

            2) area reactive Mass-Spring-Damper model of the assumed fundamental resonance of the door panel. This was 
               included since such effects are well known to be reactive, and this affects room modal frequencies. 
               The Mass value was chosen to be consistent with the assumed material. Stiffness and Damping values were 
               tuned to the desired absorption peak frequency and bandwidth. This did not however produce sufficient 
               absorption to join with the trend in 1, so an additional amount of purely resistive absorption was also 
               added.

            These are combined using the non-linear crossover of Aretz el al.

            Parameters
            ----------
            sample_rate : int
                Sampling rate [Hz]
            crossover_frequency : int
                Crossover frequency between the models [Hz]
            rho_m : int or float
                Assumed bulk density [kg/m^3]
            d : float
                Assumed thickness [m]
            area : float
                Area [m^2]
            f_res : int or float
                Assumed fundamental panel resonance frequency [Hz]
            smooth : bool
                Boolean to choose whether apply smoothing to the curve or not.
            """
            # Model 1: purely resistive fit to octave-band absorption data:
            if params is None:
                params = {"sample_rate": 44100, "crossover_frequency": 250, "rho_m": 375, "d": 0.043, "area": 2.2 * 0.97,
                          "f_res": 95, "smooth": False}

            sample_rate = params["sample_rate"]
            crossover_frequency = params["crossover_frequency"]
            rho_m = params["rho_m"]
            d = params["d"]
            area = params["area"]
            f_res = params["f_res"]
            smooth = params["smooth"]

            # Measured data:
            fMeas = [125, 250, 500, 1000, 2000, 4000, ]  # Octave band centre frequencies (Hz)
            aMeas = np.asarray([0.14, 0.10, 0.06, 0.08, 0.1, 0.1, ]) + \
                    np.asarray([0.07, 0.01, 0.02, 0.03, 0.01, 0.01, ])  # Absorption and Transmission coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = CubicSpline(fMeas, YsMeas, bc_type="natural")
            Ys1 = Yf(self.freq)

            # Model 2: reactive Mass-Spring-Damper fit to assumed fundamental panel resonance:

            M = rho_m * d * area  # Mass term
            K = M * (2 * np.pi * f_res) ** 2  # Stiffness term  - adjusted to match assumed f_res
            R = 12000  # Resistance term - adjusted to match measured coefficients
            zS = (-1j * 2 * np.pi * self.freq) * M + R + K / (-1j * 2 * np.pi * self.freq)  # Surface impedance
            Ys2 = self.rho0 * self.c0 / zS  # Specific admittance

            # Additional resistive component:
            aExtra = np.mean(aMeas[2::])
            YsExtra = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aExtra)) / (1 + np.sqrt(1 - aExtra))
            Ys2 = Ys2 + YsExtra

            # Define Butterworth filters.
            # Note these are applied twice to make Linkwitz-Riley:
            B_HP, A_HP = butter(8, crossover_frequency * 2 / sample_rate, "high")
            B_LP, A_LP = butter(8, crossover_frequency * 2 / sample_rate, "low")

            # Non-linear crossover method of Aretz et al:
            Ys = np.abs(Ys2 * np.conj(freqz(B_LP, A_LP, self.freq, fs=sample_rate)[1]) ** 2) + \
                 np.abs(Ys1 * np.conj(freqz(B_HP, A_HP, self.freq, fs=sample_rate)[1]) ** 2)  # Add the magnitudes only

            Ys = Ys * np.exp(1j * np.angle(Ys2))  # Multiply the phase from MSD model back in

            if smooth:
                Ys_real = savgol_filter(np.real(Ys), 31, 3)
                Ys_imag = savgol_filter(np.imag(Ys), 31, 3)
                Ys = Ys_real + 1j * Ys_imag

            self.z = 1 / Ys

        elif type == "concrete":
            """
            This is a model of the concrete material defined in Scene 9 of the GRAS database. 
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_concrete.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = CubicSpline(fMeas, YsMeas, bc_type="natural")
            Sf = CubicSpline(fMeas, sMeas, bc_type="natural")
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "plaster":
            """
            This is a model of the plaster material defined in Scene 9 of the GRAS database. 
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_plaster.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = CubicSpline(fMeas, YsMeas, bc_type="natural")
            Sf = CubicSpline(fMeas, sMeas, bc_type="natural")
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "mdf":
            """
            This is a model of the MDF material defined in Scene 9 of the GRAS database. 
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_MDF25mmA_plane_00deg.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = CubicSpline(fMeas, YsMeas, bc_type="natural")
            Sf = CubicSpline(fMeas, sMeas, bc_type="natural")
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "window":
            """
            This is a model of the windows material defined in Scene 9 of the GRAS database. It combines two approaches:

            1) area purely resistive fit to teh third-octave band absorption coefficient data provided with the 
               GRAS dataset.

            2) area reactive Mass-Spring-Damper model of the assumed fundamental resonance of the window panels. 
               This was included since such effects are well known be reactive, and this affects room modal frequencies. 
               It was also deemed necessary since the fundamental resonance of the panels appeared to be lower than the 
               bandwidth the measured dataset extended to (absorption rose quite sharply at the lowest frequencies). 
               The Mass value was chosen to be consistent with the assumed material. Stiffness and Damping values were 
               tuned to the desired absorption peak frequency and bandwidth. This did not however produce sufficient 
               absorption to join with the trend in 1, so an additional amount of purely resistive absorption was also 
               added.

            These are combined using the non-linear crossover of Aretz el al.

            Note that this script attempts to define realistic boundary conditions in a case where insufficient data 
            exists, and is included in this work to illustrate the sort of compromises that are often necessary, rather 
            than to propose a specific model for these materials. The reader is asked to consider it with these caveats 
            in mind.

            Parameters
            ----------
            sample_rate : int
                Sampling rate [Hz]
            crossover_frequency : int
                Crossover frequency between the models [Hz]
            rho_m : int or float
                Assumed bulk density [kg/m^3]
            d : float
                Assumed thickness [m]
            area : float
                Area [m^2]
            f_res : int or float
                Assumed fundamental panel resonance frequency [Hz]
            smooth : bool
                Boolean to choose whether apply smoothing to the curve or not.
            """
            # Model 1: purely resistive fit to provided third-octave-band absorption data:
            if params is None:
                params = {"sample_rate": 44100, "crossover_frequency": 200, "rho_m": 2500, "d": 0.0067, "area": 5.33,
                          "f_res": 6.66, "smooth": False}

            sample_rate = params["sample_rate"]
            crossover_frequency = params["crossover_frequency"]
            rho_m = params["rho_m"]
            d = params["d"]
            area = params["area"]
            f_res = params["f_res"]
            smooth = params["smooth"]

            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_windows.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = CubicSpline(fMeas, YsMeas, bc_type="natural")
            Sf = CubicSpline(fMeas, sMeas, bc_type="natural")
            Ys1 = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.scat = SsInterp

            # Model 2: reactive Mass-Spring-Damper fit to assumed fundamental panel resonance:
            M = rho_m * d * area  # Mass term
            K = M * (2 * np.pi * f_res) ** 2  # Stiffness term  - adjusted to match assumed f_res
            R = 6000  # Resistance term - adjusted to match measured coefficients
            zS = (-1j * 2 * np.pi * self.freq) * M + R + K / (-1j * 2 * np.pi * self.freq)  # Surface impedance
            Ys2 = self.rho0 * self.c0 / zS  # Specific admittance

            # Additional resistive component:
            aExtra = aMeas[8]
            YsExtra = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aExtra)) / (1 + np.sqrt(1 - aExtra))
            Ys2 = Ys2 + YsExtra

            # Define Butterworth filters.
            # Note these are applied twice to make Linkwitz-Riley:
            B_HP, A_HP = butter(8, crossover_frequency * 2 / sample_rate, "high")
            B_LP, A_LP = butter(8, crossover_frequency * 2 / sample_rate, "low")

            # Non-linear crossover method of Aretz et al:
            Ys = np.abs(Ys2 * np.conj(freqz(B_LP, A_LP, self.freq, fs=sample_rate)[1]) ** 2) + \
                 np.abs(Ys1 * np.conj(freqz(B_HP, A_HP, self.freq, fs=sample_rate)[1]) ** 2)  # Add the magnitudes only

            Ys = Ys * np.exp(1j * np.angle(Ys2))  # Multiply the phase from MSD model back in

            if smooth:
                Ys_real = savgol_filter(np.real(Ys), 31, 3)
                Ys_imag = savgol_filter(np.imag(Ys), 31, 3)
                Ys = Ys_real + 1j * Ys_imag

            self.z = 1 / Ys

        else:
            available_types = ["floor", "ceiling", "door", "concrete", "plaster", "mdf", "window"]
            raise NameError(f"Available material models: {available_types}")

        self.z = self.z * self.z0
        if "_material_model" not in self.filename:
            self.filename = self.filename + "_material_model"
        self.matrix = {"material_model": {"type": type,
                                          "params": params}}

    def field_impedance(self, z):
        """
        Calculates the field impedance for a set of angle dependent impedances.

        Parameters
        ----------
        z : array
            Multidimensional array with angle dependent set of impedances.

        Returns
        -------
        Field impedance array.
        """
        A = 1 / z
        Af1 = A * np.sin(np.deg2rad(self.incidence_angle))
        Af2 = np.sin(np.deg2rad(self.incidence_angle))
        Af_div = Af1 / Af2
        Af = integrate.simps(Af_div, np.deg2rad(self.incidence_angle))
        z_field = 1 / Af
        return z_field

    def compute(self, rigid_backing=True, conj=False, show_layers=True):
        """
        Calculates the global transfer matrix for the existing layers.

        Parameters
        ----------
        rigid_backing : bool, optional
            If True adds a rigid layer to the end of the device, else an approximation for the
            radiation impedance is used.
        conj : bool, optional
            Option to conjugate the imaginary part of the impedance.
        show_layers : bool, optional
            Option to display the layers and their details.
        """
        if list(self.matrix.keys())[0] != "termination" and "material_model" not in self.filename:
            self.matrix = dict(collections.OrderedDict(sorted(self.matrix.items())))

            if "rigid_backing" in self.matrix[list(self.matrix.keys())[-1]]:
                self.matrix.pop(list(self.matrix.keys())[-1])
                rigid_backing = True

            Tg = self.matrix[0]["matrix"]
            for matrix in range(len(self.matrix) - 1):
                Tg = np.einsum("ijna,jkna->ikna", Tg, self.matrix[matrix + 1]["matrix"])

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
            self.z_angle = self.s0 * (Ag + (Bg * zrad / self.srad)) / (Cg + (Dg * zrad / self.srad))

            if self.incidence == "diffuse":
                zc = self.field_impedance(self.z_angle)
            else:
                zc = self.z_angle[:, 0]

            if not conj:
                self.z = zc
            else:
                self.z = np.conj(zc)
                self.z_angle = np.conj(self.z_angle)

            self.matrix[len(self.matrix)] = {"type": "backing",
                                             "rigid_backing": rigid_backing,
                                             "impedance_conjugate": conj,
                                             }

            if show_layers:
                self.show_layers()

    def clear_matrix(self):
        """Removes matrix data from self.matrix to reduce file size."""
        for matrix in self.matrix.keys():
            if "matrix" in list(self.matrix[matrix].keys()):
                self.matrix[matrix]["matrix"] = None

    def reduce_size(self):
        """Removes the value of some attributes to reduce file size."""
        self.clear_matrix()
        self._z_angle = None

    def rebuild(self):
        """Rebuild treatment layers to update frequency range."""
        matrix = self.matrix.copy()
        self.matrix = {}
        for key, value in matrix.items():
            if key == "termination":
                if value["type"] == "constant_z":
                    self.constant_z(self.z[0])
            elif key != "material_model":
                if value["type"] == "porous_layer":
                    self.porous_layer(sigma=value["flow_resistivity [k*Pa*s/m²]"],
                                      t=value["thickness [mm]"],
                                      model=value["model"],
                                      layer=key)
                elif value["type"] == "air_layer":
                    self.air_layer(t=value["thickness [mm]"],
                                   layer=key)
                elif value["type"] == "perforated_panel_layer":
                    self.perforated_panel_layer(t=value["thickness [mm]"],
                                                d=value["hole_diameter [mm]"],
                                                s=value["hole_spacing [mm]"],
                                                end_correction=value["end_correction"],
                                                rho=value["rho [kg/m3]"],
                                                method=value["method"],
                                                layer=key)
                elif value["type"] == "slotted_panel_layer":
                    self.slotted_panel_layer(t=value["thickness [mm]"],
                                             w=value["slot_width [mm]"],
                                             s=value["slot_spacing [mm]"],
                                             rho=value["rho [kg/m3]"],
                                             method=value["method"],
                                             layer=key)
                elif value["type"] == "membrane_layer":
                    self.membrane_layer(t=value["thickness [mm]"],
                                        rho=value["density [kg/m³]"],
                                        layer=key)
            else:
                self.material_model(value["type"], params=value["params"])

        if matrix[list(matrix.keys())[-1]]["type"] == "backing":
            self.compute(rigid_backing=matrix[list(matrix.keys())[-1]]["rigid_backing"],
                         conj=matrix[list(matrix.keys())[-1]]["impedance_conjugate"], 
                         show_layers=False)
        else:
            self.compute(rigid_backing=False, show_layers=False)

    def log_rebuild(self):
        """Logs a list of commands calls needed to recreate the TMM object."""
        matrix = self.matrix.copy()
        logged_calls = [f"# {self.filename.capitalize()}",
                        f"{self.filename} = TMM("
                        f"fmin={self.fmin}, "
                        f"fmax={self.fmax}, "
                        f"df={self.df:0.2f}, "
                        f"project_folder=fm.folder_path, "
                        f"incidence='diffuse', "
                        f"filename='{self.filename}')"]

        for key, value in matrix.items():
            if key == "termination":
                if value["type"] == "constant_z":
                    logged_calls.append(f"{self.filename}.constant_z({self.z[0]:0.3f})")
            elif key != "material_model":
                if value["type"] == "porous_layer":
                    logged_calls.append(
                        f"{self.filename}.porous_layer("
                        f"model='{value['model']}', "
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"sigma={value['flow_resistivity [k*Pa*s/m²]']})"
                        )
                elif value["type"] == "air_layer":
                    logged_calls.append(f"{self.filename}.air_layer(t={value['thickness [mm]']:0.1f})")
                elif value["type"] == "perforated_panel_layer":
                    logged_calls.append(
                        f"{self.filename}.perforated_panel_layer("
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"d={value['hole_diameter [mm]']:0.1f}, "
                        f"s={value['hole_spacing [mm]']:0.1f}, "
                        f"end_correction='{value['end_correction']}', "
                        f"rho={value['rho [kg/m3]']}, "
                        f"method='{value['method']}'"
                        f")"
                        )
                elif value["type"] == "slotted_panel_layer":
                    logged_calls.append(
                        f"{self.filename}.slotted_panel_layer("
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"w={value['slot_width [mm]']:0.1f}, "
                        f"s={value['slot_spacing [mm]']:0.1f}, "
                        f"rho={value['rho [kg/m3]']}, "
                        f"method='{value['method']}'"
                        f")"
                        )
                elif value["type"] == "membrane_layer":
                    logged_calls.append(
                        f"{self.filename}.membrane_layer("
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"rho={value['density [kg/m³]']}"
                        f")"
                        )
            else:
                logged_calls.append(f"{self.filename}.material_model('{value['type']}', params={value['params']})")

        if matrix[list(matrix.keys())[-1]]["type"] == "backing" and \
                matrix[list(matrix.keys())[-1]]["rigid_backing"] is True:
            logged_calls.append(f"{self.filename}.compute(rigid_backing=True, show_layers=True)")
        else:
            logged_calls.append(f"{self.filename}.compute(rigid_backing=False, show_layers=True)")

        logged_calls += [f"{self.filename}.plot(figsize=(7, 4), plots=['alpha'], save_fig=True, timestamp=False)",
                         f"{self.filename}.save()"]

        return logged_calls

    def print_rebuild(self):
        """Prints the list of commands calls needed to recreate the TMM object."""
        logged_calls = self.log_rebuild()
        for call in logged_calls:
            print(call)

    def show_layers(self, conversion=None):
        """
        Method to print each layer with its details.

        Parameters
        ----------
        conversion : list or float and string, optional
            List containing conversion ratio and string containing the name of the converted unit.
        """
        if conversion is None:
            conversion = [0.0393701, "[inches]"]
        print("Device properties:")
        print("\t(1 - Front face)")
        print(f"\t({len(self.matrix)} - Rear Face)")
        print(f"\tSound incidence: {self.incidence}")
        if self.incidence == "diffuse":
            print(f"\tAngle: {min(self.incidence_angle):0.0f}° - {max(self.incidence_angle):0.0f}°\n")
        else:
            print(f"\tAngle: {(self.incidence_angle[0]):0.0f}°\n")

        total_depth = 0
        for i in range(1, len(self.matrix) + 1):
            print(f"Layer {i}:")
            for key, value in self.matrix[i - 1].items():
                if key != "matrix":
                    if isinstance(value, str) or isinstance(value, bool):
                        print(f"\t{key}: ", value)
                    else:
                        if "[mm]" in key:
                            converted = key.replace("[mm]", conversion[1])
                            print(f"\t{key}: {value:0.2f} | {converted}: {value * conversion[0]:0.2f}")
                        elif value is None:
                            print(f"\t{key}: {None}")
                        else:
                            print(f"\t{key}: {value:0.2f}")
                        if "thickness" in key:
                            total_depth += value
        print(f"\nTotal treatment depth [mm]: {total_depth:0.2f} | " +
              f"Total treatment depth {conversion[1]}: {total_depth * conversion[0]:0.2f}")

    def filter_alpha(self, n_oct=1, view=True, show_table=False, **kwargs):
        """
        Filters the absorption coefficient into fractional octave bands. See tmm._plot.acoustic data for kwargs.

        Parameters
        ----------
        n_oct : int, optional
            Fractional octave bands that the absorption will be filtered to.
        view : bool, optional
            Boolean to display plot with filtered absorption.
        show_table : bool, optional
            Boolean to display the filtered values in a table.

        Returns
        -------
        Bands' center frequency array and filtered absorption array.
        """
        # bands, result = pytta.utils.filter_values(self.freq, self.alpha, nthOct=n_oct)
        bands, result = utils.filter_values(self.freq, self.alpha, n_oct=n_oct)

        # Plot
        if view:
            _, _ = plot.oct_filter(self.freq, self.alpha, bands, result, n_oct, "Absorption Coefficient [-]",
                                   filename=self.filename, project_folder=self.project_folder, **kwargs)
            plt.show()

        if show_table:
            pandas.set_option("display.precision", 2)
            freq_bands = []
            absorption = []
            absorption_percentual = []
            for i in range(len(bands)):
                freq_bands.append(float(f"{bands[i]:0.2f}"))
                absorption.append(float(f"{result[i]:0.2f}"))
                absorption_percentual.append(float(f"{result[i] * 100:0.0f}"))
            data = {"Bands [Hz]": freq_bands, "Absorption [-]": absorption, "Absorption [%]": absorption_percentual}
            df = pandas.DataFrame(data=data).set_index("Bands [Hz]").T
            df = df.style.set_caption(f"1/{n_oct} Octave Absorption Data")

            try:
                from IPython.display import display
                display(df)
            except:
                print("IPython.display unavailable.")

        return bands, result

    def save2sheet(self, timestamp=False, conversion=None, ext=".xlsx", chart_styles=None, n_oct=3):
        """
        Save current values and layer properties to an Excel spreadsheet.

        Parameters
        ----------
        timestamp : bool, optional
            Boolean to add timestamping to the filename.
        conversion : list or float and string, optional
            List containing conversion ratio and string containing the name of the converted unit.
        ext : string, optional
            Desired file extension.
        chart_styles : list of ints, optional
            List containing indexes of styles to use in the plots inside the exported spreadsheet.
        n_oct : int, optional
            Fractional octave bands that the absorption will be filtered to.
        """

        if chart_styles is None:
            chart_styles = [35, 36]
        if conversion is None:
            conversion = [0.0393701, "[inches]"]
        timestr = time.strftime("%Y%m%d-%H%M_")
        folder_check = os.path.exists(self.project_folder + os.sep + "Treatments")
        if folder_check is False:
            os.mkdir(self.project_folder + os.sep + "Treatments")
        full_path = self.project_folder + os.sep + "Treatments" + os.sep + self.filename + ext
        if timestamp is True:
            full_path = self.project_folder + os.sep + "Treatments" + os.sep + timestr + self.filename + ext

        if ext == ".xlsx":
            workbook = xlsxwriter.Workbook(full_path)
            worksheet = workbook.add_worksheet()

            # Setting formats
            bold = workbook.add_format({"bold": True, "font_color": "black", "align": "center", "border": 2})
            regular = workbook.add_format({"bold": False, "font_color": "black", "align": "center", "border": 1})
            regular_left_bold = workbook.add_format({"bold": True, "font_color": "black", "align": "right", "border": 1,
                                                     })
            regular_left = workbook.add_format({"bold": False, "font_color": "black", "align": "left", "border": 1,
                                                })

            # Adding frequency related data
            worksheet.write(0, 0, "Frequency", bold)
            worksheet.write(0, 1, "Real Z", bold)
            worksheet.write(0, 2, "Img Z", bold)
            worksheet.write(0, 3, "Absorption", bold)
            for i in range(len(self.freq)):
                worksheet.write(1 + i, 0, self.freq[i], regular)
                worksheet.write(1 + i, 1, np.real(self.z_norm[i]), regular)
                worksheet.write(1 + i, 2, np.imag(self.z_norm[i]), regular)
                worksheet.write(1 + i, 3, self.alpha[i], regular)

            # Absorption coefficient plot
            chart_abs = workbook.add_chart({"type": "line"})
            chart_abs.add_series({"name": ["Sheet1", 0, 3],
                                  "categories": ["Sheet1", 1, 0, len(self.freq) + 1, 0],
                                  "values": ["Sheet1", 1, 3, len(self.freq) + 1, 3], })
            chart_abs.set_title({"name": "Absorption Coefficient"})
            chart_abs.set_x_axis({"name": "Frequency [Hz]"})
            chart_abs.set_y_axis({"name": "Alpha [-]"})
            chart_abs.set_style(chart_styles[0])
            worksheet.insert_chart("G1", chart_abs, {"x_offset": 0, "y_offset": 0, "x_scale": 1.334, "y_scale": 1.11})

            # Impedance plot
            chart_z = workbook.add_chart({"type": "line"})
            chart_z.add_series({"name": ["Sheet1", 0, 1],
                                "categories": ["Sheet1", 1, 0, len(self.freq) + 1, 0],
                                "values": ["Sheet1", 1, 1, len(self.freq) + 1, 1], })
            chart_z.add_series({"name": ["Sheet1", 0, 2],
                                "categories": ["Sheet1", 1, 0, len(self.freq) + 1, 0],
                                "values": ["Sheet1", 1, 2, len(self.freq) + 1, 2], })
            chart_z.set_title({"name": "Normalized Surface Impedance"})
            chart_z.set_x_axis({"name": "Frequency [Hz]"})
            chart_z.set_y_axis({"name": "Z [Pa*s/m]"})
            chart_z.set_style(chart_styles[1])
            worksheet.insert_chart("G17", chart_z, {"x_offset": 0, "y_offset": 0, "x_scale": 1.334, "y_scale": 1.11})

            # Adding n_oct band absorption coeffiecients
            line = 0
            idx = 4
            worksheet.merge_range(line, idx, line, idx + 1, f"1/{n_oct} octave band absorption coefficients", bold)
            line += 1
            worksheet.write(line, idx, "Frequency Band [Hz]", bold)
            worksheet.write(line, idx + 1, "Absorption Coeffiecient [-]", bold)
            line += 1
            xOct, yOct = self.filter_alpha(n_oct=n_oct, view=False)
            for x, y in zip(xOct, yOct):
                worksheet.write(line, idx, x, regular)
                worksheet.write(line, idx + 1, y, regular)
                line += 1
            if "material_model" not in self.filename:
                # Adding device properties
                total_depth = 0
                worksheet.merge_range(line, idx, line, idx + 1, "Device Properties", bold)
                line += 1
                worksheet.write(line, idx, "(1 - Front face)", regular)
                worksheet.write(line, idx + 1, f"({len(self.matrix)} - Rear face)", regular)
                line += 1
                worksheet.write(line, idx, "Sound incidence:", regular_left_bold)
                worksheet.write(line, idx + 1, self.incidence, regular_left)
                line += 1
                worksheet.write(line, idx, "Angle [°]:", regular_left_bold)
                if self.incidence == "diffuse":
                    worksheet.write(line, idx + 1,
                                    f"{min(self.incidence_angle):0.0f} - {max(self.incidence_angle):0.0f}",
                                    regular_left)
                else:
                    worksheet.write(line, idx + 1,
                                    f"{(self.incidence_angle[0]):0.0f}",
                                    regular_left)
                line -= 1
                for i in range(1, len(self.matrix) + 1):
                    if i > 1:
                        line -= 1
                    worksheet.merge_range(1 + i + line, idx, 1 + i + line, idx + 1, f"Layer {i}", bold)
                    line += 1
                    for key, value in self.matrix[i - 1].items():
                        if key != "matrix":
                            if isinstance(value, str) or isinstance(value, bool):
                                worksheet.write(1 + i + line, idx, f"{key}:", regular_left_bold)
                                worksheet.write(1 + i + line, idx + 1, f"{value}", regular_left)
                                line += 1
                            else:
                                if "[mm]" in key:
                                    converted = key.replace("[mm]", conversion[1])
                                    worksheet.write(1 + i + line, idx, f"{key}:", regular_left_bold)
                                    worksheet.write(1 + i + line, idx + 1, value, regular_left)
                                    line += 1
                                    worksheet.write(1 + i + line, idx, f"{converted}:", regular_left_bold)
                                    worksheet.write(1 + i + line, idx + 1, value * conversion[0], regular_left)
                                    line += 1
                                else:
                                    worksheet.write(1 + i + line, idx, f"{key}:", regular_left_bold)
                                    worksheet.write(1 + i + line, idx + 1, value, regular_left)
                                    line += 1
                                if "thickness" in key:
                                    total_depth += value

                worksheet.merge_range(1 + i + line, idx, 1 + i + line, idx + 1, "Total", bold)
                line += 1
                worksheet.write(1 + i + line, idx, f"total treatment depth [mm]:", regular_left_bold)
                worksheet.write(1 + i + line, idx + 1, total_depth, regular_left)
                line += 1
                worksheet.write(1 + i + line, idx, f"total treatment depth {conversion[1]}:", regular_left_bold)
                worksheet.write(1 + i + line, idx + 1, total_depth * conversion[0], regular_left)
                line += 1

            # Setting column widths
            worksheet.set_column("A:D", 12)
            worksheet.set_column("E:F", 28)

            workbook.close()

        elif ext == ".csv":
            df1 = pandas.DataFrame()
            df1["Frequency"] = self.freq
            df1["Real Z"] = np.real(self.z_norm)
            df1["Imag Z"] = np.imag(self.z_norm)
            df1["Absorption"] = self.alpha
            df2 = pandas.DataFrame()
            df2["Bands"], df2[f"1/{n_oct} octave band absorption coefficients"] = self.filter_alpha(n_oct=n_oct,
                                                                                                    view=False)
            df3 = pandas.concat([df1, df2], axis=1)
            df3.to_csv(full_path, index=False, float_format="%.3f", sep=";")

        else:
            raise NameError("Unidentified extension. Available extensions: ['.xlsx', '.csv']")

        print(f"Sheet saved to ", full_path)

    def save(self):
        """Saves TMM into HDF5 file."""
        folder_check = os.path.exists(self.project_folder + os.sep + "Treatments")
        if folder_check is False:
            os.mkdir(self.project_folder + os.sep + "Treatments")

        self.reduce_size()

        h5utils.save_class_to_hdf5(self, filename=self.filename,
                                   folder=self.project_folder + os.sep + "Treatments" + os.sep)
        print("HDF5 file saved at " + self.project_folder + os.sep + "Treatments" + os.sep + self.filename + ".h5")

    def load(self, filename):
        """
        Loads TMM data from HDF5 file.

        Parameters
        ----------
        filename : string
            Input filename.
        """
        folder_check = os.path.exists(self.project_folder + os.sep + "Treatments")
        if self.project_folder:
            if folder_check is False:
                h5utils.load_class_from_hdf5(self, filename, folder=self.project_folder + os.sep)
            else:
                h5utils.load_class_from_hdf5(self, filename,
                                             folder=self.project_folder + os.sep + "Treatments" + os.sep)
        else:
            h5utils.load_class_from_hdf5(self, filename)
        self.rebuild()
        print(filename + ".h5 loaded successfully.")

    def plot(self, show_fig=True, **kwargs):
        """View acoustic data. See tmm._plot.acoustic data for kwargs."""
        if "filename" not in kwargs:
            kwargs["filename"] = self.filename
        if "project_folder" not in kwargs:
            kwargs["project_folder"] = self.project_folder
        _, _, _ = plot.acoustic_data([self], **kwargs)
        if show_fig:
            plt.show()

    def view(self, show_fig=True, **kwargs):
        """View 3D treatment model and performance. See tmm._vis.view_layers and tmm._vis.view_treatment for kwargs."""
        from tmm import _vis as vis
        if "template" not in kwargs:
            kwargs["template"] = "seaborn"
        if "height" not in kwargs:
            kwargs["height"] = 400
        if "width" not in kwargs:
            kwargs["width"] = 700
        if "transparent_bg" not in kwargs:
            kwargs["transparent_bg"] = False

        fig1 = vis.view_layers(self, **kwargs)
        fig2 = vis.view_treatment(self, title="<b>Absorption Coefficient</b>", **kwargs)

        if show_fig:
            fig1.show()
            fig2.show()

        return fig1, fig2
