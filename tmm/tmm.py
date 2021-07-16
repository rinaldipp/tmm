import collections
import numpy as np
import numpy.matlib as matlib
import os
from matplotlib import ticker, gridspec, style, rcParams
from matplotlib import pyplot as plt
from matplotlib import style
import pandas
import xlsxwriter
from scipy.special import jv
import time
from scipy import integrate
import pytta
from tmm import _h5utils as h5utils

outputs = os.getcwd() + os.sep
style.use("seaborn-colorblind")


class TMM:
    """"
    Transfer Matrix Method for design and prediction of multilayered acoustic treatments.
    """
    def __init__(self, fmin=20, fmax=5000, df=0.5, incidence="diffuse", incidence_angle=[0, 78, 1],
                 project_folder=None):
        """
        Initialization method.

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
        """
        self.fmin = fmin
        self.fmax = fmax
        self.df = df
        self.z = np.zeros_like(self.freq, dtype="complex")
        self.z_normal = None  # Only used for plotting when incidence == "diffuse"
        self.s0 = 1  # Device front termination area
        self.srad = 1  # Device rear termination area
        self.matrix = {}
        self.air_prop = air_properties()
        self.project_folder = project_folder
        self.incidence = incidence
        if self.incidence == "diffuse":
            self.incidence_angle = np.linspace(incidence_angle[0] + 0.01, incidence_angle[1] - 0.01,
                                               int((incidence_angle[1] - incidence_angle[0]) / incidence_angle[2]))
        elif self.incidence == "normal":
            self.incidence_angle = np.linspace(0, 1, 1)
        elif self.incidence == "angle":
            self.incidence_angle = np.linspace(incidence_angle[0], incidence_angle[0] + 1, 1)

    @property
    def freq(self):
        return np.linspace(self.fmin, self.fmax, int((self.fmax - self.fmin) / self.df) + 1)

    @property
    def rho0(self):
        return self.air_prop["air_density"]

    @property
    def c0(self):
        return self.air_prop["speed_of_sound"]

    @property
    def w0(self):
        return 2 * np.pi * self.freq

    @property
    def k0(self):
        return self.w0 / self.c0

    @property
    def z0(self):
        return self.rho0 * self.c0

    @property
    def z_norm(self):
        return self.z / self.z0

    @property
    def y(self):
        return 1 / self.z

    @property
    def y_norm(self):
        return 1 / self.z_norm

    @property
    def alpha(self):
        R, alpha = self.reflection_and_absorption_coefficient(self.z)
        return alpha.reshape((len(alpha),))

    @property
    def alpha_normal(self):
        R, alpha = self.reflection_and_absorption_coefficient(self.z_normal)
        return alpha

    def reflection_and_absorption_coefficient(self, zs):
        """
        Calculate reflection coefficient and absorption coefficient for a given surface impedance.

        Parameters
        ----------
        zs : array
            Surface impedance.
        Returns
        -------
        R : array
            Reflection coefficient.
        alpha : array
            Absorption coefficient.
        """
        R = (zs - self.z0) / (zs + self.z0)
        alpha = 1 - np.abs(R) ** 2

        return R, alpha

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
        kc : ndarray
            Array containing the propagation constant inside the porous material.
        zc : ndarray
            Array containing the characteristic acoustic impedance the porous material.
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

        Returns
        -------
        Nothing.
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

        Returns
        -------
        Nothing.
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

        Returns
        -------
        Nothing.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters

        k0_angle = matlib.repmat(self.k0, len(self.incidence_angle), 1).T
        w0_angle = matlib.repmat(self.w0, len(self.incidence_angle), 1).T
        kt = k0_angle * np.sin(np.deg2rad(self.incidence_angle))

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

    def perforated_panel_layer(self, t=19, d=8, s=16, end_correction="jb", method="barrier", layer=None):
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
        end_correction : string, optional
            Chooses between the available end corrections for the tube length.
        method : string, optional
            Chooses between the available methods to calculate the perforated plate impedance.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.

        Returns
        -------
        Nothing.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        d_meters = d / 1000
        s_meters = s / 1000

        if d < 2 / s:
            print(f"WARNING: Hole spacing too small for {d} [mm] hole diameter.")

        open_area = np.pi / (s_meters / (d_meters / 2)) ** 2

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

        if method == "barrier":
            """
            Impedance from section 7.3.1 of  Acoustic Absorbers and Diffusers 
            by Trevor Cox and Peter D'Antonio (3rd Edition).
            """
            rm = (self.rho0 / open_area) * np.sqrt(8 * vis * self.w0) * (
                        1 + t_meters / (d_meters))  # Surface resistance
            zpp = (1j / open_area) * t_corr * self.w0 * self.rho0 + rm  # Impedance of perforated plate
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
            rm = 32 * vis / (open_area) * t_meters / d_meters ** 2 * kr
            km = 1 + 1 / np.sqrt(1 + cis ** 2 / 2) + 0.85 * d_meters / t_meters
            m = self.rho0 * t_meters / (open_area) * km

            zpp = rm + 1j * self.w0 * m
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
                              "matrix": Tp,
                              }

    def slotted_panel_layer(self, t=19, w=8, s=16, method="barrier", layer=None):
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
        method : string, optional
            Chooses between the available methods to calculate the perforated plate impedance.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.

        Returns
        -------
        Nothing.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        w_meters = w / 1000
        s_meters = s / 1000

        open_area = w_meters / s_meters

        t_corr = t_meters + 2 * w_meters * (-1 / np.pi) * np.log(np.sin(0.5 * np.pi * open_area))

        if method == "barrier":
            """
            Impedance from "On the design of resonant absorbers using a slotted plate" by Kristiansen and Vigran.
            """
            vis = self.air_prop["air_viscosity"]
            Rp = 0.5 * np.sqrt(2 * vis * self.rho0 * self.w0) * (4 + (2 * t) / w)
            Xp = self.rho0 * t_corr
            zs = (Rp + 1j * self.w0 *  Xp) / open_area
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
        kc : ndarray
            Array containing the propagation constant inside the perforations.
        zc : ndarray
            Array containing the characteristic acoustic impedance the perforations.
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
        kc : ndarray
            Array containing the propagation constant inside the slits.
        zc : ndarray
            Array containing the characteristic acoustic impedance the slits.
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

    def field_impedance(self, z):
        """
        Calculates the field impedance for a set of angle dependent impedances.

        Parameters
        ----------
        z : array
            Multidimensional array with angle dependent set of impedances.

        Returns
        -------
        z_field : array
            Field impedance.
        """
        A = 1 / z

        self.z_normal = z[:, 0]

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

        Returns
        -------
        Nothing.
        """
        self.matrix = dict(collections.OrderedDict(sorted(self.matrix.items())))

        if "rigid_backing" in self.matrix[list(self.matrix.keys())[-1]]:
            self.matrix.pop(list(self.matrix.keys())[-1])

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
        zc = self.s0 * (Ag + (Bg * zrad / self.srad)) / (Cg + (Dg * zrad / self.srad))

        if self.incidence == "diffuse":
            zc = self.field_impedance(zc)

        if not conj:
            self.z = zc
        else:
            self.z = np.conj(zc)

        self.matrix[len(self.matrix)] = {"rigid_backing": rigid_backing,
                                         "impedance_conjugate": conj}

        if show_layers:
            self.show_layers()

    def show_layers(self, conversion=[0.0393701, "[inches]"]):
        """
        Method to print each layer with its details.

        Parameters
        ----------
        conversion : list or float and string, optional
            List containing conversion ratio and string containing the name of the converted unit.

        Returns
        -------
        Nothing.
        """
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
                        else:
                            print(f"\t{key}: {value:0.2f}")
                        if "thickness" in key:
                            total_depth += value
        print(f"\nTotal treatment depth [mm]: {total_depth:0.2f} | " +
              f"Total treatment depth {conversion[1]}: {total_depth * conversion[0]:0.2f}")

    def filter_alpha(self, nthOct=1, plot=True, returnValues=False, show=False, figsize=(15, 5)):
        """
        Filters the absorption coefficient into nthOct bands.

        Parameters
        ----------
        nthOct : int, optional
            Fractional octave bands that the absorption will be filtered to.
        plot : bool, optional
            Boolean to display plot with filtered absorption.
        returnValues : bool, optional
            Boolean to return the bands and filetered values.
        show : bool, optional
            Boolean to display the filtered values in a table.
        figsize : tuple, optional
            Tuple containing the width and height of the figure.

        Returns
        -------
        bands : ndarray
            An array containing the center frequencies of the available bands.
        result : ndarray
            An array containing the filtered absorption coefficient in the available bands.
        """
        bands, result = pytta.utils.filter_values(self.freq, self.alpha, nthOct=nthOct)

        # Plot
        if plot:
            fig, ax1 = plt.subplots(figsize=figsize)
            ax1.semilogx(self.freq, self.alpha, label="Narrowband")
            ax2 = ax1.twiny()
            ax2.set_xscale("log")
            ax1.semilogx(bands, result, "o-", label=f"1/{nthOct} octave band")
            ax2.set_xticks([freq for freq in bands.tolist()])
            ax2.set_xticklabels([f"{freq:0.1f}" for freq in bands.tolist()])
            ax2.set_xlim(ax1.get_xlim())
            ax1.set_ylabel("Absorption Coefficient [-]")
            ax1.set_xlabel("Narrowband Frequency [Hz]")
            ax2.set_xlabel(f"1/{nthOct} Octave Bands [Hz]")
            ax1.set_ylim([-0.1, 1.1])
            ax1.legend(loc="best")
            ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax1.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax1.tick_params(which="minor", length=5, rotation=-90,
                            axis="x")  # Set major and minor ticks to same length
            ax1.tick_params(which="major", length=5, rotation=-90,
                            axis="x")  # Set major and minor ticks to same length
            ax2.tick_params(which="major", length=5, rotation=-90,
                            axis="x")  # Set major and minor ticks to same length
            ax1.minorticks_on()  # Set major and minor ticks to same length
            ax2.minorticks_off()
            ax1.grid("minor")
            plt.show()

        if show:
            pandas.set_option("display.precision", 2)
            freq_bands = []
            absorption = []
            absorption_percentual = []
            #             for key, value in available_data.items():
            for i in range(len(bands)):
                freq_bands.append(float(f"{bands[i]:0.2f}"))
                absorption.append(float(f"{result[i]:0.2f}"))
                absorption_percentual.append(float(f"{result[i] * 100:0.0f}"))
            data = {"Bands [Hz]": freq_bands, "Absorption [-]": absorption, "Absorption [%]": absorption_percentual}
            df = pandas.DataFrame(data=data).set_index("Bands [Hz]").T
            df = df.style.set_caption(f"1/{nthOct} Octave Absorption Data")

            try:
                from IPython.display import display
                display(df)
            except:
                print("IPython.diplay unavailable.")

        if returnValues:
            return bands, result

    def save2sheet(self, filename="TMM", timestamp=True, conversion=[0.0393701, "[inches]"],
                   ext=".xlsx", chart_styles=[35, 36], nthOct=3):
        """
        Save current values and layer properties to an Excel spreadsheet.

        Parameters
        ----------
        filename : strint, optional
            Name to be added in the filename.
        timestamp : bool, optional
            Boolean to add timestamping to the filename.
        conversion : list or float and string, optional
            List containing conversion ratio and string containing the name of the converted unit.
        ext : string, optional
            Desired file extension.
        chart_styles : list of ints, optional
            List containing indexes of styles to use in the plots inside the exported spreadsheet.
        nthOct : int, optional
            Fractional octave bands that the absorption will be filtered to.

        Returns
        -------
        Nothing.
        """

        timestr = time.strftime("%Y%m%d-%H%M_")
        if self.project_folder is None:
            full_path = outputs + filename + ext
            if timestamp is True:
                full_path = outputs + timestr + filename + ext
        else:
            folderCheck = os.path.exists(self.project_folder + os.sep + "Treatments")
            if folderCheck is False:
                os.mkdir(self.project_folder + os.sep + "Treatments")
            full_path = self.project_folder + os.sep + "Treatments" + os.sep + filename + ext
            if timestamp is True:
                full_path = self.project_folder + os.sep + "Treatments" + os.sep + timestr + filename + ext

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

            # Adding nthOct band absorption coeffiecients
            line = 0
            idx = 4
            worksheet.merge_range(line, idx, line, idx + 1, f"1/{nthOct} octave band absorption coefficients", bold)
            line += 1
            worksheet.write(line, idx, "Frequency Band [Hz]", bold)
            worksheet.write(line, idx + 1, "Absorption Coeffiecient [-]", bold)
            line += 1
            xOct, yOct = self.filter_alpha(nthOct=nthOct, plot=False, returnValues=True)
            for x, y in zip(xOct, yOct):
                worksheet.write(line, idx, x, regular)
                worksheet.write(line, idx + 1, y, regular)
                line += 1

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
            df2["Bands"], df2[f"1/{nthOct} octave band absorption coefficients"] = self.filter_alpha(nthOct=nthOct,
                                                                                                     returnValues=True,
                                                                                                     plot=False)
            df3 = pandas.concat([df1, df2], axis=1)
            df3.to_csv(full_path, index=False, float_format="%.3f", sep=";")

        else:
            raise NameError("Unidentified extension. Available extensions: ['.xlsx', '.csv']")

        print(f"Sheet saved to ", full_path)

    def save(self, filename="tmm"):
        """
        Saves TMM into HDF5 file.

        Parameters
        ----------
        filename : string
            Output filename.

        Returns
        -------
        Nothing.
        """
        if self.project_folder:
            h5utils.save_class_to_hdf5(self, filename=filename, folder=self.project_folder)
            print("HDF5 file saved at " + self.project_folder + filename + ".h5")
        else:
            h5utils.save_class_to_hdf5(self, filename=filename, folder=outputs)
            print("HDF5 file saved at " + filename + ".h5")

    def load(self, filename):
        """
        Loads TMM data from HDF5 file.

        Parameters
        ----------
        filename : string
            Input filename.

        Returns
        -------
        Nothing.
        """
        if self.project_folder:
            h5utils.load_class_from_hdf5(self, filename, folder=self.project_folder)
            print(self.project_folder + filename + ".h5 loaded successfully.")
        else:
            h5utils.load_class_from_hdf5(self, filename)
            print(filename + ".h5 loaded successfully.")

    def plot(self, figsize=(15, 5), plots=["z", "y", "alpha"], saveFig=False, filename="TMM", timestamp=True,
             ext=".png", max_mode="all"):
        """
        Plots impedance, admittance and absorption curves.

        Parameters
        ----------
        figsize : tuple, optional
            Tuple containing the width and height of the figure.
        plots : list of strings, optional
            Desired curves to be plotted. 'z' for impedance, 'y' for admittance and 'alpha' for absorption.
        saveFig : bool, optional
            Option to save the plot as an image.
        filename : strint, optional
            Name to be added in the filename:
        timestamp : bool, optional
            Boolean to add timestamping to the filename.
        ext : string, optional
            Desired file extension.
        max_mode : None, int or string
            Variable to set a maximum limit to peak detection in the absorption coefficient.
            'all' for no limit, None for  no detection or int for maximum detection frequency.

        Returns
        -------
        Nothing.
        """

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, len(plots))

        i = 0
        if "z" in plots or "Z" in plots:
            ax_z = plt.subplot(gs[0, i])
            ax_z.set_title(r"Impedance ($Z$)")
            ax_z.set_xlabel("Frequency [Hz]")
            ax_z.set_ylabel("Normalized Surface Impedance [Z/Z0]")
            ax_z.semilogx(self.freq, np.real(self.z_norm), linewidth=2, label="Real")
            ax_z.semilogx(self.freq, np.imag(self.z_norm), linewidth=2, label="Imag")
            ax_z.set_xlim([(np.min(self.freq)), (np.max(self.freq))])
            ax_z.axhline(y=0, color="k", linewidth=0.5)
            ax_z.axhline(y=1, linestyle="--", color="gray")
            ax_z.legend(loc="best")
            ax_z.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_z.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_z.tick_params(which="minor", length=5, rotation=-90,
                             axis="x")  # Set major and minor ticks to same length
            ax_z.tick_params(which="major", length=5, rotation=-90,
                             axis="x")  # Set major and minor ticks to same length
            ax_z.minorticks_on()  # Set major and minor ticks to same length
            ax_z.grid("minor")
            i += 1

        if "y" in plots or "Y" in plots:
            ax_y = plt.subplot(gs[0, i])
            ax_y.set_title(r"Admittance ($Y$)")
            ax_y.set_xlabel("Frequency [Hz]")
            ax_y.set_ylabel("Normalized Surface Admittance [Z0/Z]")
            ax_y.semilogx(self.freq, np.real(self.y_norm), linewidth=2, label="Real")
            ax_y.semilogx(self.freq, np.imag(self.y_norm), linewidth=2, label="Imag")
            ax_y.set_xlim([(np.min(self.freq)), (np.max(self.freq))])
            ax_y.legend(loc="best")
            ax_y.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_y.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_y.tick_params(which="minor", length=5, rotation=-90,
                             axis="x")  # Set major and minor ticks to same length
            ax_y.tick_params(which="major", length=5, rotation=-90,
                             axis="x")  # Set major and minor ticks to same length
            ax_y.minorticks_on()  # Set major and minor ticks to same length
            ax_y.grid("minor")
            i += 1

        if "alpha" in plots or "abs" in plots:
            ax_a = plt.subplot(gs[0, i])
            ax_a.set_title(r"Absorption Coefficient ($\alpha$)")
            ax_a.set_xlabel("Frequency [Hz]")
            ax_a.set_ylabel("Absorption Coefficient [-]")
            if self.incidence == "diffuse":
                ax_a.semilogx(self.freq, self.alpha, linewidth=2,
                              label=f"Diffuse Incidence " +
                                    f"({min(self.incidence_angle):0.0f}° - {max(self.incidence_angle):0.0f}°)")
                ax_a.semilogx(self.freq, self.alpha_normal, linewidth=2, label="Normal Incidence", linestyle=":")
            else:
                ax_a.semilogx(self.freq, self.alpha, linewidth=2)
            if max_mode == "all":
                abs_value, idx = find_nearest(self.alpha, max(self.alpha))
            elif max_mode is not None:
                max_mode_val, idx_max_mode = find_nearest(self.freq, max_mode)
                abs_value, idx = find_nearest(self.alpha[0:idx_max_mode], max(self.alpha[0:idx_max_mode]))
            if max_mode is not None:
                ax_a.axvline(x=self.freq[idx], label=f"Resonance at {self.freq[idx]} Hz", linestyle="--", color="green")
            ax_a.set_xlim([(np.min(self.freq)), (np.max(self.freq))])
            ax_a.set_ylim([-0.1, 1.1])
            ax_a.legend(loc="best")
            ax_a.axhline(y=0, color="k", linewidth=0.5)
            ax_a.axhline(y=1, linestyle="--", color="gray")
            ax_a.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_a.get_xaxis().set_minor_formatter(ticker.ScalarFormatter())  # Remove scientific notation from xaxis
            ax_a.tick_params(which="minor", length=5, rotation=-90,
                             axis="x")  # Set major and minor ticks to same length
            ax_a.tick_params(which="major", length=5, rotation=-90,
                             axis="x")  # Set major and minor ticks to same length
            ax_a.minorticks_on()  # Set major and minor ticks to same length
            ax_a.grid("minor")
            i += 1

        gs.tight_layout(fig, pad=4, w_pad=1, h_pad=1)

        if saveFig:
            timestr = time.strftime("%Y%m%d-%H%M_")
            if self.project_folder is None:
                full_path = outputs + filename + ext
                if timestamp is True:
                    full_path = outputs + timestr + filename + ext
            else:
                folderCheck = os.path.exists(self.project_folder + os.sep + "Treatments")
                if folderCheck is False:
                    os.mkdir(self.project_folder + os.sep + "Treatments")
                full_path = self.project_folder + os.sep + "Treatments" + os.sep + filename + ext
                if timestamp is True:
                    full_path = self.project_folder + os.sep + "Treatments" + os.sep + timestr + filename + ext

            plt.savefig(full_path, dpi=100)
            print("Image saved to ", full_path)
        plt.show()


def air_properties(t0=20, p0=101320, rh=30):
    """
    Computes properties of humid air.

    Parameters
    ----------
    t0 : float or int, optional
        Temperature in Celsius [C]
    p0 : float or int, optional
        Atmospheric pressure in Pascal [Pa]
    rh : float or int, optional
        Relative humidity in percentage [%]

    Returns
    -------
    air_properties : dict
        Dictionary containing the followint air properties:
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

    air_properties = {"temperature_in_celsius": t0,
                      "relative_humidity": rh,
                      "atmospheric_pressure": p0,
                      "prandtl_number": pn,
                      "specific_heat_ratio": gam,
                      "air_density": rho0,
                      "speed_of_sound": c0,
                      "air_viscosity": vis,
                      "air_thermal_conductivity": kappla,
                      "constant_pressure_specific_heat": Cp}

    return air_properties


def find_nearest(array, value):
    """
    Function to find closest frequency in frequency array. Returns closest value and position index.

    Parameters
    ----------
    array : array
        Array in wich to search the closest value.
    value : int or floar
        Value to be searched.

    Returns
    -------
    value : float
        Closest value found in the array.
    idx : int
        Index of the closest value.
    """
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    value = array[idx]

    return value, idx

