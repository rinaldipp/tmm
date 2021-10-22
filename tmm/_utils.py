""""
Utility functions.

For further information check the function specific documentation.
"""
import numpy as np


class AirProperties:
    """
    Computes properties of humid air.
    """
    def __init__(self, t0=20.0, rh=30.0, p0=101325.0):
        """
        Parameters
        ----------
        c0 : float, optional
            Speed of sound [m/s] in air
        rho0 : float, optional
            Air volume density [kg/m³]
        t0 : float or int, optional
            Temperature in Celsius [C]
        rh : float, optional
            Relative humidity in percentage [%]
        p0 : float, optional
            Atmospheric pressure in Pascal [Pa]
        """
        self.t0 = np.array(t0, dtype=np.float32)
        self.rh = np.array(rh, dtype=np.float32)
        self.p0 = np.array(p0, dtype=np.float32)

    @property
    def temp_kelvin(self):
        """Returns temperature in Kelvins."""
        return self.t0 + 273.16

    @property
    def rho0(self):
        """Return air density."""
        return self.standardized_c0_rho0()["air_density"]

    @property
    def c0(self):
        """Return speed of sound in air."""
        return self.standardized_c0_rho0()["speed_of_sound"]

    @property
    def z0(self):
        """Return characteristic impedance of air."""
        return self.rho0 * self.c0

    @property
    def viscosity(self):
        """Return air viscosity."""
        return self.standardized_c0_rho0()["air_viscosity"]

    def standardized_c0_rho0(self):
        """
        This method is used to calculate the standardized value of the sound speed and air density based on measurements
        of temperature, humidity and atmospheric pressure. It will overwrite the user supplied values.
        """
        # Constants
        r = 287.031  # Gas constant for air [J/K/kg]
        rvp = 461.521  # Gas constant for water vapor [J/K/kg]
        kappla = 0.026  # Air thermal conductivity [W/m*k]
        # pvp from Pierce Acoustics 1955 - pag. 555
        pvp = (0.0658 * self.temp_kelvin ** 3 - 53.7558 * self.temp_kelvin ** 2
               + 14703.8127 * self.temp_kelvin - 1345485.0465)
        # Constant pressure specific heat
        cp = 4168.8 * (0.249679 - 7.55179e-5 * self.temp_kelvin + 1.69194e-7 * self.temp_kelvin ** 2
                       - 6.46128e-11 * self.temp_kelvin ** 3)
        # Constant volume specific heat [J/kg/K] for 260 K < T < 600 K
        cv = cp - r
        # Specific heat constant ratio
        gam = cp / cv
        # Air viscosity
        viscosity = (7.72488e-8 * self.temp_kelvin - 5.95238e-11 * self.temp_kelvin ** 2
                          + 2.71368e-14 * self.temp_kelvin ** 3)
        # Prandtl number (fewly varies at typical air conditions (0°C=0.715; 60°C=0.709)
        pn = viscosity * cp / kappla
        # Air density [kg/m³]
        rho0 = self.p0 / (r * self.temp_kelvin) - (1 / r - 1 / rvp) * self.rh / 100 * pvp / self.temp_kelvin
        # Air sound speed [m/s]
        c0 = (gam * self.p0 / rho0) ** 0.5

        prop_dict = {"temperature_in_celsius": self.t0,
                     "relative_humidity": self.rh,
                     "atmospheric_pressure": self.p0,
                     "air_density": rho0,
                     "speed_of_sound": c0,
                     "air_viscosity": viscosity,
                     "prandtl_number": pn,
                     "specific_heat_ratio": gam,
                     "air_thermal_conductivity": kappla,
                     "constant_pressure_specific_heat": cp}
        return prop_dict

    def air_absorption(self, freq):
        """
        Calculates the air absorption coefficient in [m^-1].

        Parameters
        ----------
        freq : array
            Array of frequencies.

        Returns
        ----------
        Array of air absorption values.
        """
        t_0 = 293.15  # Reference temperature [k]
        t_01 = 273.15  # 0 [C] in [k]
        patm_atm = self.p0 / 101325  # Atmospheric pressure [atm]
        f = freq / patm_atm  # Relative frequency
        # Saturation pressure
        psat = patm_atm * 10 ** (-6.8346 * (t_01 / self.temp_kelvin) ** 1.261 + 4.6151)
        h = patm_atm * self.rh * (psat / patm_atm)
        # Oxygen gas molecule (N2) relaxation frequency
        f_rO = 1/patm_atm * (24 + 4.04 * 10**4 * h * (0.02 + h) / (0.391 + h))
        # Nytrogen gas molecule (N2) relaxation frequency
        f_rn = (1 / patm_atm * (t_0 / self.temp_kelvin) ** (1 / 2)
                * (9 + 280 * h * np.exp(-4.17 * ((t_0 / self.temp_kelvin) ** (1 / 3) - 1))))
        # Air absorption in [dB/m]
        alpha_ps = (100 * f ** 2 / patm_atm * (1.84 * 10 ** (-11) * (self.temp_kelvin / t_0) ** (1 / 2)
                                               + (self.temp_kelvin / t_0) ** (-5 / 2)
                                               * (0.01278 * np.exp(-2239.1 / self.temp_kelvin) / (f_rO + f ** 2 / f_rO)
                                                  + 0.1068 * np.exp(-3352 / self.temp_kelvin) / (f_rn + f**2 / f_rn))))
        a_ps_ar = alpha_ps * 20 / np.log(10)
        # Air absorption in [1/m]
        m = (1/100) * a_ps_ar * patm_atm / (10 * np.log10(np.exp(1)))
        return m


def find_nearest(array, value):
    """
    Function to find closest frequency in frequency array. Returns closest value and position index.

    Parameters
    ----------
    array : array
        Array in which to search the closest value.
    value : int or float
        Value to be searched.

    Returns
    -------
    Closest value found in the array and index of the closest value.
    """
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    value = array[idx]

    return value, idx
