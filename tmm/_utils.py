""""
Utility functions.

For further information check the function specific documentation.
"""
import numpy as np
import scipy

# Nominal frequencies to calculate octave bands with the 1kHz band as reference using bases 10 or 2
base_nominal_frequencies = np.array([
    0.1, 0.125, 0.16, 0.2, 0.25, 0.315, 0.4, 0.5, 0.6, 3., 0.8,
    1., 1.25, 1.6, 2., 2.5, 3.15, 4., 5., 6.3, 8., 10., 12.5, 16.,
    20., 25., 31.5, 40., 50., 63., 80., 100., 125., 160., 200., 250.,
    315., 400., 500., 630., 800., 1000., 1250., 1600., 2000., 2500.,
    3150., 4000., 5000., 6300., 8000., 10000., 12500., 16000., 20000.
])

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


def filter_values(freq, values, n_oct=1):
    """
    Filters the given values into n_oct bands.

    Parameters
    ----------
    freq : ndarray
        Array containing the frequency axis.
    values : ndarray
        Array containing the magnitude values to be filtered.
    n_oct : int, optional
        Fractional octave bands that the absorption will be filtered to.

    Returns
    -------
    bands : ndarray
        An array containing the center frequencies of the available bands.
    result : ndarray
        An array containing the filtered values in the available bands.
    """
    bands = nth_octave(n_oct, fmin=min(freq), fmax=max(freq))
    df_bands = bands[1][1] - bands[0][1]
    df_freq = freq[1] - freq[0]
    if df_bands < df_freq:
        interp_fun = scipy.interpolate.interp1d(freq, values)
        freq = np.linspace(min(freq), max(freq), int((max(freq) - min(freq) - 1) / df_bands))
        values = interp_fun(freq)
        
    edges = np.delete(bands, 1, axis=1)
    lidx = np.searchsorted(freq, edges[:, 0], 'right')
    ridx = np.searchsorted(freq, edges[:, 1], 'left')
    result = np.array([np.sum(values[i:j]) / len(values[i:j]) for (i, j) in zip(lidx, ridx) if values[i:j].size > 0])

    return bands[:, 1], result.astype(values.dtype)


def nth_octave(fraction, fmin=20, fmax=20000):
    """ ANSI s1.11-2004 && IEC 61260-1-2014
    Array of frequencies and its edges according to the ANSI and IEC standard.
    :param fraction: Bandwidth 'b'. Examples: 1/3-octave b=3, 1-octave b=1,
    2/3-octave b = 3/2
    :param limits: It is a list with the minimum and maximum frequency that
    the array should have. Example: [12,20000]
    :returns: Frequency array, lower edge array and upper edge array
    :rtype: list, list, list
    """
    limits = [fmin, fmax]

    # Octave ratio g (ANSI s1.11, 3.2, pg. 2)
    g = 10 ** (3 / 10)  # Or g = 2
    # Reference frequency (ANSI s1.11, 3.4, pg. 2)
    fr = 1000

    # Get starting index 'x' and first center frequency
    x = initindex(limits[0], fr, g, fraction)
    freq = ratio(g, x, fraction) * fr
    freq = find_nearest(base_nominal_frequencies, freq)[0]

    # Get each frequency until reach maximum frequency
    freq_x = 0
    while freq_x * bandedge(g, fraction) < limits[1]:
        # Increase index
        x = x + 1
        # New frequency
        freq_x = ratio(g, x, fraction) * fr
        freq_x = find_nearest(base_nominal_frequencies, freq_x)[0]

        # Store new frequency
        freq = np.append(freq, freq_x)

    # Get band-edges
    freq_d = freq / bandedge(g, fraction)
    freq_u = freq * bandedge(g, fraction)

    return np.array([freq_d, freq, freq_u]).T


def band_edges(freq, fraction):
    """ ANSI s1.11-2004 && IEC 61260-1-2014
    Frequency band edge values according to the ANSI and IEC standard.
    """
    # Octave ratio g (ANSI s1.11, 3.2, pg. 2)
    g = 10 ** (3 / 10)  # Or g = 2

    # Get band-edges
    freq_d = freq / bandedge(g, fraction)
    freq_u = freq * bandedge(g, fraction)

    return np.array([freq_d, freq, freq_u]).T


def ratio(g, x, b):
    if b % 2:  # ODD (ANSI s1.11, eq. 3)
        return g ** ((x - 30) / b)
    else:  # EVEN (ANSI s1.11, eq. 4)
        return g ** ((2 * x - 59) / (2 * b))


def bandedge(g, b):
    # Band-edge ratio (ANSI s1.11, 3.7, pg. 3)
    return g ** (1 / (2 * b))


def initindex(f, fr, g, b):
    if b % 2:  # ODD ('x' solve from ANSI s1.11, eq. 3)
        return np.round(
                (b * np.log(f / fr) + 30 * np.log(g)) / np.log(g)
                )
    else:  # EVEN ('x' solve from ANSI s1.11, eq. 4)
        return np.round(
                (2 * b * np.log(f / fr) + 59 * np.log(g)) / (2 * np.log(g))
                )



