""""
Utility functions.

For further information check the function specific documentation.
"""
import numpy as np


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
    Dictionary containing the following air properties:
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

    air_properties_dict = {"temperature_in_celsius": t0,
                           "relative_humidity": rh,
                           "atmospheric_pressure": p0,
                           "prandtl_number": pn,
                           "specific_heat_ratio": gam,
                           "air_density": rho0,
                           "speed_of_sound": c0,
                           "air_viscosity": vis,
                           "air_thermal_conductivity": kappla,
                           "constant_pressure_specific_heat": Cp}

    return air_properties_dict


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
