"""
Utility functions for air properties and for fractional-octave bands per IEC 61260-1 [1] and ANSI S1.11 [2].

References
----------
[1] IEC 61260-1:2014, "Electroacoustics - Octave-band and fractional-octave-band filters - Part 1: Specifications",
    International Electrotechnical Commission, 2014.
[2] ANSI S1.11-2004, "Specification for Octave-Band and Fractional-Octave-Band Analog and Digital Filters",
    Acoustical Society of America, 2004 (reaffirmed 2009).

For further information check the function specific documentation.
"""
import numpy as np
import scipy

# Nominal mid-band frequencies of octave and one-third-octave bands in IEC 61260-1:2014, Table E.1, as hundredths of
# their decade, which clause E.1 extends to any decade by placing the decimal sign. Integers keep the labels exact.
NOMINAL_HUNDREDTHS = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000])

class AirProperties:
    """
    Compute the properties of humid air from temperature, humidity and pressure.
    """
    def __init__(self, t0=20.0, rh=30.0, p0=101325.0):
        """
        Parameters
        ----------
        t0 : float, optional
            Temperature [°C]
        rh : float, optional
            Relative humidity [%]
        p0 : float, optional
            Atmospheric pressure [Pa]
        """
        # Double precision is required here. The saturated vapour pressure below cancels terms of order
        # 1e6 down to order 1e3, which costs roughly three decimal digits, and the remaining properties
        # are derived from it. In single precision that leaves about four significant digits in pvp and
        # makes save/load round trips non-reproducible, because HDF5 scalars are read back as float64.
        self.t0 = np.array(t0, dtype=np.float64)
        self.rh = np.array(rh, dtype=np.float64)
        self.p0 = np.array(p0, dtype=np.float64)

    @property
    def temp_kelvin(self):
        """Return the temperature in kelvin."""
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
        Return the air properties derived from temperature, humidity and pressure.

        The dictionary holds the density, speed of sound, viscosity, Prandtl number, specific heat ratio, thermal
        conductivity and constant-pressure specific heat, plus the three inputs.
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
        Return the air absorption coefficient [1/m] at each frequency.

        Parameters
        ----------
        freq : array
            Frequencies [Hz].

        Returns
        -------
        Array of air absorption values, same length as ``freq``.
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
    Return the array element closest to ``value`` and its index.

    Parameters
    ----------
    array : array
        Values to search.
    value : float
        Target value.

    Returns
    -------
    Closest element and its index.
    """
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    value = array[idx]

    return value, idx


def filter_values(freq, values, n_oct=1):
    """
    Average ``values`` into fractional-octave bands.

    When the first spacing of ``freq`` is wider than the spacing of the first two band centres, the values are first
    interpolated linearly onto the original frequencies together with a geometric grid of eight points per band, so
    every band gets samples and no original sample is lost. ``band_average`` then averages them.

    Parameters
    ----------
    freq : ndarray
        Frequency axis [Hz].
    values : ndarray
        Values to average, same length as ``freq``.
    n_oct : int, optional
        Fractional-octave resolution: ``1`` for octave bands, ``3`` for third-octave bands.

    Returns
    -------
    bands : ndarray
        Nominal mid-band frequencies of the bands of ``nth_octave`` over ``freq`` that hold at least one sample.
    result : ndarray
        Band-averaged values, one per entry of ``bands``.
    """
    freq = np.asarray(freq, dtype=float)
    values = np.asarray(values)
    b = _validated_fraction(n_oct)
    bands = nth_octave(b, fmin=min(freq), fmax=max(freq))
    if len(freq) > 1:
        df_bands = bands[1][1] - bands[0][1] if len(bands) > 1 else bands[0][2] - bands[0][0]
        if df_bands < freq[1] - freq[0]:
            per_band = 8
            count = int(np.ceil(b * np.log(freq[-1] / freq[0]) / np.log(10 ** (3 / 10)) * per_band)) + 1
            fine = np.union1d(freq, np.geomspace(freq[0], freq[-1], max(count, 2)))
            values = scipy.interpolate.interp1d(freq, values)(fine)
            freq = fine
    return band_average(freq, values, b)


def band_average(freq, values, n_oct):
    """
    Average ``values`` over the bands of ``nth_octave`` that cover ``freq``, without interpolation.

    Each sample goes to the band whose index it rounds to, with the tolerance ``nth_octave`` uses to choose its bands,
    so every sample is counted exactly once and a sample on an edge belongs to the upper band. Samples outside the
    bands, which can only be the end samples within rounding, go to the first or last band.

    Parameters
    ----------
    freq : ndarray
        Frequency axis [Hz], positive.
    values : ndarray
        Values to average, same length as ``freq``.
    n_oct : int
        Bandwidth denominator ``b``.

    Returns
    -------
    bands : ndarray
        Nominal mid-band frequencies of the bands that hold at least one sample.
    result : ndarray
        Band-averaged values, one per entry of ``bands``, as float, or complex for complex ``values``.
    """
    freq = np.asarray(freq, dtype=float)
    values = np.asarray(values)
    b = _validated_fraction(n_oct)
    first, last = _band_range(b, float(np.min(freq)), float(np.max(freq)))
    bands = nth_octave(b, float(np.min(freq)), float(np.max(freq)))
    member = np.clip(np.floor(_band_index(freq, b) + 0.5 + 1e-9).astype(int), first, last) - first
    counts = np.bincount(member, minlength=len(bands))
    sums = np.bincount(member, weights=values.real.astype(float), minlength=len(bands))
    if np.iscomplexobj(values):
        sums = sums + 1j * np.bincount(member, weights=values.imag.astype(float), minlength=len(bands))
    filled = counts > 0
    # Means are float (complex for complex input) whatever the input type, so integer values are not truncated.
    return bands[filled, 1], sums[filled] / counts[filled]


def _band_range(b, fmin, fmax):
    """Return the first and last band index of ``nth_octave`` for ``fmin`` and ``fmax``."""
    first = int(np.floor(_band_index(fmin, b) + 0.5 + 1e-9))
    last = max(first, int(np.ceil(_band_index(fmax, b) - 0.5 - 1e-9)))
    return first, last


def _band_index(f, b):
    """Return the band index ``x`` of ``nth_octave`` as a real number, so a band covers ``x - 0.5`` to ``x + 0.5``."""
    g = 10 ** (3 / 10)
    octaves = np.log(np.asarray(f, dtype=float) / 1000) / np.log(g)
    return 30 + b * octaves if b % 2 else (59 + 2 * b * octaves) / 2


def _validated_fraction(fraction):
    """Return the bandwidth denominator ``b`` as an int, refusing values that are not positive integers."""
    value = float("nan")
    if not isinstance(fraction, (bool, np.bool_, str, bytes)):
        try:
            value = float(fraction)
        except (TypeError, ValueError):
            pass
    if not np.isfinite(value) or value % 1 or value < 1:
        raise ValueError(f"fraction must be a positive integer bandwidth denominator, got {fraction!r}.")
    return int(value)


def nth_octave(fraction, fmin=20, fmax=20000):
    """
    Return fractional-octave band centres and edges per IEC 61260-1:2014 and ANSI S1.11-2004.

    With ``G = 10**(3/10)`` (IEC 61260-1, Formula 1), integer band index ``x`` and bandwidth designator ``1/b``, the
    exact mid-band frequencies are ``fm = 1000*G**(x/b)`` for odd ``b`` and ``fm = 1000*G**((2*x + 1)/(2*b))`` for even
    ``b`` (Formulas 2 and 3), and the band edges are ``fm*G**(-1/(2*b))`` and ``fm*G**(1/(2*b))`` (Formulas 4 and 5).
    The code numbers the bands as ANSI S1.11-2004 does (equations 3 and 4): the ANSI band number minus 30 is the IEC
    index ``x``. The bands run from the one that contains ``fmin`` to the first one whose upper edge reaches ``fmax``,
    with no other limit on the range, and a frequency on an edge counts in the upper band. Neighbouring bands share one
    computed edge value. Each band is labelled with its nominal mid-band frequency, see ``nominal_frequency``.

    Parameters
    ----------
    fraction : int
        Bandwidth denominator ``b``: ``1`` for octave bands, ``3`` for third-octave bands.
    fmin : float, optional
        Lowest frequency the bands must cover [Hz]. Must be positive.
    fmax : float, optional
        Highest frequency the bands must cover [Hz]. Must not be below ``fmin``.

    Returns
    -------
    ndarray
        Shape ``(n_bands, 3)``: lower edge, nominal mid-band frequency and upper edge of each band [Hz].
    """
    b = _validated_fraction(fraction)
    try:
        if isinstance(fmin, (str, bytes, bool, np.bool_)) or isinstance(fmax, (str, bytes, bool, np.bool_)):
            raise TypeError
        fmin, fmax = float(fmin), float(fmax)
    except (TypeError, ValueError):
        fmin = fmax = float("nan")
    if not (np.isfinite(fmin) and np.isfinite(fmax)) or fmin <= 0 or fmax < fmin:
        raise ValueError("Fractional-octave bands need a positive, finite fmin and a finite fmax at or above it [Hz].")

    g = 10 ** (3 / 10)
    first, last = _band_range(b, fmin, fmax)
    exact = ratio(g, np.arange(first, last + 1), b) * 1000
    upper = exact * bandedge(g, b)
    lower = np.concatenate(([exact[0] / bandedge(g, b)], upper[:-1]))
    return np.array([lower, nominal_frequency(exact, b), upper]).T


def nominal_frequency(exact, fraction):
    """
    Return the nominal mid-band frequency of each exact mid-band frequency per IEC 61260-1:2014, Annex E.

    Octave and one-third-octave bands take the values of Table E.1, extended to any decade (clause E.1). One-half-octave
    bands round to three significant digits (E.2). Bands from 1/4 to 1/24 octave round to three significant digits when
    the leading digit is 1 to 4 and to two when it is 5 to 9 (E.3.2 and E.3.3). Narrower bands add digits until the
    labels are unique in any 10:1 frequency ratio (E.3.5).

    Parameters
    ----------
    exact : ndarray
        Exact mid-band frequencies [Hz].
    fraction : int
        Bandwidth denominator ``b``.

    Returns
    -------
    ndarray
        Nominal mid-band frequencies [Hz].
    """
    exact = np.asarray(exact, dtype=float)
    b = _validated_fraction(fraction)
    exponent = np.floor(np.log10(exact)).astype(int)
    if b in (1, 3):
        hundredths = exact / 10.0 ** exponent * 100
        nearest = np.argmin(np.abs(np.log(hundredths[..., None] / NOMINAL_HUNDREDTHS)), axis=-1)
        return _place_decimal(NOMINAL_HUNDREDTHS[nearest], exponent - 2)

    def rounded(values, extra):
        power = np.floor(np.log10(values)).astype(int)
        leading = np.floor(values / 10.0 ** power)
        digits = 3 if b == 2 else np.where(leading <= 4, 3, 2)
        place = power - digits - extra + 1
        return _place_decimal(np.round(_place_decimal(values, -place)), place)

    extra = 0
    if b > 24:
        # The base-ten bands repeat, a decimal place up, every 10*b indices (G**10 = 1000), so three decades hold every
        # pair of neighbours; labels that rise strictly there are unique in any 10:1 frequency ratio.
        start = int(np.floor(_band_index(1000, b)))
        span = ratio(10 ** (3 / 10), np.arange(start, start + 10 * b + 2), b) * 1000
        while not np.all(np.diff(rounded(span, extra)) > 0):
            extra += 1
    return rounded(exact, extra)


def _place_decimal(values, places):
    """
    Return ``values * 10**places`` with the shift done by an exact power of ten, so 237 at ``-1`` gives 23.7 and not
    23.700000000000003.
    """
    values = np.asarray(values, dtype=float)
    places = np.broadcast_to(np.asarray(places), values.shape)
    scale = 10.0 ** np.abs(places)
    return np.where(places >= 0, values * scale, values / scale)


def band_edges(freq, fraction):
    """
    Return band edges for given nominal mid-band frequencies per IEC 61260-1:2014, Formulas 4 and 5.

    Each nominal frequency is matched to the nearest exact mid-band frequency of ``nth_octave``, and the edges come from
    that exact frequency.

    Parameters
    ----------
    freq : ndarray
        Nominal mid-band frequencies [Hz].
    fraction : int
        Bandwidth denominator ``b``, as in ``nth_octave``.

    Returns
    -------
    ndarray
        Shape ``(len(freq), 3)``: lower edge, nominal mid-band frequency and upper edge of each band [Hz].
    """
    b = _validated_fraction(fraction)
    freq = np.asarray(freq, dtype=float)
    g = 10 ** (3 / 10)
    exact = ratio(g, np.round(_band_index(freq, b)), b) * 1000
    edge = bandedge(g, b)
    return np.array([exact / edge, freq, exact * edge]).T


def ratio(g, x, b):
    if b % 2:  # ODD (ANSI s1.11, eq. 3)
        return g ** ((x - 30) / b)
    else:  # EVEN (ANSI s1.11, eq. 4)
        return g ** ((2 * x - 59) / (2 * b))


def bandedge(g, b):
    # Band-edge ratio (ANSI s1.11, 3.7, pg. 3)
    return g ** (1 / (2 * b))
