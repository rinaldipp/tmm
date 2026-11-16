"""
Utility functions for air properties, for fractional-octave bands per IEC 61260-1 [1] and ANSI S1.11 [2], and for the
checks of physical inputs that the layer methods, terminations and helpers share.

References
----------
[1] IEC 61260-1:2014, "Electroacoustics - Octave-band and fractional-octave-band filters - Part 1: Specifications",
    International Electrotechnical Commission, 2014.
[2] ANSI S1.11-2004, "Specification for Octave-Band and Fractional-Octave-Band Analog and Digital Filters",
    Acoustical Society of America, 2004 (reaffirmed 2009).

For further information check the function specific documentation.
"""
import numbers

import numpy as np
import scipy

# Nominal mid-band frequencies of octave and one-third-octave bands in IEC 61260-1:2014, Table E.1, as hundredths of
# their decade, which clause E.1 extends to any decade by placing the decimal sign. Integers keep the labels exact.
NOMINAL_HUNDREDTHS = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000])

# Characters of a value's repr that an input-check message echoes before it cuts the text.
ECHO_LENGTH = 60

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


def report_layer_keys(matrix):
    """
    Return the keys of a ``TMM.matrix`` in the order of the layer reports: the layer indices in ascending order, then
    the other entries, such as a termination record, in their stored order.
    """
    return sorted(key for key in matrix if isinstance(key, int)) + [key for key in matrix if not isinstance(key, int)]


def layer_label(matrix, key):
    """
    Return ``"Layer n"`` for the entry ``key`` of a ``TMM.matrix``, with ``n`` counted from 1 in the order of
    ``report_layer_keys()``, the numbering of ``TMM.show_layers()``.
    """
    return f"Layer {report_layer_keys(matrix).index(key) + 1}"


def plain_scalar(value):
    """Return a NumPy scalar or 0-d array as the Python scalar it holds, and anything else unchanged."""
    if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and value.ndim == 0):
        return value.item()
    return value


def _shape(value):
    """Return ``np.shape(value)``, or ``(len(value),)`` for a nested sequence of unequal lengths."""
    try:
        return np.shape(value)
    except ValueError:
        return (len(value),)


def describe_value(value):
    """
    Return the text with which an input check echoes ``value``: its ``repr``, NumPy scalars and 0-d arrays as the Python
    number they hold, a sequence or an array as its kind and its length or shape without the values, a dictionary or a
    set as its kind and size, and any other ``repr`` longer than ``ECHO_LENGTH`` characters cut, with its type named. A
    value whose ``repr`` raises is named by its type alone.
    """
    value = plain_scalar(value)
    if isinstance(value, (dict, set, frozenset)):
        return f"a {type(value).__name__} of {len(value)} item{'' if len(value) == 1 else 's'}"
    shape = () if isinstance(value, (str, bytes)) else _shape(value)
    if shape:
        kind = "an array" if isinstance(value, np.ndarray) else f"a {type(value).__name__}"
        if len(shape) == 1:
            return f"{kind} of {shape[0]} value{'' if shape[0] == 1 else 's'}"
        return f"{kind} of shape {shape}"
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001  (an input check raises its own message, whatever the value's repr does)
        return f"a {type(value).__name__}"
    if len(text) > ECHO_LENGTH:
        return f"a {type(value).__name__} starting {text[:ECHO_LENGTH]}..."
    return text


def holds_boolean(value):
    """Return True when ``value`` is a boolean or a sequence holding one, which NumPy would read as 0 or 1."""
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b"
    try:
        return any(isinstance(item, (bool, np.bool_)) for item in np.ravel(np.asarray(value, dtype=object)))
    except (TypeError, ValueError):
        return False


def number_texts(*values):
    """
    Return ``values`` as text with at least six significant digits, and with as many more as it takes for different
    values to read differently, so that a message comparing them shows the difference. Where 16 digits do not suffice,
    each value is written as its ``repr``, the shortest text that reads back as the same float.
    """
    values = [float(value) for value in values]
    for digits in range(6, 17):
        texts = [f"{value:.{digits}g}" for value in values]
        if all((texts[i] == texts[j]) == (values[i] == values[j])
               for i in range(len(values)) for j in range(i + 1, len(values))):
            return texts
    return [repr(value) for value in values]


def real_number(value):
    """
    Return ``value`` as a float when it is one real number: a Python or NumPy real scalar, a ``fractions.Fraction`` or a
    0-d real array. Return ``None`` for anything else, booleans, complex numbers and arrays of one element included.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "iuf" or value.ndim != 0:
            return None
        return float(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):  # A Real type that does not convert, or a huge Fraction.
        return None


def real_array(value):
    """
    Return ``np.asarray(value)``, converting an array of objects that are all real numbers in the sense of
    ``real_number``, as a list of ``fractions.Fraction`` gives, to an array of floats of the same shape.
    """
    array = np.asarray(value)
    if array.dtype.kind == "O" and array.size:
        items = [real_number(item) for item in array.ravel()]
        if all(item is not None for item in items):
            return np.array(items, dtype=float).reshape(array.shape)
    return array


def check_number(method, name, what, value, rule, accept, note=""):
    """
    Return ``value`` as a Python float when it is one real number for which ``accept`` holds, and raise a ``ValueError``
    otherwise. The input checks of the package share one message form,
    ``"{method}: {name}, {what}, must be {rule}, not {value}."``, followed by ``note``. ``what`` names the quantity with
    its unit, and the value is echoed by ``describe_value``. For a sequence or an array, ``rule`` reads "a single finite
    number" in place of "a finite number".
    """
    number = real_number(value)
    if number is not None and accept(number):
        return number
    if not isinstance(value, (str, bytes, dict, set, frozenset)) and _shape(value):
        rule = rule.replace("a finite number", "a single finite number", 1)
    raise ValueError(f"{method}: {name}, {what}, must be {rule}, not {describe_value(value)}.{note}")


def check_positive(method, name, what, value, note=""):
    """Check that ``value`` is one finite number above 0, as ``check_number`` does."""
    return check_number(method, name, what, value, "a finite number above 0", lambda x: np.isfinite(x) and x > 0,
                        note)


def check_at_most(method, name, what, value, bound_name, bound_what, bound, unit, note=""):
    """
    Raise a ``ValueError`` in the shared form when the checked number ``value`` is above ``bound``, the value of the
    parameter ``bound_name``, which ``bound_what`` describes. Both are printed in ``unit`` with as many digits as tell
    them apart.
    """
    if value <= bound:
        return value
    shown, limit = number_texts(value, bound)
    raise ValueError(f"{method}: {name}, {what}, must be at most {bound_what} {bound_name} = {limit} {unit}, not "
                     f"{shown}.{note}")


def check_flag(method, name, value):
    """Return ``value`` as a bool when it is ``True`` or ``False``, a NumPy boolean included, and raise otherwise."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{method}: {name} must be True or False, not {describe_value(value)}.")


def check_choice(method, name, what, value, choices):
    """
    Return ``value`` when it is one of the names ``choices``, matched exactly, and raise a ``ValueError`` that lists
    them otherwise, also for a name in another case and for a value that is not text.
    """
    if isinstance(value, str) and value in choices:
        return value
    listed = ", ".join(repr(choice) for choice in choices)
    raise ValueError(f"{method}: {name}, {what}, must be one of {listed}, not {describe_value(value)}.")


def check_index(method, value, name="layer", what="the position of the layer in the stack"):
    """
    Return ``None``, or a whole number given as a Python or NumPy integer or as a float with an integer value as an int,
    and raise otherwise.
    """
    if value is None:
        return None
    if isinstance(value, (numbers.Integral, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{method}: {name}, {what}, must be None or an integer, not {describe_value(value)}.")


def check_sample_size(method, value, limit):
    """
    Return the sides ``(a, b)`` [m] of ``sample_size``: one finite number above 0 for a square, or a pair of them, each
    at most ``limit``. Anything else, booleans, text, complex numbers and nested sequences included, raises a
    ``ValueError`` in the shared form.
    """
    what = "the sample side lengths [m]"
    rule = "a finite number above 0 or a pair of them"
    if isinstance(value, np.ma.MaskedArray):
        sides = [None]
    elif isinstance(value, (str, bytes, dict, set, frozenset)) or not _shape(value):
        sides = [real_number(value)] * 2
    elif len(_shape(value)) == 1 and _shape(value)[0] in (1, 2):
        sides = [real_number(side) for side in value]
        sides = sides * 2 if len(sides) == 1 else sides
    else:
        sides = [None]
    bad = [side is None or not np.isfinite(side) or side <= 0 for side in sides]
    if any(bad):
        named = ""
        if not isinstance(value, np.ma.MaskedArray) and _shape(value) == (2,):
            position = bad.index(True)
            named = f" with {describe_value(list(value)[position])} as its {('first', 'second')[position]} side"
        raise ValueError(f"{method}: sample_size, {what}, must be {rule}, not {describe_value(value)}{named}.")
    if max(sides) > limit:
        texts = number_texts(*sides, limit)[:2]  # Digits enough to tell each side from the limit.
        shown = ", ".join(texts) if _shape(value) else texts[0]
        raise ValueError(f"{method}: sample_size, {what}, must have no side above SAMPLE_SIZE_LIMIT = {limit:g} m, not "
                         f"{'(' + shown + ')' if _shape(value) else shown}. sample_size is in metres, and a side that "
                         f"large is likely one in millimetres.")
    return tuple(sides)


def check_non_negative(method, name, what, value):
    """Check that ``value`` is one finite number of at least 0, for a quantity whose 0 means none."""
    return check_number(method, name, what, value, "a finite number of at least 0",
                        lambda x: np.isfinite(x) and x >= 0)


def check_fraction(method, name, what, value):
    """Check that ``value`` is one finite number above 0 and at most 1, as a porosity or an open area is."""
    return check_number(method, name, what, value, "a finite number above 0 and at most 1", lambda x: 0 < x <= 1)


def check_tortuosity(method, name, what, value):
    """Check that ``value`` is one finite number of at least 1, as a tortuosity is."""
    return check_number(method, name, what, value, "a finite number of at least 1", lambda x: 1 <= x < np.inf)


def check_interval(method, name, what, value, low, high, closed=(True, True)):
    """
    Check that ``value`` is one finite number between ``low`` and ``high``, each bound included where ``closed`` says
    so.
    """
    if all(closed):
        rule = f"a finite number from {low:g} to {high:g}"
    else:
        rule = (f"a finite number {'of at least' if closed[0] else 'above'} {low:g} and "
                f"{'at most' if closed[1] else 'below'} {high:g}")
    return check_number(method, name, what, value, rule,
                        lambda x: (low <= x if closed[0] else low < x) and (x <= high if closed[1] else x < high))


def check_thickness(method, value, name="t", what="the thickness [mm]"):
    """Check that a layer thickness is one finite number above 0; a zero thickness is told to leave the layer out."""
    note = " Leave the layer out instead of giving it zero thickness." if real_number(value) == 0 else ""
    return check_positive(method, name, what, value, note)


def check_plate_density(method, value, what="the plate density [kg/m³]"):
    """Check that a plate density is ``None``, a rigid plate, or one finite number above 0."""
    if value is None:
        return value
    note = " Pass rho=None for a rigid plate." if real_number(value) == 0 else ""
    return check_number(method, "rho", what, value, "None (a rigid plate) or a finite number above 0",
                        lambda x: np.isfinite(x) and x > 0, note)


def check_values(method, name, what, value, rule, accept):
    """
    Check an input that is an array by definition: every element a finite real number for which ``accept`` holds,
    evaluated on the whole array.
    """
    try:
        array = np.asarray(value)
        valid = (array.dtype.kind in "iuf" and array.size > 0 and bool(np.all(np.isfinite(array)))
                 and bool(np.all(accept(array))))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{method}: {name}, {what}, must be {rule}, not {describe_value(value)}.")
    return value


def check_impedance(method, name, what, value):
    """
    Return an impedance given as one number, real or complex (a Python or NumPy scalar or a 0-d array), as a Python
    float or complex, and raise a ``ValueError`` for a NaN, a negative real part, an array or a list of any length, and
    anything that is not a number. Zero (pressure release) and positive infinity (rigid) pass.
    """
    number = plain_scalar(value)
    if isinstance(number, (numbers.Real, numbers.Complex)) and not isinstance(number, bool):
        number = complex(number) if isinstance(number, complex) else real_number(number)
        if number is not None and not np.isnan(number) and number.real >= 0:
            return number
    rule = "a single number, real or complex," if _shape(value) else "a number"
    raise ValueError(f"{method}: {name}, {what}, must be {rule} without NaN and with a real part of at least 0, not "
                     f"{describe_value(value)}.")


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
