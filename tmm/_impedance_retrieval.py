"""
Surface impedance of a locally reacting boundary from band absorption coefficients.

Absorption fixes only the magnitude of the reflection factor, so an absorption coefficient below 1 belongs to two real
impedances and to infinitely many complex ones, and the functions here take the choice as an argument. Each relates the
normalized impedance ``zeta = Z/(rho0*c0)`` of a locally reacting surface to an absorption coefficient through one of
three forward models. These are the absorption at one angle of incidence ([1], eq. 2.11), the random-incidence (Paris)
average in closed form ([1], eq. 2.54), and the statistical absorption of a finite sample in a reverberation room ([2],
eq. 25; [3], eqs. 5 and 6). A real impedance follows from inverting one of these relations with a chosen root, or from
the 55 degree rule ([4], sec. III.E; [5], sec. 13.1.1). A complex impedance follows from fitting the passive, causal
RMK+1 model of Mondet et al. ([6], eq. 24) to the band values under the constraints of a named absorber class.

The functions are pure and take plain arrays, so they serve measured or tabulated data as well as ``TMM`` results. The
time convention is ``exp(+j*omega*t)``, as in the rest of the package.

References
----------
[1] H. Kuttruff, "Room Acoustics", 6th ed., CRC Press, 2017.
[2] S.-I. Thomasson, "On the absorption coefficient", Acustica, vol. 44, pp. 265-273, 1980.
[3] C.-H. Jeong, "Converting Sabine absorption coefficients to random incidence absorption coefficients", Journal of
    the Acoustical Society of America, vol. 133, pp. 3951-3962, 2013.
[4] J. A. Hargreaves, L. R. Rendell and Y. W. Lam, "A framework for auralization of boundary element method
    simulations including source and receiver directivity", Journal of the Acoustical Society of America, vol. 145,
    pp. 2625-2637, 2019.
[5] T. J. Cox and P. D'Antonio, "Acoustic Absorbers and Diffusers: Theory, Design and Application", 3rd ed.,
    CRC Press, 2017.
[6] B. Mondet, J. Brunskog, C.-H. Jeong and J. H. Rindel, "From absorption to impedance: Enhancing boundary
    conditions in room acoustic simulations", Applied Acoustics, vol. 157, 106884, 2020.

For further information check the function specific documentation.
"""
from math import comb

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from tmm import _utils as utils

# Cosine of the angle at which the 55 degree rule evaluates the oblique-incidence relation.
COS_55 = float(np.cos(np.deg2rad(55.0)))

# Maximum of the random-incidence absorption of a locally reacting surface ([1], eq. 2.54) and the real impedance at
# which it occurs, evaluated to the digits given; [1], p. 41, prints 0.951 and 1.567.
PARIS_ALPHA_MAX = 0.9512220138
PARIS_ZETA_MAX = 1.5669241

# abs(zeta) below which paris_alpha sums its power series instead of the closed form, whose terms cancel there, and the
# number of series terms. With these values both forms agree with numerical quadrature to about 1e-12 relative.
PARIS_SERIES_LIMIT = 0.05
PARIS_SERIES_TERMS = 16

# Points per band, uniform in linear frequency, over which a band mean of a model absorption is taken.
BAND_POINTS = 5

# Penalty weight of a class-constraint violation in the least-squares residual of fit_absorber_model.
CONSTRAINT_WEIGHT = 100.0

# Points, uniform in log(f) over the band range, on which fit_absorber_model holds the resistance constraints.
CONSTRAINT_CHECK_POINTS = 64

# Tolerance (xtol, ftol and gtol) and largest number of function evaluations of each least-squares start.
FIT_TOLERANCE = 1e-8
FIT_EVALUATIONS = 400

# Largest root-mean-square band error above the best one at which another start of fit_absorber_model counts as an
# equally good fit for its spread.
SPREAD_TOLERANCE = 0.005

# Names of the RMK+1 parameters, in the order rmk1_impedance takes them. zeta is normalized and omega is in rad/s.
RMK1_PARAMETERS = ("R1", "M [s]", "K [1/s]", "G [s**gamma]", "gamma")

# Absorber classes of fit_absorber_model and the constraints of [6] each one applies: "hard" is eq. 28, "soft" eq. 29,
# "porous" eq. 30, "hard_porous" eq. 31 and "membrane" eq. 32. Perforated panels and membranes count as soft materials
# and microperforated panels as hard ones ([6], p. 5).
ABSORBER_CLASSES = {
    "hard": ("hard",),
    "soft": ("soft",),
    "soft_porous": ("soft", "porous"),
    "hard_porous": ("hard", "porous", "hard_porous"),
    "membrane": ("soft", "membrane"),
    "perforated": ("soft",),
    "microperforated": ("hard",),
}

# Validity limits of [6], eqs. 34 to 36, on the cost F of eq. 26 summed over the six octave bands 125 Hz to 4 kHz, and
# that number of bands. fit_absorber_model applies them as the root-mean-square band errors they correspond to.
VALIDITY_COST = {"above_one": 0.250, "below_0.1": 0.001, "other": 0.015}
VALIDITY_BANDS = 6

# Search box of the normalized RMK+1 parameters (log10 R1, log10 M*1e4, log10 K*1e-4, log10 G*1e4**gamma, gamma), the
# starting point of [6], p. 5 (R1 = 1.6, the other parameters at zero, which the logarithms place at the lower end of
# the box and gamma at 0), and fixed further starting points: the first unscrambled Sobol points in five dimensions,
# mapped to the start box.
RMK1_LOWER = np.array([-4.0, -6.0, -6.0, -6.0, -1.0])
RMK1_UPPER = np.array([3.0, 6.0, 6.0, 6.0, 1.0])
RMK1_FIRST_START = np.array([np.log10(1.6), -6.0, -6.0, -6.0, 0.0])
RMK1_START_LOWER = np.array([-1.0, -3.0, -3.0, -3.0, -1.0])
RMK1_START_UPPER = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
RMK1_START_FRACTIONS = np.array([[0.5, 0.5, 0.5, 0.5, 0.5],
                                 [0.75, 0.25, 0.25, 0.25, 0.75],
                                 [0.25, 0.75, 0.75, 0.75, 0.25],
                                 [0.375, 0.375, 0.625, 0.875, 0.375],
                                 [0.875, 0.875, 0.125, 0.375, 0.875],
                                 [0.625, 0.125, 0.875, 0.625, 0.625],
                                 [0.125, 0.625, 0.375, 0.125, 0.125]])

# Frequency at which the porous constraint of [6], eq. 30, holds the reactance negative [Hz].
POROUS_CHECK_FREQUENCY = 5000.0

# Frequency above which the hard-porous constraint of [6], eq. 31, holds the resistance below 2 [Hz].
HARD_POROUS_CHECK_FREQUENCY = 1600.0


def alpha_at_angle(zeta, angle=0.0):
    """
    Return the absorption coefficient of a locally reacting surface at one angle of incidence.

    The relation is ``alpha = 4*Re(zeta)*cos(theta)/(abs(zeta)**2*cos(theta)**2 + 2*Re(zeta)*cos(theta) + 1)``, Kuttruff
    [1], eq. 2.11, p. 26. An infinite ``zeta`` gives 0.

    Parameters
    ----------
    zeta : complex or array of complex
        Normalized surface impedance ``Z/(rho0*c0)``.
    angle : float, optional
        Angle of incidence from the surface normal [deg].

    Returns
    -------
    array
        Absorption coefficient.
    """
    zeta = np.asarray(zeta, dtype=complex)
    cos_theta = np.cos(np.deg2rad(angle))
    with np.errstate(invalid="ignore", over="ignore"):
        alpha = 4 * zeta.real * cos_theta / (np.abs(zeta) ** 2 * cos_theta ** 2 + 2 * zeta.real * cos_theta + 1)
    return np.where(np.isinf(zeta.real) | np.isinf(zeta.imag), 0.0, alpha)


def paris_alpha(zeta):
    """
    Return the random-incidence absorption coefficient of a locally reacting surface in closed form.

    The Paris average ``2*integral of alpha(theta)*cos(theta)*sin(theta)`` over 0 to 90 degrees ([1], eq. 2.53) of the
    oblique relation ([1], eq. 2.11) gives, for ``zeta = xi + j*eta``, [1], eq. 2.54, p. 40::

        alpha = 8*xi/abs(zeta)**2 * (1 + (xi**2 - eta**2)/(eta*abs(zeta)**2)*arctan(eta/(1 + xi))
                                     - xi/abs(zeta)**2*ln(1 + 2*xi + abs(zeta)**2))

    The arctangent term is evaluated as ``(xi**2 - eta**2)/(abs(zeta)**2*(1 + xi)) * arctan(x)/x`` with
    ``x = eta/(1 + xi)``, which has the finite limit 1 for ``arctan(x)/x`` at ``eta = 0``. The relation is even in
    ``eta``, so it holds in either time convention. Its largest value is 0.951, at the real impedance ``zeta = 1.567``
    ([1], p. 41). An infinite ``zeta``, one whose squared magnitude overflows, and a purely reactive one give 0.

    Below ``abs(zeta) = PARIS_SERIES_LIMIT`` the terms of the closed form cancel, and the function sums instead the
    first ``PARIS_SERIES_TERMS`` terms of a power series, a package derivation. With ``u = cos(theta)`` and
    ``m = abs(zeta)**2`` the average is ``alpha = 8*xi*integral of u**2/(1 + e) du`` over 0 to 1, with
    ``e = 2*xi*u + m*u**2``. Expanding ``1/(1 + e)`` in powers of ``-e`` and ``e**n`` binomially gives
    ``alpha = 8*xi*sum over n of (-1)**n*sum over k of C(n, k)*(2*xi)**k*m**(n - k)/(2*n - k + 3)``.

    Parameters
    ----------
    zeta : complex or array of complex
        Normalized surface impedance ``Z/(rho0*c0)``.

    Returns
    -------
    array
        Random-incidence absorption coefficient.
    """
    zeta = np.asarray(zeta, dtype=complex)
    xi, eta = zeta.real, zeta.imag
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        magnitude = np.abs(zeta) ** 2
        x = eta / (1 + xi)
        arctan_ratio = np.where(x == 0, 1.0, np.arctan(x) / np.where(x == 0, 1.0, x))
        alpha = 8 * xi / magnitude * (1 + (xi ** 2 - eta ** 2) / (magnitude * (1 + xi)) * arctan_ratio
                                      - xi / magnitude * np.log1p(2 * xi + magnitude))
    small = np.abs(zeta) < PARIS_SERIES_LIMIT
    if np.any(small):
        alpha = np.array(alpha, dtype=float)
        alpha[small] = 8 * xi[small] * _paris_series(xi[small], magnitude[small])
    return np.where(~np.isfinite(magnitude) | (xi == 0), 0.0, alpha)


def _paris_series(xi, magnitude):
    """Return the power series of ``paris_alpha`` for small ``abs(zeta)``, without its factor ``8*xi``."""
    total = np.zeros_like(xi)
    with np.errstate(under="ignore"):
        for n in range(PARIS_SERIES_TERMS):
            for k in range(n + 1):
                total += (-1) ** n * comb(n, k) * (2 * xi) ** k * magnitude ** (n - k) / (2 * n - k + 3)
    return total


def finite_sample_alpha(zeta, z_f, theta, phi_weights):
    """
    Return the statistical absorption coefficient of a finite, locally reacting sample.

    This is Thomasson's statistical absorption coefficient [2], eq. 25, p. 270, as ``TMM.finite_sample_absorption()``
    evaluates it with ``normalization="incident"`` and uniform angular weights:
    ``alpha = 2 * integral over theta of mean over phi of 4*Re(zeta)*sin(theta)/abs(zeta + Z_F)**2``, by the trapezoidal
    rule in ``theta``. ``Z_F`` is Thomasson's field impedance of the sample in the ``exp(+j*omega*t)`` convention. Jeong
    [3], eq. 6 with ``Z_r`` from eq. 5, writes the same coefficient with ``Z_F`` averaged over azimuth before the
    magnitude is taken, as does [2], eq. 26. This function averages the absorption instead, as
    ``TMM.finite_sample_absorption()`` does.

    Parameters
    ----------
    zeta : array of complex
        Normalized surface impedance, one value per row of ``z_f``.
    z_f : array of complex
        Field impedance with shape ``(len(zeta), len(theta), len(phi_weights))``.
    theta : array of float
        Polar angles of the ``z_f`` columns [rad], from 0 to the upper integration limit.
    phi_weights : array of float
        Quadrature weights of the azimuth nodes of ``z_f``, summing to 1.

    Returns
    -------
    array
        Statistical absorption coefficient, one value per entry of ``zeta``.
    """
    zeta = np.asarray(zeta, dtype=complex).reshape(-1, 1, 1)
    theta = np.asarray(theta, dtype=float)
    with np.errstate(invalid="ignore"):
        integrand = np.where(np.isinf(zeta), 0.0, 4 * zeta.real / np.abs(zeta + z_f) ** 2) @ np.asarray(phi_weights)
    integrator = getattr(np, "trapezoid", None) or np.trapz
    return 2 * integrator(integrand * np.sin(theta)[None, :], theta, axis=1)


def real_impedance_normal(alpha, root="hard"):
    """
    Return the real normalized impedance that has a given normal-incidence absorption coefficient.

    With a real reflection factor ``R = +s`` or ``R = -s``, ``s = sqrt(1 - alpha)``, Kuttruff [1], eq. 2.9 at
    ``theta = 0``, p. 26, gives ``zeta = (1 + s)/(1 - s)`` for the hard root and ``zeta = (1 - s)/(1 + s)``, its
    reciprocal, for the soft root. ``alpha = 1`` gives ``zeta = 1``, and ``alpha = 0`` gives an infinite impedance (hard
    root) or zero (soft root).

    Parameters
    ----------
    alpha : float or array of float
        Normal-incidence absorption coefficient, from 0 to 1.
    root : str, optional
        ``"hard"`` for ``zeta >= 1`` or ``"soft"`` for ``zeta <= 1``.

    Returns
    -------
    array
        Real normalized impedance.
    """
    _validate_root(root)
    s = np.sqrt(np.clip(1 - _finite_alpha(alpha), 0.0, 1.0))
    with np.errstate(divide="ignore"):
        return (1 + s) / (1 - s) if root == "hard" else (1 - s) / (1 + s)


def admittance_55_degrees(alpha, root="hard"):
    """
    Return the real normalized admittance of the 55 degree rule for a random-incidence absorption coefficient.

    For a locally reacting surface, the absorption at 55 degrees is taken as the random-incidence value (Cox and
    D'Antonio [5], sec. 13.1.1, pp. 441-442). Inverting the oblique relation ([1], eq. 2.11) at 55 degrees with a real
    reflection factor gives the admittance ``Y = cos(55 deg)*(1 - s)/(1 + s)``, ``s = sqrt(1 - alpha)``, as Hargreaves
    et al. [4], sec. III.E, p. 2633, write it. That is the hard root; ``root="soft"`` returns the other one,
    ``Y = cos(55 deg)*(1 + s)/(1 - s)``, a package extension. ``alpha = 0`` gives ``Y = 0`` (hard root), a rigid
    surface. The rule is an approximation, not an inversion of [1], eq. 2.54.

    Parameters
    ----------
    alpha : float or array of float
        Random-incidence absorption coefficient, from 0 to 1.
    root : str, optional
        ``"hard"`` or ``"soft"``.

    Returns
    -------
    array
        Real normalized admittance ``rho0*c0/Z``.
    """
    _validate_root(root)
    s = np.sqrt(np.clip(1 - _finite_alpha(alpha), 0.0, 1.0))
    with np.errstate(divide="ignore"):
        return COS_55 * ((1 - s) / (1 + s) if root == "hard" else (1 + s) / (1 - s))


def real_impedance_inverse(alpha, freq=None, forward="random", root="hard"):
    """
    Return the real normalized impedance whose forward-model absorption equals ``alpha`` at each frequency.

    For a real ``zeta`` each forward model rises from 0 to a single maximum and falls back to 0, so every value below
    the maximum has two roots. The method finds the maximum at each frequency by golden-section search in ``ln(zeta)``
    and the root on the chosen side by bisection. Values at or above the maximum return the impedance of the maximum,
    and ``alpha <= 0`` returns an infinite impedance (hard root) or zero (soft root). With ``forward="random"`` this is
    the numerical inverse of [1], eq. 2.54, whose maximum is 0.951 at ``zeta = 1.567`` ([1], p. 41). Close to the
    maximum the absorption changes with the square of the distance from it, so there the root is resolved only to about
    the square root of the double-precision step, while its absorption is exact. For a very small ``alpha`` the soft
    root loses accuracy, since eq. 2.54 then cancels to rounding level.

    Parameters
    ----------
    alpha : float or array of float
        Absorption coefficient to match. NaN and infinite values raise ``ValueError``.
    freq : float or array of float, optional
        Frequencies of ``alpha`` [Hz], passed on to a callable ``forward``.
    forward : str or callable, optional
        ``"normal"``, ``"random"``, or a callable ``forward(zeta, freq)`` returning the absorption of the impedances
        ``zeta`` at the frequencies ``freq``, both arrays of the same shape.
    root : str, optional
        ``"hard"`` for the root above the maximum or ``"soft"`` for the one below it.

    Returns
    -------
    zeta : array
        Real normalized impedance.
    alpha_max : array
        Largest absorption a real impedance reaches at each frequency under ``forward``.
    """
    _validate_root(root)
    model = forward_model(forward)
    alpha = np.atleast_1d(_finite_alpha(alpha))
    freq = np.broadcast_to(np.zeros(1) if freq is None else np.asarray(freq, dtype=float), alpha.shape)

    def absorption(log_zeta):
        return np.asarray(model(np.exp(log_zeta).astype(complex), freq), dtype=float)

    golden = (np.sqrt(5) - 1) / 2
    low, high = np.full(alpha.shape, np.log(1e-3)), np.full(alpha.shape, np.log(1e3))
    for _ in range(90):
        left, right = high - golden * (high - low), low + golden * (high - low)
        rising = absorption(left) < absorption(right)
        low, high = np.where(rising, left, low), np.where(rising, high, right)
    peak = (low + high) / 2
    alpha_max = absorption(peak)

    if root == "hard":
        low, high = peak.copy(), np.full(alpha.shape, np.log(1e15))
    else:
        low, high = np.full(alpha.shape, np.log(1e-15)), peak.copy()
    for _ in range(120):
        middle = (low + high) / 2
        above = absorption(middle) > alpha
        if root == "hard":
            low, high = np.where(above, middle, low), np.where(above, high, middle)
        else:
            low, high = np.where(above, low, middle), np.where(above, middle, high)
    zeta = np.exp((low + high) / 2)
    zeta = np.where(alpha >= alpha_max, np.exp(peak), zeta)
    zeta = np.where(alpha <= 0, np.inf if root == "hard" else 0.0, zeta)
    return zeta, alpha_max


def real_impedance_random(alpha, root="hard"):
    """
    Return the real normalized impedance that has a given random-incidence absorption coefficient.

    This is the numerical inverse of Kuttruff [1], eq. 2.54, see ``real_impedance_inverse()``. Values at or above the
    maximum of 0.951 ([1], p. 41) return ``zeta = 1.567``, the impedance of the maximum.

    Parameters
    ----------
    alpha : float or array of float
        Random-incidence absorption coefficient.
    root : str, optional
        ``"hard"`` for ``zeta >= 1.567`` or ``"soft"`` for ``zeta <= 1.567``.

    Returns
    -------
    array
        Real normalized impedance.
    """
    return real_impedance_inverse(alpha, forward="random", root=root)[0]


def rmk1_impedance(params, freq):
    """
    Return the normalized impedance of the RMK+1 model.

    Mondet et al. [6], eq. 24, ``zeta = K*(j*omega)**-1 + R1 + M*(j*omega) + G*(j*omega)**gamma``, with
    ``(j*omega)**gamma = cos(gamma*pi/2)*omega**gamma + j*sin(gamma*pi/2)*omega**gamma`` ([6], eq. 10). With
    ``R1, M, K, G >= 0`` and ``-1 <= gamma <= 1`` the impedance is passive, causal and real in the time domain ([6],
    p. 4), and its resistance ``R1 + G*cos(gamma*pi/2)*omega**gamma`` is at least ``R1``.

    Parameters
    ----------
    params : sequence of float
        ``(R1, M, K, G, gamma)``: resistance [-], mass [s], stiffness [1/s], fractional coefficient [s**gamma] and
        exponent [-], all for the impedance normalized by ``rho0*c0`` and ``omega`` in rad/s.
    freq : float or array of float
        Frequency [Hz].

    Returns
    -------
    array
        Normalized surface impedance.
    """
    r1, m, k, g, gamma = (float(value) for value in params)
    jw = 2j * np.pi * np.asarray(freq, dtype=float)
    return k / jw + r1 + m * jw + g * jw ** gamma


def band_fraction(bands, fraction=None):
    """
    Return the bandwidth denominator ``b`` of a sequence of nominal band centre frequencies.

    Without ``fraction``, the bands must be contiguous: the smallest spacing between neighbouring centres is taken as
    one band, and every other spacing must be one band too, so a gap or a stray band of another series raises
    ``ValueError`` instead of setting a wrong bandwidth. Either way, every centre must lie on the fractional-octave
    series of ``tmm._utils.nth_octave`` for ``b``, within a quarter band.

    Parameters
    ----------
    bands : array of float
        Nominal mid-band frequencies [Hz], ascending; at least two when ``fraction`` is omitted.
    fraction : int, optional
        Bandwidth denominator to check against ``bands``.

    Returns
    -------
    int
        ``1`` for octave bands, ``3`` for third-octave bands, and so on.
    """
    bands = _validated_bands(bands)
    if fraction is not None:
        fraction = utils._validated_fraction(fraction)
    elif bands.size < 2:
        raise ValueError("A single band does not show its bandwidth; pass fraction (1 for octaves, 3 for "
                         "third octaves).")
    else:
        steps = np.diff(np.log2(bands))
        fraction = int(np.round(1 / np.min(steps)))
        if fraction < 1:
            raise ValueError("bands are more than an octave apart everywhere; pass fraction.")
        if np.any(np.abs(steps * fraction - 1) > 0.25):
            raise ValueError(f"bands are not contiguous 1/{fraction}-octave bands, so their bandwidth cannot be "
                             f"inferred; pass fraction.")
    _check_band_series(bands, fraction)
    return fraction


def band_points(bands, fraction, points=BAND_POINTS):
    """
    Return frequencies inside each band at which a band mean is evaluated.

    The edges of each band follow IEC 61260-1 through ``tmm._utils.band_edges``. The band is split into ``points`` equal
    intervals in linear frequency and their midpoints are returned, so the mean over them approximates the band mean on
    a linear frequency axis, the mean ``TMM.filter_alpha()`` takes on a linear grid.

    Parameters
    ----------
    bands : array of float
        Nominal mid-band frequencies [Hz].
    fraction : int
        Bandwidth denominator ``b``.
    points : int, optional
        Number of points per band.

    Returns
    -------
    array
        Frequencies [Hz] with shape ``(len(bands), points)``.
    """
    edges = utils.band_edges(_validated_bands(bands), fraction)
    steps = (np.arange(points) + 0.5) / points
    return edges[:, :1] + (edges[:, 2:] - edges[:, :1]) * steps[None, :]


def bands_to_grid(bands, values, freq, extrapolation="hold"):
    """
    Interpolate band values onto a frequency grid.

    Each band value is placed at its nominal centre and the values are joined by a monotone piecewise cubic (PCHIP) in
    ``log2(f)``, which does not overshoot neighbouring band values. With ``extrapolation="hold"``, the only rule,
    frequencies below the first centre take the first value and frequencies above the last centre the last value.
    Hargreaves et al. [4], sec. III.E, p. 2633, hold the absorption constant below their lowest band, which is 20 Hz.
    Holding it below a higher lowest band and above the highest band is a package choice. At very low frequencies,
    Mondet et al. [6], p. 11, write that the extrapolation of their fitted model "can provide more accuracy than the
    simple and common assumption of constant values between the 63 Hz and 125 Hz octave bands". The PCHIP law is also a
    package choice.

    Parameters
    ----------
    bands : array of float
        Nominal mid-band frequencies [Hz], ascending.
    values : array of float
        One finite value per band.
    freq : array of float
        Frequencies to interpolate to [Hz].
    extrapolation : str, optional
        ``"hold"``.

    Returns
    -------
    array
        Interpolated values on ``freq``.
    """
    if extrapolation != "hold":
        raise ValueError("bands_to_grid supports extrapolation='hold' only.")
    bands = _validated_bands(bands)
    values = _finite_alpha(values, "values")
    if values.shape != bands.shape:
        raise ValueError("values must have one entry per band.")
    log_freq = np.log2(np.asarray(freq, dtype=float))
    if bands.size == 1:
        return np.full(log_freq.shape, values[0])
    nodes = np.log2(bands)
    return PchipInterpolator(nodes, values)(np.clip(log_freq, nodes[0], nodes[-1]))


def fit_validity_limit(alpha):
    """
    Return the root-mean-square band error below which a model fit counts as valid, and the rule that sets it.

    Mondet et al. [6], eqs. 34 to 36, p. 5, accept a fit when the cost of eq. 26 over their six octave bands stays below
    0.250 if any input value exceeds 1, below 0.001 if three or more input values are below 0.1, and below 0.015
    otherwise, which correspond to root-mean-square errors of 0.204, 0.013 and 0.050. The rules are tried in that order,
    so data that meet the first two take the first. [6] set these limits for Sabine data of six octave bands. Applying
    their root-mean-square errors to any number of bands and to random- or normal-incidence data is a package extension.

    Parameters
    ----------
    alpha : array of float
        Input band absorption coefficients.

    Returns
    -------
    limit : float
        Largest valid root-mean-square band error.
    rule : str
        ``"above_one"``, ``"below_0.1"`` or ``"other"``.
    """
    alpha = _finite_alpha(alpha)
    if np.any(alpha > 1):
        rule = "above_one"
    elif np.count_nonzero(alpha < 0.1) >= 3:
        rule = "below_0.1"
    else:
        rule = "other"
    return float(np.sqrt(VALIDITY_COST[rule] / VALIDITY_BANDS)), rule


def fit_absorber_model(bands, alpha, absorber=None, forward="random", fraction=None, check_freq=None):
    """
    Fit the RMK+1 impedance model to band absorption coefficients.

    The method minimizes ``sum((alpha_in - alpha_model)**2)`` over the bands, Mondet et al. [6], eq. 26, where
    ``alpha_model`` is the band mean of the forward-model absorption of ``rmk1_impedance()`` over ``BAND_POINTS``
    frequencies per band ([6] use three per octave band). [6] fit Sabine data through the statistical absorption
    coefficient of a finite sample ([6], eq. 5), which a callable ``forward`` provides. ``"random"`` for [1], eq. 2.54,
    and ``"normal"`` for [1], eq. 2.11 at normal incidence, are package extensions. The parameters are searched as
    ``(log10 R1, log10 M*1e4, log10 K*1e-4, log10 G*1e4**gamma, gamma)``, the normalization of [6], p. 5, taken in
    logarithms so that ``R1, M, K, G`` stay positive, a package choice.

    ``absorber`` names a class of ``ABSORBER_CLASSES``, which lists the constraints of [6] that each class applies. The
    constraints are ``"hard"``, ``Re(zeta) > 1`` ([6], eq. 28), through the bound ``R1 >= 1``, since ``R1`` is the
    smallest resistance of the model; ``"soft"``, ``Re(zeta) < 2`` (eq. 29); ``"porous"``, ``Im(zeta) < 0`` at 5 kHz
    (eq. 30); ``"hard_porous"``, ``Re(zeta) < 2`` from 1.6 kHz up (eq. 31); and ``"membrane"``, ``gamma >= 0.9``
    (eq. 32), through the bound. ``"porous"`` is not a class of its own; the classes ``"soft_porous"`` and
    ``"hard_porous"`` carry it. The resistance constraints hold on ``check_freq``, and all constraints except the bounds
    enter the residual as penalties weighted by ``CONSTRAINT_WEIGHT``, where [6] use a constrained solver. ``None`` fits
    without constraints, a package extension.

    The fit starts from ``R1 = 1.6`` with the other parameters at zero ([6], p. 5), which the logarithmic search places
    at the lower end of the box for ``M``, ``K`` and ``G``, with ``gamma = 0``. Seven fixed further points
    (``RMK1_START_FRACTIONS``) follow, where [6] restart from random points while the fit is invalid. Each start runs
    ``scipy.optimize.least_squares`` with the tolerance ``FIT_TOLERANCE`` and at most ``FIT_EVALUATIONS`` evaluations,
    and the lowest cost wins, so the result is deterministic.

    Parameters
    ----------
    bands : array of float
        Nominal mid-band frequencies [Hz], ascending.
    alpha : array of float
        Band absorption coefficients, one finite value per band.
    absorber : str or None, optional
        Absorber class, a key of ``ABSORBER_CLASSES``, or ``None``.
    forward : str or callable, optional
        ``"random"``, ``"normal"``, or a callable ``forward(zeta, freq)`` as in ``real_impedance_inverse()``.
    fraction : int, optional
        Bandwidth denominator ``b``. Inferred from ``bands`` when omitted, see ``band_fraction()``.
    check_freq : array of float, optional
        Frequencies at which the resistance constraints hold [Hz]. Defaults to ``CONSTRAINT_CHECK_POINTS`` points
        uniform in ``log(f)`` from the lower edge of the first band to the upper edge of the last.

    Returns
    -------
    dict
        ``parameters`` (dict by ``RMK1_PARAMETERS``), ``band_alpha`` (model band means), ``rms`` (root-mean-square band
        error), ``rms_limit`` and ``rule`` (see ``fit_validity_limit()``), ``valid``, ``constraint_violation`` (largest
        remaining violation of a penalty constraint) and ``spread``, the largest median difference, over ``check_freq``,
        between the normal-incidence reflection factor of the best fit and that of another start whose root-mean-square
        error is within ``SPREAD_TOLERANCE`` of the best.
    """
    bands = _validated_bands(bands)
    alpha = _finite_alpha(alpha)
    if alpha.shape != bands.shape:
        raise ValueError("alpha must have one entry per band.")
    if absorber is not None and absorber not in ABSORBER_CLASSES:
        raise ValueError(f"absorber must be None or one of {sorted(ABSORBER_CLASSES)}.")
    fraction = band_fraction(bands, fraction)
    model = forward_model(forward)
    constraints = ABSORBER_CLASSES[absorber] if absorber is not None else ()

    points = band_points(bands, fraction)
    flat = points.ravel()
    edges = utils.band_edges(bands, fraction)
    if check_freq is None:
        check_freq = np.geomspace(edges[0, 0], edges[-1, 2], CONSTRAINT_CHECK_POINTS)
    check_freq = np.asarray(check_freq, dtype=float)
    high_check = check_freq[check_freq >= HARD_POROUS_CHECK_FREQUENCY]

    lower, upper = RMK1_LOWER.copy(), RMK1_UPPER.copy()
    if "hard" in constraints:
        lower[0] = 0.0
    if "membrane" in constraints:
        lower[4] = 0.9

    def penalties(params):
        terms = []
        if "soft" in constraints:
            terms.append(np.clip(rmk1_impedance(params, check_freq).real - 2, 0, None))
        if "porous" in constraints:
            terms.append(np.clip(rmk1_impedance(params, [POROUS_CHECK_FREQUENCY]).imag, 0, None))
        if "hard_porous" in constraints and high_check.size:
            terms.append(np.clip(rmk1_impedance(params, high_check).real - 2, 0, None))
        return np.concatenate(terms) if terms else np.zeros(0)

    def band_alpha(params):
        values = model(rmk1_impedance(params, flat), flat)
        return np.asarray(values, dtype=float).reshape(points.shape).mean(axis=1)

    def residuals(x):
        params = _rmk1_from_normalized(x)
        return np.concatenate([alpha - band_alpha(params), CONSTRAINT_WEIGHT * penalties(params)])

    starts = [RMK1_FIRST_START] + [RMK1_START_LOWER + (RMK1_START_UPPER - RMK1_START_LOWER) * row
                                   for row in RMK1_START_FRACTIONS]
    solutions = []
    for start in starts:
        x0 = np.clip(start, lower, upper)
        solution = least_squares(residuals, x0, bounds=(lower, upper), method="trf", xtol=FIT_TOLERANCE,
                                 ftol=FIT_TOLERANCE, gtol=FIT_TOLERANCE, max_nfev=FIT_EVALUATIONS)
        params = _rmk1_from_normalized(solution.x)
        fitted = band_alpha(params)
        solutions.append((float(np.sqrt(np.mean((alpha - fitted) ** 2))), solution.cost, params, fitted))
    best = min(solutions, key=lambda item: item[1])
    rms, _, params, fitted = best
    limit, rule = fit_validity_limit(alpha)
    violation = penalties(params)

    reflection = _normal_reflection(params, check_freq)
    spread = 0.0
    for other_rms, _, other, _ in solutions:
        if other_rms <= rms + SPREAD_TOLERANCE:
            spread = max(spread, float(np.median(np.abs(_normal_reflection(other, check_freq) - reflection))))

    return {"parameters": dict(zip(RMK1_PARAMETERS, (float(value) for value in params))),
            "band_alpha": fitted,
            "rms": rms,
            "rms_limit": limit,
            "rule": rule,
            "valid": bool(rms < limit),
            "constraint_violation": float(violation.max()) if violation.size else 0.0,
            "spread": spread}


def _normal_reflection(params, freq):
    """Return the normal-incidence reflection factor of the RMK+1 model on ``freq``."""
    zeta = rmk1_impedance(params, freq)
    return (zeta - 1) / (zeta + 1)


def _rmk1_from_normalized(x):
    """Return ``(R1, M, K, G, gamma)`` from the normalized search vector of ``fit_absorber_model()``."""
    gamma = float(x[4])
    return np.array([10 ** x[0], 10 ** x[1] * 1e-4, 10 ** x[2] * 1e4, 10 ** x[3] * 1e-4 ** gamma, gamma])


def forward_model(forward):
    """
    Return a forward model as a callable ``forward(zeta, freq)``.

    Parameters
    ----------
    forward : str or callable
        ``"normal"`` for ``alpha_at_angle()`` at normal incidence, ``"random"`` for ``paris_alpha()``, or a callable,
        which is returned unchanged.

    Returns
    -------
    callable
        ``forward(zeta, freq)``, the absorption of the impedances ``zeta`` at the frequencies ``freq`` [Hz].
    """
    if callable(forward):
        return forward
    if forward == "random":
        return lambda zeta, freq: paris_alpha(zeta)
    if forward == "normal":
        return lambda zeta, freq: alpha_at_angle(zeta)
    raise ValueError("forward must be 'normal', 'random' or a callable forward(zeta, freq).")


def _validate_root(root):
    """Raise for a root other than ``"hard"`` or ``"soft"``."""
    if not isinstance(root, str) or root not in ("hard", "soft"):
        raise ValueError(f"root must be 'hard' or 'soft', not {utils.describe_value(root)}.")


def _real_values(values, name):
    """
    Return ``values`` as a float array, refusing text, booleans, masked arrays and a nonzero imaginary part instead of
    converting, using or dropping them.
    """
    if isinstance(values, np.ma.MaskedArray):
        raise ValueError(f"{name} must not be a masked array; pass only the values to use.")
    values = np.asarray(values)
    if values.dtype.kind in "USb" or (values.dtype.kind == "O" and any(
            isinstance(value, (str, bytes, bool, np.bool_)) for value in values.ravel())):
        raise ValueError(f"{name} must be numbers, not text or booleans.")
    if np.iscomplexobj(values):
        if np.any(values.imag != 0):
            raise ValueError(f"{name} must be real, not complex with a nonzero imaginary part.")
        values = values.real
    return np.asarray(values, dtype=float)


def _finite_alpha(alpha, name="alpha"):
    """Return absorption values as a float array, refusing complex, NaN and infinite entries."""
    alpha = _real_values(alpha, name)
    if not np.all(np.isfinite(alpha)):
        raise ValueError(f"{name} must be finite; NaN and infinite absorption coefficients have no impedance.")
    return alpha


def _validated_bands(bands):
    """Return band centres as a float array, refusing complex values and values not positive, finite and ascending."""
    bands = np.atleast_1d(_real_values(bands, "bands"))
    if bands.ndim != 1 or bands.size == 0 or not np.all(np.isfinite(bands)) or np.any(bands <= 0):
        raise ValueError("bands must be a one-dimensional sequence of positive band centre frequencies [Hz].")
    if np.any(np.diff(bands) <= 0):
        raise ValueError("bands must be strictly ascending.")
    return bands


def _check_band_series(bands, fraction):
    """Raise when a band centre is more than a quarter band away from the series of ``nth_octave`` for ``fraction``."""
    exact = utils.band_edges(bands, fraction)[:, 0] * utils.bandedge(10 ** (3 / 10), fraction)
    offset = np.abs(np.log2(bands / exact)) * fraction
    if np.any(offset > 0.25):
        raise ValueError(f"bands {bands[offset > 0.25].tolist()} are not 1/{fraction}-octave centre frequencies.")
