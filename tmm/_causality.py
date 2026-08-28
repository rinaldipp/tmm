"""
Causality constraint on rigid-backed sound absorbers.

Implements the Fano/Rozanov-type sum rule adapted to acoustics by Yang, Chen, Fu and Sheng, which ties the
absorption spectrum of a rigid-backed layer to a lower bound on its thickness::

    d_min = (1 / (4 * pi**2)) * (Beff / B0) * abs(integral of ln[1 - A(lambda)] dlambda) <= d

With ``lambda = c0 / f``, the frequency-domain form evaluated here is::

    d_min = (c0 / (4 * pi**2)) * (Beff / B0) * integral of -ln[1 - A(f)] / f**2 df

The functions in this module are pure and take plain arrays, so they can be used with measured absorption
spectra as well as with TMM results.

References
----------
[1] M. Yang, S. Chen, C. Fu and P. Sheng, "Optimal sound-absorbing structures", Materials Horizons, vol. 4,
    pp. 673-680, 2017. Equation (1) and ESI section I, equations (S1) to (S10).

For further information check the function specific documentation.
"""
import numpy as np

# The 4*pi**2 factor of equation (S10). Kept as a module constant because it appears in every branch below.
FOUR_PI_SQUARED = 4 * np.pi ** 2

# Layer types whose static compressibility is not described by Wood's law with an air volume fraction.
UNSUPPORTED_POROSITY_TYPES = {
    "membrane_layer": "a membrane adds static compliance from its own tension and stiffness, which Wood's law "
                      "discards by assuming a rigid solid phase",
    "constant_z": "a constant-impedance termination carries no thickness or porosity metadata",
}

# Appended to every porosity error, since an override is the way past each of them.
BULK_MODULUS_OVERRIDE_HINT = (
    "Pass bulk_modulus_ratio to override: 1/porosity for a stack whose static compression is adiabatic, "
    "1/(gamma*porosity) for one that is isothermal. A membrane needs a value below both, since its own "
    "compliance is not described by Wood's law."
)

# Layer types whose pore scale puts the thermal transition inside the audio band, so their static bulk modulus is
# the isothermal P0/phi rather than the adiabatic B0/phi. An open air layer stays adiabatic across the audio band
# because its thermal transition sits near a millihertz.
ISOTHERMAL_LAYER_TYPES = {"porous_layer", "porous_facing_layer"}

VALID_TAILS = ("model", "none")
VALID_SOLVE_FOR = ("frequency", "depth", "absorption")
VALID_THERMAL = ("isothermal", "adiabatic")


def _trapezoidal_integral(y, x):
    """Integrate by the trapezoidal rule across NumPy 1.x and 2.x APIs."""
    integrator = getattr(np, "trapezoid", None)
    if integrator is None:
        integrator = np.trapz
    return integrator(y, x)


def effective_porosity(matrix):
    """
    Return the thickness-weighted mean porosity of a treatment and the per-layer breakdown.

    Static compliances add in series, ``d / Beff = sum(d_i / Beff_i)``, so with ``Beff_i = B0 / phi_i`` the stack
    behaves as ``Beff / B0 = d / sum(d_i * phi_i) = 1 / phi_eff``. The porosity of each layer is read from the
    metadata already stored by the layer builders: ``porosity`` for porous layers, ``open_area [%]`` for
    perforated and slotted panels, and unity for air layers.

    Parameters
    ----------
    matrix : dict
        A ``TMM.matrix`` dictionary. Entries without a ``'thickness [mm]'`` key, such as the backing record, are
        ignored.

    Returns
    -------
    Effective porosity and a list of per-layer dictionaries with the layer key, type, thickness and porosity.

    Raises
    ------
    ValueError
        If the stack contains a layer whose static compressibility is not described by Wood's law, or if no layer
        carries a thickness.
    """
    layers = []
    for key, value in matrix.items():
        if not isinstance(value, dict):
            continue

        layer_type = value.get("type")
        if layer_type == "backing":
            continue

        if layer_type in UNSUPPORTED_POROSITY_TYPES or key == "material_model":
            reason = UNSUPPORTED_POROSITY_TYPES.get(
                layer_type,
                "a spline-fitted material model exposes no layer geometry",
            )
            raise ValueError(
                f"Layer {key!r} of type {layer_type!r} has no Wood's-law porosity: {reason}. "
                f"{BULK_MODULUS_OVERRIDE_HINT}"
            )

        if "thickness [mm]" not in value:
            continue

        thickness = float(value["thickness [mm]"])
        porosity = _layer_porosity(key, value)
        layers.append({"layer": key, "type": layer_type, "thickness [mm]": thickness, "porosity": porosity})

    if not layers:
        raise ValueError("The treatment has no layers with a thickness, so an effective porosity cannot be built.")

    total_thickness = sum(layer["thickness [mm]"] for layer in layers)
    if total_thickness <= 0:
        raise ValueError("The treatment has zero total thickness, so an effective porosity cannot be built.")

    open_thickness = sum(layer["thickness [mm]"] * layer["porosity"] for layer in layers)

    return open_thickness / total_thickness, layers


def effective_bulk_modulus_ratio(matrix, thermal="isothermal", gamma=1.4):
    """
    Return the treatment's static ``Beff / B0`` and the per-layer breakdown.

    Static compliances add in series, ``d / Beff = sum(d_i / Beff_i)``, so with ``b_i = Beff_i / B0`` the stack
    behaves as ``Beff / B0 = d / sum(d_i / b_i)``. Each layer contributes ``b_i = 1 / phi_i`` when its compression
    is adiabatic and ``b_i = 1 / (gamma * phi_i)`` when it is isothermal, the latter being the true static limit
    of a medium whose thermal transition lies inside the audio band.

    The distinction is not cosmetic. A rigid-frame porous layer relaxes from the adiabatic ``gamma * P0 / phi`` at
    high frequency to the isothermal ``P0 / phi`` at zero frequency, and the causality sum rule is evaluated in the
    static limit. Assuming the adiabatic value for such a layer inflates ``d_min`` by a factor of ``gamma``, which
    is enough to make a perfectly ordinary absorber appear to violate the bound.

    Parameters
    ----------
    matrix : dict
        A ``TMM.matrix`` dictionary.
    thermal : string, optional
        ``'isothermal'`` applies the ``gamma`` factor to layers in ``ISOTHERMAL_LAYER_TYPES``. ``'adiabatic'``
        uses ``b_i = 1 / phi_i`` throughout, which is the assumption made in the source paper.
    gamma : float, optional
        Specific heat ratio of air.

    Returns
    -------
    Effective ``Beff / B0`` and a list of per-layer dictionaries including the ratio each layer contributed.
    """
    if thermal not in VALID_THERMAL:
        raise ValueError(f"thermal must be one of {VALID_THERMAL}.")
    if gamma <= 0:
        raise ValueError("gamma must be greater than zero.")

    _, layers = effective_porosity(matrix)

    total_thickness = 0.0
    total_compliance = 0.0
    for layer in layers:
        isothermal = thermal == "isothermal" and layer["type"] in ISOTHERMAL_LAYER_TYPES
        ratio = 1 / (layer["porosity"] * (gamma if isothermal else 1.0))
        layer["bulk_modulus_ratio"] = ratio
        layer["thermal"] = "isothermal" if isothermal else "adiabatic"
        total_thickness += layer["thickness [mm]"]
        total_compliance += layer["thickness [mm]"] / ratio

    return total_thickness / total_compliance, layers


def _layer_porosity(key, value):
    """Return the air volume fraction of a single layer entry."""
    layer_type = value.get("type")

    if layer_type in UNSUPPORTED_POROSITY_TYPES:
        raise ValueError(
            f"Layer {key!r} of type {layer_type!r} has no Wood's-law porosity: "
            f"{UNSUPPORTED_POROSITY_TYPES[layer_type]}. {BULK_MODULUS_OVERRIDE_HINT}"
        )

    if layer_type == "air_layer":
        return 1.0

    if value.get("porosity") is not None:
        return float(value["porosity"])

    if value.get("open_area [%]") is not None:
        return float(value["open_area [%]"]) / 100

    raise ValueError(
        f"Layer {key!r} of type {layer_type!r} carries neither 'porosity' nor 'open_area [%]', so its air "
        f"volume fraction is unknown. {BULK_MODULUS_OVERRIDE_HINT}"
    )


def _clamped_log_term(alpha, alpha_floor):
    """Return ``-ln(1 - A)`` with ``1 - A`` clamped away from zero, and the number of clamped samples."""
    residual = 1 - np.asarray(alpha, dtype=float)
    clamped = residual < alpha_floor
    residual = np.where(clamped, alpha_floor, residual)

    return -np.log(residual), int(np.count_nonzero(clamped))


def _low_frequency_tail(freq, log_term, fit_points):
    """
    Return the ``0`` to ``fmin`` contribution, its fitted exponent and any warning raised while fitting.

    As ``f -> 0`` the absorption of a passive rigid-backed layer vanishes as ``A ~ Rs(w) * w**2``, so
    ``-ln(1 - A) ~ K * f**p``. The exponent is fitted rather than assumed because it depends on how the surface
    resistance behaves at low frequency: a frequency-flat resistance gives ``p = 2`` while a Delany-Bazley or Miki
    characteristic impedance gives ``p`` near 1.37. The extrapolated integral
    ``integral from 0 to fmin of K * f**(p - 2) df = K * fmin**(p - 1) / (p - 1)`` converges only for ``p > 1``.
    """
    fmin = freq[0]
    if fmin <= 0:
        return 0.0, float("nan"), "The frequency vector starts at or below zero, so no low-frequency tail was added."

    # The window is a frequency ratio, not a fraction of the point count. On a linearly spaced band a tenth of the
    # points can span two decades, which is nowhere near the asymptotic regime the power law describes.
    if fit_points is None:
        window = max(int(np.count_nonzero(freq <= 2 * fmin)), 5)
    else:
        window = max(int(fit_points), 2)

    sample = slice(0, min(window, freq.size))
    freq_fit = freq[sample]
    term_fit = log_term[sample]

    usable = (freq_fit > 0) & (term_fit > 0) & np.isfinite(term_fit)
    if np.count_nonzero(usable) < 2:
        return 0.0, float("nan"), (
            "The lowest in-band absorption is zero or non-finite, so the low-frequency tail could not be fitted "
            "and was set to zero."
        )

    exponent, intercept = np.polyfit(np.log(freq_fit[usable]), np.log(term_fit[usable]), 1)
    if not np.isfinite(exponent) or exponent <= 1:
        return 0.0, float(exponent), (
            f"The low-frequency absorption exponent fitted to {exponent:.3f}, which makes the extrapolated "
            f"integral divergent. The low-frequency tail was set to zero, so d_min is underestimated."
        )

    coefficient = np.exp(intercept)

    return float(coefficient * fmin ** (exponent - 1) / (exponent - 1)), float(exponent), None


def _high_frequency_tail(freq, log_term):
    """
    Return the ``fmax`` to infinity contribution and the level it assumed.

    There is no asymptotic law for the absorption of a general structure at high frequency, so ``-ln(1 - A)`` is
    held at its mean over the top octave. An octave mean is used rather than the value at ``fmax`` so the result
    does not swing by several times depending on whether the band edge lands on an absorption peak or a dip.
    """
    fmax = freq[-1]
    if fmax <= 0:
        return 0.0, float("nan")

    top_octave = freq >= fmax / 2
    if np.count_nonzero(top_octave) < 2:
        level = float(log_term[-1])
    else:
        level = float(_trapezoidal_integral(log_term[top_octave], freq[top_octave])
                      / (freq[top_octave][-1] - freq[top_octave][0]))

    return level / fmax, level


def cumulative_thickness(freq, alpha, c0, bulk_modulus_ratio, alpha_floor=1e-6):
    """
    Return the minimum thickness accumulated up to each frequency, in millimetres.

    This is the in-band integral of ``causal_minimum_thickness`` accumulated rather than totalled, so its last
    value equals that function's ``d_min_in_band`` for the same inputs. Reading it against the treatment depth
    shows which part of the spectrum drives ``d_min``, which is rarely where the absorption looks best: the
    ``1 / f**2`` weight makes a mediocre low-frequency roll-off contribute far more minimum thickness than a
    flat high-frequency plateau.

    Parameters
    ----------
    freq : array
        Frequency vector in Hz, strictly increasing and positive.
    alpha : array
        Normal-incidence absorption coefficient of a rigid-backed treatment, same length as ``freq``.
    c0 : float
        Speed of sound in air [m/s].
    bulk_modulus_ratio : float
        Static ``Beff / B0`` of the treatment.
    alpha_floor : float, optional
        Lower clamp applied to ``1 - A``, matching ``causal_minimum_thickness``.

    Returns
    -------
    Array of the same length as ``freq``, starting at zero and ending at ``d_min_in_band``.
    """
    freq = np.asarray(freq, dtype=float).reshape(-1)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    if freq.size != alpha.size:
        raise ValueError("freq and alpha must have the same length.")
    if freq.size < 2:
        raise ValueError("freq must contain at least two points to integrate.")
    if np.any(freq <= 0):
        raise ValueError("freq must be strictly positive to integrate ln[1 - A(f)] / f**2.")
    if np.any(np.diff(freq) <= 0):
        raise ValueError("freq must be strictly increasing.")
    if bulk_modulus_ratio <= 0:
        raise ValueError("bulk_modulus_ratio must be greater than zero.")

    log_term, _ = _clamped_log_term(np.clip(alpha, 0.0, None), alpha_floor)
    integrand = log_term / freq ** 2
    scale = 1000 * c0 * bulk_modulus_ratio / FOUR_PI_SQUARED

    steps = np.diff(freq) * (integrand[1:] + integrand[:-1]) / 2

    return scale * np.concatenate(([0.0], np.cumsum(steps)))


def band_contributions(freq, cumulative, n_oct=1):
    """
    Split a cumulative minimum-thickness curve into fractional-octave bands.

    The band edges are clipped to the ends of ``freq``, so the returned contributions always sum to the last
    value of ``cumulative``. If that identity stops holding, the binning has run off the end of the frequency
    vector.

    Parameters
    ----------
    freq : array
        Frequency vector in Hz, matching the one used to build ``cumulative``.
    cumulative : array
        Accumulated minimum thickness from ``cumulative_thickness``.
    n_oct : int, optional
        Fractional octave resolution, following the ``TMM.filter_alpha`` convention. ``1`` gives octave bands
        and ``3`` gives third-octave bands.

    Returns
    -------
    Band edges in Hz and each band's contribution to ``d_min``, in millimetres. There is one more edge than
    band.
    """
    freq = np.asarray(freq, dtype=float).reshape(-1)
    cumulative = np.asarray(cumulative, dtype=float).reshape(-1)

    if freq.size != cumulative.size:
        raise ValueError("freq and cumulative must have the same length.")
    if n_oct < 1 or int(n_oct) != n_oct:
        raise ValueError("n_oct must be a positive whole number.")

    step = np.log10(2) / n_oct
    edges = np.clip(10 ** np.arange(np.log10(freq[0]), np.log10(freq[-1]) + step, step), freq[0], freq[-1])

    # The log round trip does not land exactly on the band ends: 10**log10(20) is 20.000000000000004, which
    # pushes searchsorted one sample inward and silently drops the first band's contribution. Pinning both ends
    # by index rather than by value is what keeps the contributions summing to the cumulative total.
    edges[0] = freq[0]
    edges[-1] = freq[-1]
    edges = np.unique(edges)
    if edges.size < 2:
        raise ValueError("The frequency vector is too narrow to split into bands at this n_oct.")

    index = np.searchsorted(freq, edges).clip(0, freq.size - 1)
    index[0] = 0
    index[-1] = freq.size - 1

    return edges, np.diff(cumulative[index])


def causal_minimum_thickness(freq, alpha, c0, bulk_modulus_ratio, tail="model", alpha_floor=1e-6,
                             low_f_fit_points=None):
    """
    Return the causality-limited minimum thickness for a normal-incidence, rigid-backed absorption spectrum.

    Evaluates ``d_min = (c0 / (4 * pi**2)) * (Beff / B0) * integral of -ln[1 - A(f)] / f**2 df`` as three separately
    reported pieces: the trapezoidal integral over the supplied band, an extrapolation below ``fmin`` and a tail
    above ``fmax``. The tails are reported rather than folded in silently because on a band ending at 5 kHz the
    high-frequency tail alone carries a noticeable share of ``d_min``.

    Parameters
    ----------
    freq : array
        Frequency vector in Hz, strictly increasing and positive.
    alpha : array
        Normal-incidence absorption coefficient of a rigid-backed treatment, same length as ``freq``.
    c0 : float
        Speed of sound in air [m/s].
    bulk_modulus_ratio : float
        Static ``Beff / B0`` of the treatment. For a stack of rigid-frame layers this is the reciprocal of the
        thickness-weighted mean porosity, with a further factor of ``1 / gamma`` on any layer whose static
        compression is isothermal. See ``effective_bulk_modulus_ratio``. The ratio is taken here rather than a
        porosity because a porosity cannot express a ratio below unity, which the isothermal limit and any
        membrane compliance both require.
    tail : string, optional
        ``'model'`` adds the fitted low-frequency and top-octave high-frequency tails. ``'none'`` returns the
        in-band integral only, which is a strict lower bound on ``d_min``.
    alpha_floor : float, optional
        Lower clamp applied to ``1 - A`` so a grid point landing on a perfect-absorption zero stays finite.
    low_f_fit_points : int, optional
        Number of lowest in-band points used to fit the low-frequency exponent. Defaults to the points within an
        octave of ``fmin``, with a floor of five. The window is a frequency ratio rather than a share of the
        point count, because on a linearly spaced band a fixed share of the points can span decades.

    Returns
    -------
    Dictionary with ``d_min`` and ``d_min_in_band`` in millimetres, the three integral pieces in seconds, the
    fitted low-frequency exponent, the assumed high-frequency level, the clamp count and any warnings raised.
    ``tail_fraction`` is the share of the reported ``d_min`` that the tails carry, so it is zero under
    ``tail='none'``; ``tail_fraction_modelled`` is the share they would carry either way, which is what says
    whether the band is wide enough.
    """
    if tail not in VALID_TAILS:
        raise ValueError(f"tail must be one of {VALID_TAILS}.")

    freq = np.asarray(freq, dtype=float).reshape(-1)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    if freq.size != alpha.size:
        raise ValueError("freq and alpha must have the same length.")
    if freq.size < 2:
        raise ValueError("freq must contain at least two points to integrate.")
    if np.any(freq <= 0):
        raise ValueError("freq must be strictly positive to integrate ln[1 - A(f)] / f**2.")
    if np.any(np.diff(freq) <= 0):
        raise ValueError("freq must be strictly increasing.")
    if bulk_modulus_ratio <= 0:
        raise ValueError("bulk_modulus_ratio must be greater than zero.")
    if not 0 < alpha_floor < 1:
        raise ValueError("alpha_floor must be greater than zero and less than one.")

    messages = []

    if np.any(alpha < 0):
        messages.append("The absorption spectrum contains negative values, which a passive treatment cannot "
                        "produce. They were clamped to zero before integrating.")
        alpha = np.clip(alpha, 0.0, None)

    log_term, n_clamped = _clamped_log_term(alpha, alpha_floor)
    if n_clamped:
        messages.append(f"{n_clamped} sample(s) reached the alpha_floor clamp at perfect absorption, so d_min is "
                        f"slightly underestimated there.")

    integrand = log_term / freq ** 2
    in_band = float(_trapezoidal_integral(integrand, freq))

    # The tails are evaluated whichever mode is asked for. Under 'none' they stay out of d_min, which is then a
    # strict lower bound, but their size is still the only thing that says whether the band was wide enough to
    # draw a conclusion from, so it is reported as tail_fraction_modelled for the diagnosis to read.
    low_f, exponent, low_f_message = _low_frequency_tail(freq, log_term, low_f_fit_points)

    # An extrapolation that outweighs everything actually computed is not evidence about the treatment, it is
    # evidence that the band does not reach far enough down for the asymptote to have set in.
    runaway_message = None
    if low_f > in_band:
        runaway_message = (
            f"The low-frequency extrapolation below {freq[0]:.3g} Hz came out larger than the whole in-band "
            f"integral, so the band does not reach low enough for the asymptote to hold. It was discarded and "
            f"d_min is underestimated. Recompute with a lower fmin."
        )
        low_f = 0.0

    high_f, level = _high_frequency_tail(freq, log_term)
    modelled = in_band + low_f + high_f
    modelled_fraction = (low_f + high_f) / modelled if modelled > 0 else 0.0

    if tail == "model":
        messages.extend(message for message in (low_f_message, runaway_message) if message is not None)
    else:
        low_f = 0.0
        high_f = 0.0
        exponent = float("nan")
        level = float("nan")

    scale = 1000 * c0 * bulk_modulus_ratio / FOUR_PI_SQUARED  # Millimetres, to match TMM.depth.
    total = in_band + low_f + high_f

    return {
        "d_min": scale * total,
        "d_min_in_band": scale * in_band,
        "tail": tail,
        "integral": {"in_band": in_band, "low_f": low_f, "high_f": high_f},
        "tail_fraction": (low_f + high_f) / total if total > 0 else 0.0,
        "tail_fraction_modelled": modelled_fraction,
        "low_f_exponent": exponent,
        "high_f_level": level,
        "freq_range": (float(freq[0]), float(freq[-1])),
        "n_clamped": n_clamped,
        "warnings": messages,
    }


def causality_limit(solve_for, c0, bulk_modulus_ratio, alpha_target=None, f_low=None, f_high=None, depth=None):
    """
    Solve the causality bound for an idealised flat absorption target.

    For a target that holds ``A0`` over ``[f_low, f_high]`` and vanishes elsewhere, the bound integrates in closed
    form to ``d = (c0 * L0 / (4 * pi**2)) * (Beff / B0) * (1 / f_low - 1 / f_high)`` with ``L0 = abs(ln(1 - A0))``. Any one
    of the three quantities follows from the other two, which is the thickness, bandwidth and absorption-magnitude
    tradeoff described in the source paper.

    This describes an idealised step spectrum, not the absorption of an actual stack. The result reads as "no
    causal, passive, rigid-backed structure of this thickness and porosity can hold a flat ``A0`` below
    ``f_low``", never as "this treatment works down to ``f_low``".

    Parameters
    ----------
    solve_for : string
        ``'frequency'``, ``'depth'`` or ``'absorption'``.
    c0 : float
        Speed of sound in air [m/s].
    bulk_modulus_ratio : float
        Static ``Beff / B0`` of the treatment.
    alpha_target : float, optional
        Flat absorption coefficient held across the band. Required unless solving for it.
    f_low : float, optional
        Lower band edge in Hz. Required unless solving for it.
    f_high : float or None, optional
        Upper band edge in Hz. ``None`` means the target extends to infinite frequency.
    depth : float, optional
        Treatment thickness in millimetres. Required unless solving for it.

    Returns
    -------
    Dictionary with the solved quantity and every input actually used, so a printed result states its own
    assumptions.
    """
    if solve_for not in VALID_SOLVE_FOR:
        raise ValueError(f"solve_for must be one of {VALID_SOLVE_FOR}.")
    if bulk_modulus_ratio <= 0:
        raise ValueError("bulk_modulus_ratio must be greater than zero.")

    if f_high is not None:
        if f_high <= 0:
            raise ValueError("f_high must be greater than zero.")
        if f_low is not None and f_high <= f_low:
            raise ValueError("f_high must be greater than f_low.")
    inverse_f_high = 0.0 if f_high is None else 1 / f_high

    scale = 1000 * c0 * bulk_modulus_ratio / FOUR_PI_SQUARED  # Millimetres, to match TMM.depth.

    if solve_for == "frequency":
        _require(alpha_target=alpha_target, depth=depth)
        level = _absorption_level(alpha_target)
        inverse_f_low = depth / (scale * level) + inverse_f_high
        if inverse_f_low <= 0:
            raise ValueError("The requested thickness and target cannot be met at any positive frequency.")
        f_low = 1 / inverse_f_low

    elif solve_for == "depth":
        _require(alpha_target=alpha_target, f_low=f_low)
        if f_low <= 0:
            raise ValueError("f_low must be greater than zero.")
        level = _absorption_level(alpha_target)
        depth = scale * level * (1 / f_low - inverse_f_high)

    else:
        _require(f_low=f_low, depth=depth)
        if f_low <= 0:
            raise ValueError("f_low must be greater than zero.")
        if depth <= 0:
            raise ValueError("depth must be greater than zero.")
        level = depth / (scale * (1 / f_low - inverse_f_high))
        alpha_target = float(-np.expm1(-level))

    return {
        "solve_for": solve_for,
        "alpha_target": float(alpha_target),
        "f_low": float(f_low),
        "f_high": None if f_high is None else float(f_high),
        "depth": float(depth),
        "bulk_modulus_ratio": float(bulk_modulus_ratio),
        "c0": float(c0),
    }


def _require(**arguments):
    """Raise if any of the named arguments needed by a ``causality_limit`` branch is missing."""
    missing = sorted(name for name, value in arguments.items() if value is None)
    if missing:
        raise ValueError(f"{', '.join(missing)} must be supplied for this solve_for branch.")


def _absorption_level(alpha_target):
    """Return ``L0 = abs(ln(1 - A0))`` for a flat absorption target."""
    if not 0 < alpha_target < 1:
        raise ValueError("alpha_target must be greater than zero and less than one.")

    return float(-np.log1p(-alpha_target))


def _contiguous_bands(freq, mask):
    """Return the frequency spans covered by contiguous runs of a boolean mask."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []

    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]]))

    return [(float(freq[start]), float(freq[end])) for start, end in zip(starts, ends)]


def passivity_report(freq, alpha, tolerance=0.0):
    """
    Report whether an absorption spectrum is physically realizable.

    A passive treatment absorbs between none and all of the incident energy, so ``0 <= A <= 1`` everywhere. A
    spectrum that leaves that range describes a treatment that either creates energy or reflects more than it
    receives, which means the layer model has been pushed outside its valid range. The empirical Delany-Bazley
    family is the usual cause, returning negative absorption at low ``f / sigma`` where its curve fits no longer
    hold.

    This matters for the causality bound because the derivation assumes passivity. A non-passive spectrum does not
    merely fail the bound, it falls outside the bound's premises entirely.

    Parameters
    ----------
    freq : array
        Frequency vector in Hz.
    alpha : array
        Absorption coefficient, same length as ``freq``.
    tolerance : float, optional
        Slack allowed outside ``[0, 1]`` before a sample counts as non-passive, for absorbing float noise.

    Returns
    -------
    Dictionary with ``passive``, the observed ``alpha_range``, the contiguous ``bands [Hz]`` that offend, and the
    number of offending samples.
    """
    freq = np.asarray(freq, dtype=float).reshape(-1)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    if freq.size != alpha.size:
        raise ValueError("freq and alpha must have the same length.")
    if tolerance < 0:
        raise ValueError("tolerance must not be negative.")

    offending = (alpha < -tolerance) | (alpha > 1 + tolerance)

    return {
        "passive": not bool(offending.any()),
        "alpha_range": (float(np.min(alpha)), float(np.max(alpha))),
        "bands [Hz]": _contiguous_bands(freq, offending),
        "n_offending": int(np.count_nonzero(offending)),
        "tolerance": float(tolerance),
    }


def _describe_bands(bands, limit=3):
    """Return a compact human-readable summary of frequency spans."""
    shown = ", ".join(f"{low:.4g}-{high:.4g} Hz" for low, high in bands[:limit])
    if len(bands) > limit:
        shown += f" and {len(bands) - limit} more"

    return shown


def _truncation_phrase(result, modelled_fraction):
    """Return a phrase for how much of ``d_min`` the out-of-band tails account for, in the caller's tail mode."""
    share = f"{100 * modelled_fraction:.1f}%"
    if result.get("tail", "model") == "model":
        return f"{share} of d_min comes from the out-of-band tails rather than from the computed band"

    return f"the band is narrow enough that the tails this result leaves out would have carried {share} of d_min"


def diagnose(result, depth, bulk_modulus_ratio, freq, alpha, available_ratios=None, tail_threshold=0.15):
    """
    Explain a causality result, naming the cause of a violated bound rather than listing candidates.

    No passive rigid-backed structure can exceed the bound, so ``d_min > d`` always means an assumption broke. The
    evidence needed to tell the candidates apart is already available, so this returns the one cause it finds
    rather than the whole list:

    - A non-passive spectrum settles it by itself: the bound assumes passivity, so it does not apply at all.
    - ``d_min_in_band > depth`` rules out the out-of-band tails, because the computed part alone already exceeds
      the depth. The converse points straight at the extrapolation.
    - The static bulk modulus needed to satisfy the bound is a number, ``bulk_modulus_ratio / ratio``, which can be
      compared against the values this stack actually supports.

    Clamping is deliberately not a candidate. It only ever lowers ``d_min``, so it cannot cause an overshoot.

    A satisfied bound is reported on as well, because ``ratio <= 1`` is not by itself evidence that the number is
    trustworthy. When the out-of-band tails carry more than ``tail_threshold`` of ``d_min``, the verdict rests on
    extrapolation rather than on the computed band, and the report says so whichever side of the bound it lands
    on. The test reads ``tail_fraction_modelled`` rather than ``tail_fraction``, so it still fires under
    ``tail='none'``, where the tails are excluded from ``d_min`` but their size is exactly what says whether the
    band was wide enough.

    Parameters
    ----------
    result : dict
        The dictionary returned by ``causal_minimum_thickness``.
    depth : float
        Treatment thickness in millimetres.
    bulk_modulus_ratio : float
        The static ``Beff / B0`` used for the result.
    freq, alpha : array
        The spectrum the result was built from.
    available_ratios : dict, optional
        Static ``Beff / B0`` values this stack supports, keyed by thermal assumption, used to judge whether the
        bulk modulus choice explains a violation.
    tail_threshold : float, optional
        Share of ``d_min`` above which the out-of-band extrapolation is considered untrustworthy.

    Returns
    -------
    The diagnosis dictionary and a message naming the leading finding, or ``None`` when the bound is satisfied by
    a result whose tails are small enough to trust.
    """
    available_ratios = {name: float(value) for name, value in dict(available_ratios or {}).items()}
    modelled_fraction = float(result.get("tail_fraction_modelled", result["tail_fraction"]))
    ratio = result["d_min"] / depth if depth > 0 else float("inf")
    passivity = passivity_report(freq, alpha)
    in_band_alone_violates = bool(result["d_min_in_band"] > depth)
    required = bulk_modulus_ratio / ratio if ratio > 0 and np.isfinite(ratio) else float("nan")

    diagnosis = {
        "bound_satisfied": bool(ratio <= 1),
        "ratio": float(ratio),
        "excess [mm]": float(max(0.0, result["d_min"] - depth)),
        "passivity": passivity,
        "in_band_alone_violates": in_band_alone_violates,
        "tail_fraction": float(result["tail_fraction"]),
        "tail_fraction_modelled": modelled_fraction,
        "required_bulk_modulus_ratio": float(required),
        "available_bulk_modulus_ratios": available_ratios,
        "causes": [],
    }

    if ratio <= 1:
        if not passivity["passive"]:
            diagnosis["causes"].append("non_passive_spectrum")

        # If most of d_min was extrapolated rather than computed, the headroom being reported is headroom
        # against a number the band was too narrow to establish.
        if modelled_fraction > tail_threshold:
            diagnosis["causes"].append("band_truncation")
            return diagnosis, (
                f"The causality constraint is satisfied at ratio {ratio:.3g}, but "
                f"{_truncation_phrase(result, modelled_fraction)}, so the verdict rests on extrapolation. "
                f"Recompute over a wider band before relying on it. A log-spaced frequency vector keeps the "
                f"point count down: a band of roughly 7 Hz to 20 kHz brings both tails under 2% with fewer "
                f"points than the default 1 Hz grid needs for two decades."
            )

        return diagnosis, None

    headline = (f"d_min exceeds the treatment depth by {100 * (ratio - 1):.1f}% "
                f"({diagnosis['excess [mm]']:.1f} mm), which no passive rigid-backed structure can do.")

    if not passivity["passive"]:
        diagnosis["causes"].append("non_passive_spectrum")
        return diagnosis, (
            f"{headline} The computed spectrum is not passive: absorption reaches "
            f"{passivity['alpha_range'][0]:.3g} over {_describe_bands(passivity['bands [Hz]'])}, so the bound's "
            f"premises do not hold for it. The empirical Delany-Bazley family is not causal at low f/sigma and is "
            f"the usual cause."
        )

    if not in_band_alone_violates or modelled_fraction > tail_threshold:
        diagnosis["causes"].append("band_truncation")
        if in_band_alone_violates:
            detail = _truncation_phrase(result, modelled_fraction)
        else:
            detail = (f"the in-band integral alone stays under the depth, so the out-of-band tails are what "
                      f"pushed it over ({100 * modelled_fraction:.1f}% of d_min)")
        return diagnosis, (
            f"{headline} However, {detail}. Recompute over a wider band before drawing a conclusion."
        )

    workable = {name: value for name, value in available_ratios.items() if value <= required}
    if workable:
        name, value = min(workable.items(), key=lambda item: item[1])
        diagnosis["causes"].append("bulk_modulus_assumption")
        return diagnosis, (
            f"{headline} Satisfying it needs Beff/B0 <= {required:.4g} against the {bulk_modulus_ratio:.4g} used. "
            f"This stack's {name} value is {value:.4g}, so thermal={name!r} would resolve it."
        )

    diagnosis["causes"].append("layer_model")
    return diagnosis, (
        f"{headline} The in-band integral alone exceeds the depth, the tails are a small share, and no static "
        f"bulk modulus this stack supports would satisfy it (needs Beff/B0 <= {required:.4g}). The layer model is "
        f"the remaining candidate."
    )
