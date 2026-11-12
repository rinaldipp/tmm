"""
Toolbox for design and prediction of multilayered acoustic treatments.
Also contains a material model based on the GRAS database.

Developed by Rinaldi Petrolli.
For questions about usage, bugs, licensing and/or contributions contact rinaldipp@gmail.com.

References
----------
[1] R. Petrolli, A. Zorzo and P. D'Antonio, "Comparison of measurement and prediction for acoustical treatments
    designed with Transfer Matrix Models", in Euronoise, October 2021.

For further information check the function specific documentation.
"""
import collections
import copy
import functools
import os
import warnings

import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy import integrate, special
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from scipy.signal import butter, freqz, savgol_filter
from scipy.special import jv

from tmm import _causality as causality
from tmm import _h5utils as h5utils
from tmm import _plot as plot
from tmm import _utils as utils
from tmm._reference_data import (  # noqa: F401  (re-exported for tmm.tmm imports)
    ALPHA_W_CLASSES, ALPHA_W_REFERENCE, ALPHA_W_THIRD_OCTAVE_INDICES, JEONG_ANGLE_WEIGHTS, JEONG_WEIGHT_ANGLES,
    JEONG_WEIGHT_BANDS, POROUS_PRESETS, POROUS_VALIDITY, PRESET_PARAMETERS, _FullTextFrame,
)
from tmm.database.path import path as database_path

plot.set_style()

# Cap on the rounding precision _grid_decimals derives for a frequency grid. Twelve decimals is far finer
# than any real grid needs, so it only matters for a degenerate grid, which is then left effectively unrounded.
MAX_GRID_DECIMALS = 12

# Truncation of the series in rectangular_slot_layer: the Stinson-Champoux density sum over the slot-length index
# (the width index is summed in closed form), and the Ingard end-correction sum over indices 0 to this value.
RECT_SLOT_DENSITY_TERMS = 1000
RECT_SLOT_END_CORRECTION_TERMS = 180

# Equivalent-fluid models built from the Johnson et al. dynamic tortuosity, and the frame options of porous_layer.
JCA_MODELS = ("jca", "jcal", "jcapl")
POROUS_FRAMES = ("rigid", "limp")

# porous_layer arguments that log_rebuild writes when set, paired with their self.matrix record keys.
POROUS_LOG_KEYS = (
    ("viscous_length", "viscous_length [µm]"),
    ("thermal_length", "thermal_length [µm]"),
    ("thermal_permeability", "thermal_permeability [m²]"),
    ("static_tortuosity", "static_tortuosity"),
    ("static_thermal_tortuosity", "static_thermal_tortuosity"),
    ("bulk_density", "bulk_density [kg/m³]"),
    ("pore_shape_factor", "pore_shape_factor"),
    ("perforation_diameter", "perforation_diameter [mm]"),
    ("perforation_spacing", "perforation_spacing [mm]"),
)


class TMM:
    """
    Transfer Matrix Method model for multilayer acoustic treatments.

    A ``TMM`` object defines a frequency grid, incidence condition, reference areas, and a front-to-rear layer stack.
    Layers are added in physical order from the incident face toward the rear termination, then ``compute()`` evaluates
    surface impedance, admittance, and absorption for the selected backing condition.

    The class also provides plotting, HDF5 persistence, spreadsheet/CSV export, and fractional-octave post-processing
    helpers. ``display_name`` and ``color`` are cosmetic metadata used by plots and reports; they do not change the
    transfer-matrix calculation.
    """
    def __init__(self, fmin=20, fmax=5000, df=1, incidence="diffuse", incidence_angle=None, project_folder=None,
                 filename=None, display_name=None, color=None, x_scale="lin", diffuse_method="field", s0=1.0,
                 srad=None, freq=None):
        """
        Parameters
        ----------
        fmin : float, optional
            Minimum frequency of interest.
        fmax : float, optional
            Maximum frequency of interest.
        df : float, optional
            Sets how many points the grid has, as ``int((fmax - fmin) / df) + 1``. On a linear ``x_scale`` that is the
            frequency resolution in Hz. On a logarithmic one it only sets the point count: the spacing is geometric,
            running from 0.02 Hz to 5.5 Hz across the default 20 Hz to 5 kHz band. It does not apply while an explicit
            ``freq`` vector is set.
        incidence : string, optional
            ``'normal'`` for normal incidence, ``'angle'`` for a single oblique angle, or ``'diffuse'`` for a
            diffuse-field approximation.
        incidence_angle : list of float, optional
            Minimum angle, maximum angle and step, in degrees.
        project_folder : string, optional
            Path to which files will be saved. If None is passed the current directory will be used.
        filename : string, optional
            Filename that will be used to save data and image files.
        display_name : string, optional
            Human-readable treatment name used in reports and plots. This is independent from ``filename``.
        color : string, optional
            Matplotlib color specification used for cosmetic plotting and reporting only.
        x_scale : string, optional
            Spacing law of the grid built from ``fmin``/``fmax``/``df``: ``'lin'`` for a linear grid or ``'log'`` for a
            logarithmic one. It does not affect plotting, and it does not apply while an explicit ``freq`` vector is
            set.
        diffuse_method : string, optional
            Method used when ``incidence='diffuse'``. ``'field'`` averages admittance over the configured angles into a
            field-incidence impedance, see ``field_impedance()``. ``'paris'`` averages angle-wise absorption into a
            statistical diffuse absorption coefficient, see ``diffuse_absorption_coefficient()``.
        s0 : float, optional
            Front/reference area in square meters used by the volume-velocity transfer matrices. The default value of
            1.0 keeps impedances in specific-acoustic-impedance form.
        srad : float, optional
            Rear termination area in square meters. For ``backing='radiation'`` this is interpreted as the area of the
            circular radiating aperture. If omitted, ``srad`` defaults to ``s0``.
        freq : array_like, optional
            Explicit frequency vector in Hz. When provided it owns the grid: ``fmin``/``fmax`` report its endpoints
            while ``df`` and ``x_scale`` report ``None``. The spacing may be non-uniform because the TMM evaluates every
            frequency independently. When omitted, the grid built from ``fmin``/``fmax``/``df`` is used. The four
            derived-grid parameters are retained either way, so assigning ``freq = None`` later restores the grid this
            constructor described.
        """
        if diffuse_method not in {"paris", "field"}:
            raise ValueError("diffuse_method must be 'paris' or 'field'.")
        incidence = self._validate_incidence(incidence)
        if incidence_angle is None:
            incidence_angle = self._default_diffuse_incidence_angle(diffuse_method)
        incidence_angle = self._validate_incidence_angle(incidence_angle, incidence)
        if srad is None:
            srad = s0

        self._df = df
        self._fmin = fmin
        self._fmax = fmax
        self._freq = None if freq is None else self._as_frequency_vector(freq)
        self._s0 = self._validate_positive_area(s0, "s0")
        self._srad = self._validate_positive_area(srad, "srad")
        self._air_prop = utils.AirProperties().standardized_c0_rho0()
        self._incidence = incidence
        self._incidence_angle = incidence_angle
        self._z = None
        self._z_angle = None
        self._z_angle_angles = None
        self._alpha = None
        self._scat = None
        self._matrix = {}
        self._layers_stale = False
        self._results_stale = False
        self._stale_reasons = []
        self._project_folder = self._validate_project_folder(project_folder)
        self._filename = self._validate_filename(filename)
        self._display_name = self._validate_display_name(display_name)
        self._color = color
        self._params = {}
        self._x_scale = self._validate_x_scale(x_scale)
        self._diffuse_method = diffuse_method

    def __repr__(self):
        return f"TMM_{self.filename}_{len(self.matrix) - 1}layers_{self.first_peak[0]:0.0f}Hz"

    def _has_model_data(self):
        """Return True when layer, termination, or material-model metadata is stored."""
        return bool(getattr(self, "_matrix", {}))

    def _has_computed_results(self):
        """Return True when cached impedance or absorption results are stored."""
        return (
            getattr(self, "_z", None) is not None
            or getattr(self, "_z_angle", None) is not None
            or getattr(self, "_alpha", None) is not None
        )

    def _record_stale_reason(self, reason):
        """Store a stale-state reason without duplicating repeated messages."""
        if not hasattr(self, "_stale_reasons"):
            self._stale_reasons = []
        if reason not in self._stale_reasons:
            self._stale_reasons.append(reason)

    def _mark_layers_stale(self, reason):
        """Mark stored transfer matrices and computed results as stale."""
        if not self._has_model_data():
            return
        self._layers_stale = True
        self._results_stale = True
        self._record_stale_reason(reason)
        warnings.warn(
            f"{reason} Stored transfer matrices are now stale. Call rebuild() before using computed results.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _mark_results_stale(self, reason):
        """Mark computed results as stale while keeping layer matrices valid."""
        if not self._has_computed_results():
            return
        self._results_stale = True
        self._record_stale_reason(reason)
        warnings.warn(
            f"{reason} Computed results are now stale. Call compute() or rebuild() before using them.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _clear_stale_state(self):
        """Clear stale-state markers after a successful explicit rebuild or compute."""
        self._layers_stale = False
        self._results_stale = False
        self._stale_reasons = []

    def _broadcast_to_incidence_angles(self, values):
        """Return a frequency vector as an ``(n_freq, n_angles)`` broadcast view."""
        values = np.asarray(values)
        n_freq = len(self.freq)
        if values.ndim != 1 or values.shape[0] != n_freq:
            raise ValueError(
                "values must be a one-dimensional frequency vector with length matching self.freq."
            )
        return np.broadcast_to(values[:, None], (n_freq, len(self.incidence_angle)))

    @staticmethod
    def _trapezoidal_integral(y, x, axis=-1):
        """Integrate by the trapezoidal rule across NumPy 1.x and 2.x APIs."""
        integrator = getattr(np, "trapezoid", None)
        if integrator is None:
            integrator = np.trapz
        return integrator(y, x, axis=axis)

    def _raise_if_stale_results(self, operation):
        """Raise when an operation would use stale layer matrices or computed results."""
        if getattr(self, "_layers_stale", False):
            reasons = "; ".join(getattr(self, "_stale_reasons", []))
            details = f" Reasons: {reasons}." if reasons else ""
            raise RuntimeError(f"{operation} cannot use stale layer matrices. Call rebuild() first.{details}")
        if getattr(self, "_results_stale", False):
            reasons = "; ".join(getattr(self, "_stale_reasons", []))
            details = f" Reasons: {reasons}." if reasons else ""
            raise RuntimeError(f"{operation} cannot use stale computed results. Call compute() or rebuild() first."
                               f"{details}")

    def _has_partial_z_angle(self):
        """Return True when stored ``z_angle`` columns do not span ``incidence_angle``."""
        return getattr(self, "_z_angle_angles", None) is not None

    def _stored_z_angle_angles(self):
        """Return the incidence angles that the stored ``z_angle`` columns correspond to."""
        retained = getattr(self, "_z_angle_angles", None)
        if retained is None:
            return np.asarray(self.incidence_angle, dtype=float)
        return np.asarray(retained, dtype=float)

    @staticmethod
    def _match_stored_angles(requested, available, atol=1e-6):
        """Return ascending, de-duplicated column indices for requested angles in degrees.

        The tolerance absorbs the float error of the ``linspace`` angular grid, including the value just below 90
        degrees that the diffuse range substitutes for grazing incidence.
        """
        indices = []
        for angle in requested:
            matches = np.flatnonzero(np.isclose(available, angle, rtol=0.0, atol=atol))
            if matches.size == 0:
                nearest = available[np.argmin(np.abs(available - angle))]
                raise ValueError(
                    f"{angle:g} deg is not one of this treatment's incidence angles ({available[0]:g} to "
                    f"{available[-1]:g} deg, {available.size} values). Nearest available is {nearest:g} deg."
                )
            indices.append(int(matches[0]))
        return np.unique(indices)

    def _raise_if_partial_z_angle(self, operation):
        """Raise when an operation needs every angle but only a reduced subset is stored."""
        if not self._has_partial_z_angle():
            return
        retained = ", ".join(f"{angle:g}" for angle in self._stored_z_angle_angles())
        raise RuntimeError(
            f"{operation} needs angle-dependent impedance for every angle in incidence_angle, but this object "
            f"came from reduced_copy() and only retains {retained} deg. Call rebuild() to restore the full "
            f"angular data first."
        )

    @staticmethod
    def _as_frequency_vector(freq):
        """Validate an explicit frequency vector (Hz): non-empty, 1-D, strictly ascending.
        Spacing may be non-uniform. TMM evaluates each frequency independently."""
        freq = np.asarray(freq, dtype=float).ravel()
        if freq.size == 0:
            raise ValueError("freq must contain at least one frequency in Hz.")
        if freq.size > 1 and np.any(np.diff(freq) <= 0):
            raise ValueError("freq must be strictly ascending.")
        return freq

    @staticmethod
    def _grid_decimals(grid):
        """Return a rounding precision ten times finer than the tightest spacing in ``grid``.

        Rounding keeps the stored frequencies tidy, but a fixed precision cannot serve every grid: one decimal suits the
        default 1 Hz linear grid and collapses anything finer into duplicate values, which is a silently shortened grid
        rather than an error. Deriving the precision from the grid instead bounds the displacement of any point at 5% of
        its local spacing, which is what one decimal already gives at 1 Hz, so grids at or above that spacing keep the
        values they have always had.

        A grid whose points coincide has no spacing to derive a precision from, so it is left unrounded rather than
        collapsed further.
        """
        if grid.size < 2:
            return 1

        spacing = float(np.diff(grid).min())
        if not np.isfinite(spacing) or spacing <= 0:
            return MAX_GRID_DECIMALS

        return int(np.clip(np.ceil(-np.log10(spacing)) + 1, 1, MAX_GRID_DECIMALS))

    @staticmethod
    def _validate_positive_area(value, name):
        """Return ``value`` as a positive scalar area."""
        try:
            array_value = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a positive scalar area.") from None
        if array_value.shape != ():
            raise ValueError(f"{name} must be a positive scalar area.")
        scalar_value = float(array_value)
        if not np.isfinite(scalar_value) or scalar_value <= 0:
            raise ValueError(f"{name} must be a positive scalar area.")
        return scalar_value

    @staticmethod
    def _validate_incidence(value):
        """Return a supported incidence mode."""
        if value not in {"normal", "angle", "diffuse"}:
            raise ValueError("incidence must be 'normal', 'angle', or 'diffuse'.")
        return value

    @staticmethod
    def _validate_display_name(value):
        """Return an optional human-readable display name."""
        if value is None or isinstance(value, str):
            return value
        raise TypeError("display_name must be a string or None.")

    @staticmethod
    def _validate_x_scale(value):
        """Return a supported frequency-axis scale."""
        if value not in {"lin", "log"}:
            raise ValueError("x_scale must be 'lin' or 'log'.")
        return value

    @staticmethod
    def _validate_filename(value):
        """Return an optional file stem used for TMM output files."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("filename must be a string or None.")
        if value.strip() == "":
            raise ValueError("filename cannot be empty.")
        invalid_chars = set('<>:"|?*')
        separators = {"/", "\\"}
        if os.sep:
            separators.add(os.sep)
        if os.altsep:
            separators.add(os.altsep)
        if any(char in value for char in invalid_chars | separators) or any(ord(char) < 32 for char in value):
            raise ValueError("filename must be a file stem without path separators or invalid filename characters.")
        return value

    @staticmethod
    def _validate_project_folder(value):
        """Return an optional project folder path string."""
        if value is None:
            return None
        try:
            folder = os.fspath(value)
        except TypeError:
            raise TypeError("project_folder must be a path-like value or None.") from None
        if not isinstance(folder, str):
            raise TypeError("project_folder must resolve to a string path.")
        if folder.strip() == "":
            raise ValueError("project_folder cannot be empty.")
        return folder

    @staticmethod
    def _default_diffuse_incidence_angle(diffuse_method):
        """Return the default angular range for the selected diffuse method."""
        return [0, 90, 1] if diffuse_method == "paris" else [0, 78, 1]

    @staticmethod
    def _validate_incidence_angle(value, incidence):
        """Return incidence-angle metadata compatible with the current incidence mode."""
        if value is None:
            return [0]
        try:
            values = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            raise ValueError("incidence_angle must contain finite numeric values.") from None

        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("incidence_angle must contain finite numeric values.")

        if incidence == "diffuse":
            if values.size != 3:
                raise ValueError("diffuse incidence_angle must be [start, stop, step].")
            start, stop, step = values
            if step <= 0:
                raise ValueError("diffuse incidence_angle step must be greater than zero.")
            if stop <= start:
                raise ValueError("diffuse incidence_angle stop must be greater than start.")
            if start < 0 or stop > 90:
                raise ValueError("diffuse incidence_angle must stay within 0 to 90 degrees.")
        elif incidence == "angle":
            angle = values[0]
            if angle < 0 or angle >= 90:
                raise ValueError("angle incidence_angle must be greater than or equal to 0 and less than 90 degrees.")

        return values.tolist()

    def _refuse_derived_grid_change(self, name):
        """Raise when a derived-grid parameter is assigned while an explicit vector owns the grid.

        ``fmin``, ``fmax``, ``df`` and ``x_scale`` describe the grid built from them, which an explicit ``freq`` vector
        replaces. Assigning to one of them there would edit a specification that is not in force, changing what the
        treatment reports without changing what it computes.
        """
        if self._freq is not None:
            raise AttributeError(
                f"{name} does not control the frequency grid while an explicit freq vector is set, so assigning "
                f"to it would change what the treatment reports without changing what it computes. Assign freq "
                f"with the band you want, or set freq = None to go back to the derived grid."
            )

    @property
    def fmin(self):
        """Return minimum frequency of analysis, from the explicit vector when one is set."""
        if self._freq is not None:
            return float(self._freq[0])

        return self._fmin

    @fmin.setter
    def fmin(self, new_fmin):
        """Set new minimum frequency value."""
        self._refuse_derived_grid_change("fmin")
        self._fmin = new_fmin
        self._mark_layers_stale("fmin was changed after model data existed.")

    @property
    def fmax(self):
        """Return maximum frequency of analysis, from the explicit vector when one is set."""
        if self._freq is not None:
            return float(self._freq[-1])

        return self._fmax

    @fmax.setter
    def fmax(self, new_fmax):
        """Set new maximum frequency value."""
        self._refuse_derived_grid_change("fmax")
        self._fmax = new_fmax
        self._mark_layers_stale("fmax was changed after model data existed.")

    @property
    def df(self):
        """Return the parameter that sets the derived grid's density, or ``None`` while a vector owns the grid.

        ``df`` fixes the point count as ``int((fmax - fmin) / df) + 1``, which is the frequency resolution in Hz on a
        linear ``x_scale``, while on a logarithmic one it sets nothing beyond the number of points. A vector has no such
        parameter behind it, and a non-uniform one has no single spacing to report, so there is no honest number to give
        here. The value passed to the constructor is retained and comes back when the vector is cleared.
        """
        if self._freq is not None:
            return None

        return self._df

    @df.setter
    def df(self, new_df):
        """Set new frequency resolution value."""
        self._refuse_derived_grid_change("df")
        self._df = new_df
        self._mark_layers_stale("df was changed after model data existed.")

    @property
    def freq(self):
        """Return frequency values."""
        if self._freq is not None:
            return self._freq

        points = int((self._fmax - self._fmin) / self._df) + 1
        if self._x_scale == "lin":
            grid = np.linspace(self._fmin, self._fmax, points)
        else:
            grid = np.logspace(np.log10(self._fmin), np.log10(self._fmax), points)

        return grid.round(self._grid_decimals(grid))

    @freq.setter
    def freq(self, new_freq):
        """Set an explicit frequency vector (Hz), or ``None`` to return to the fmin/fmax/df grid.

        Spacing may be non-uniform. The derived-grid parameters are left untouched, so clearing the vector restores the
        grid they describe rather than some mixture of the two.
        """
        self._freq = None if new_freq is None else self._as_frequency_vector(new_freq)
        self._mark_layers_stale("freq was changed after model data existed.")

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
        self._s0 = self._validate_positive_area(new_s0, "s0")
        self._mark_layers_stale("s0 was changed after model data existed.")

    @property
    def srad(self):
        """Return device rear termination area."""
        return self._srad

    @srad.setter
    def srad(self, new_srad):
        """Set device rear termination area."""
        self._srad = self._validate_positive_area(new_srad, "srad")
        self._mark_results_stale("srad was changed after results were computed.")

    @property
    def z0(self):
        """Return air impedance."""
        return self.rho0 * self.c0

    @property
    def z(self):
        """Return surface impedance."""
        self._raise_if_stale_results("z")
        if self._z is not None:
            return self._z
        else:
            return np.zeros_like(self.freq, dtype="complex")

    @z.setter
    def z(self, new_z):
        """Set surface impedance."""
        self._z = new_z
        self._alpha = None

    @property
    def z_angle(self):
        """Return angle-dependent surface impedance."""
        self._raise_if_stale_results("z_angle")
        if self._z_angle is not None:
            return self._z_angle
        else:
            return np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

    @z_angle.setter
    def z_angle(self, new_z_angle):
        """Set angle-dependent surface impedance."""
        self._z_angle = new_z_angle
        self._z_angle_angles = None

    @property
    def stored_angles(self):
        """
        Return the incidence angles, in degrees, for which angle-dependent impedance is actually stored.

        This matches ``incidence_angle`` for a normally computed treatment. On an object returned by ``reduced_copy()``
        it reports only the retained angles, and it is empty when ``z_angle`` was discarded. Column ``i`` of ``z_angle``
        corresponds to ``stored_angles[i]``, so this is the honest answer to "which angles does this object still have?"
        for a trimmed treatment.
        """
        if self._z_angle is None:
            return np.empty(0, dtype=float)
        return self._stored_z_angle_angles()

    @property
    def y(self):
        """Return admittance."""
        return 1 / self.z

    @property
    def alpha(self):
        """Return absorption coefficient."""
        self._raise_if_stale_results("alpha")
        if self._alpha is not None:
            return self._alpha.reshape((len(self._alpha),))
        _, alpha = self.reflection_and_absorption_coefficient(self.z)
        return alpha.reshape((len(alpha),))

    @property
    def diffuse_method(self):
        """Return the diffuse-incidence calculation method."""
        return self._diffuse_method

    @diffuse_method.setter
    def diffuse_method(self, method):
        """Set diffuse method: ``'field'`` for impedance averaging or ``'paris'`` for statistical absorption."""
        if method not in {"paris", "field"}:
            raise ValueError("diffuse_method must be 'paris' or 'field'.")
        self._diffuse_method = method
        self._alpha = None
        self._mark_results_stale("diffuse_method was changed after results were computed.")

    @property
    def x_scale(self):
        """Return frequency-grid spacing, or ``None`` while an explicit vector owns the grid.

        ``'lin'`` and ``'log'`` describe how the grid built from ``fmin``/``fmax``/``df`` is spaced. A vector carries
        its own spacing, which need not be uniform in frequency or in its logarithm, so there is no law to report here.
        The value passed to the constructor is retained and comes back when the vector is cleared.
        """
        if self._freq is not None:
            return None

        return self._x_scale

    @x_scale.setter
    def x_scale(self, new_x_scale):
        """Set frequency-grid spacing to ``'lin'`` or ``'log'``."""
        self._refuse_derived_grid_change("x_scale")
        self._x_scale = self._validate_x_scale(new_x_scale)
        self._mark_layers_stale("x_scale was changed after model data existed.")

    @property
    def first_peak(self):
        """Return the frequency in Hz and the absorption coefficient of the first meaningful absorption peak."""
        alpha = np.asarray(self.alpha)
        freq = np.asarray(self.freq)

        if len(alpha) < 3:
            peak_idx = int(np.nanargmax(alpha))
            return freq[peak_idx], alpha[peak_idx]

        previous_alpha = alpha[:-2]
        peak_alpha = alpha[1:-1]
        next_alpha = alpha[2:]
        local_maxima = np.flatnonzero((peak_alpha >= previous_alpha) & (peak_alpha > next_alpha)) + 1
        meaningful_maxima = local_maxima[alpha[local_maxima] >= 0.3]

        if meaningful_maxima.size:
            peak_idx = int(meaningful_maxima[0])
        elif local_maxima.size:
            peak_idx = int(local_maxima[np.nanargmax(alpha[local_maxima])])
        else:
            peak_idx = int(np.nanargmax(alpha))

        return freq[peak_idx], alpha[peak_idx]

    @property
    def scat(self):
        """
        Return the scattering coefficient, or ``None`` when none has been set.

        ``material_model()`` sets it from the GRAS third-octave scattering data of the bundled CSV file,
        spline-interpolated onto ``self.freq``. It can also be assigned directly, for example from a value measured or
        computed elsewhere.
        """
        return self._scat

    @scat.setter
    def scat(self, new_scat):
        """Set the scattering coefficient on ``self.freq``."""
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

    @incidence.setter
    def incidence(self, new_incidence):
        """Set incidence mode and mark existing layer matrices stale."""
        new_incidence = self._validate_incidence(new_incidence)
        if new_incidence == "diffuse":
            try:
                self._validate_incidence_angle(self._incidence_angle, new_incidence)
            except ValueError:
                self._incidence_angle = self._default_diffuse_incidence_angle(self.diffuse_method)
        self._incidence = new_incidence
        self._alpha = None
        self._mark_layers_stale("incidence was changed after model data existed.")

    @property
    def incidence_angle(self):
        """Return incidence angle values."""
        if self.incidence == "diffuse":
            start, stop, step = self._incidence_angle
            if np.isclose(stop, 90.0):
                stop = np.nextafter(90.0, 0.0)
            n_steps = int(np.round((stop - start) / step))
            return np.linspace(start, stop, n_steps + 1)
        elif self.incidence == "normal":
            return np.linspace(0, 1, 1)
        elif self.incidence == "angle":
            return np.linspace(self._incidence_angle[0], self._incidence_angle[0] + 1, 1)

    @incidence_angle.setter
    def incidence_angle(self, new_incidence_angle):
        """Set incidence-angle metadata and mark existing layer matrices stale."""
        self._incidence_angle = self._validate_incidence_angle(new_incidence_angle, self.incidence)
        self._alpha = None
        self._mark_layers_stale("incidence_angle was changed after model data existed.")

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
        """Return the treatment depth in millimeters."""
        return sum(value["thickness [mm]"] for value in self.matrix.values() if "thickness [mm]" in value)

    @property
    def color(self):
        """Return optional Matplotlib color specification used for cosmetic plotting."""
        return self._color

    @color.setter
    def color(self, new_color):
        """Set optional Matplotlib color specification used for cosmetic plotting."""
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
        self._project_folder = self._validate_project_folder(new_folder)

    @property
    def filename(self):
        """Return file-safe treatment identifier used by save and export paths."""
        if self._filename is not None:
            return self._filename
        else:
            return "TMM"

    @property
    def display_name(self):
        """Return optional human-readable treatment label."""
        return self._display_name

    @display_name.setter
    def display_name(self, new_display_name):
        """Set optional human-readable treatment label."""
        self._display_name = self._validate_display_name(new_display_name)

    @filename.setter
    def filename(self, new_filename):
        """Set file-safe treatment identifier used by save and export paths."""
        self._filename = self._validate_filename(new_filename)

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
        
    def reflection_and_absorption_coefficient(self, zs, angle=None):
        """
        Calculate reflection and absorption coefficients for a surface impedance.

        The normal-incidence relation is ``r = (Zs - Z0) / (Zs + Z0)``. For oblique incidence, ``Zs`` is treated as the
        surface-normal impedance at the boundary and the incident plane wave has normal characteristic impedance
        ``Z0 / cos(theta)``. The implemented reflection coefficient is therefore
        ``r(theta) = (Zs*cos(theta) - Z0) / (Zs*cos(theta) + Z0)`` and ``alpha(theta) = 1 - abs(r(theta))**2``.

        Parameters
        ----------
        zs : array
            Surface impedance.
        angle : float or array, optional
            Incidence angle in degrees. Zero degrees is normal incidence.

        Returns
        -------
        Reflection coefficient and absorption coefficient arrays.
        """
        if angle is None:
            cos_theta = 1.0
        else:
            cos_theta = np.cos(np.deg2rad(angle))

        r = (zs * cos_theta - self.z0) / (zs * cos_theta + self.z0)
        alpha = 1 - np.abs(r) ** 2

        return r, alpha

    def alpha_angle(self, angle_idx=0):
        """
        Return angle-dependent absorption coefficient.

        Parameters
        ----------
        angle_idx : int, optional
            Positional index of the desired angle in ``self.incidence_angle``. On an object returned by
            ``reduced_copy()`` this indexes the retained angles instead, and the reflection is evaluated at the angle
            those columns were actually computed for.

        Returns
        -------
        Angle-dependent absorption coefficient.
        """
        angles = self._stored_z_angle_angles()
        _, alpha = self.reflection_and_absorption_coefficient(
            self.z_angle[:, angle_idx],
            angle=angles[angle_idx],
        )

        return alpha

    def diffuse_absorption_coefficient(self, z_angle=None, angles=None):
        """Return statistical diffuse absorption from angle-dependent surface impedances.

        This is the plane-wave angular average often referred to as the Paris formula. The numerator integrates
        ``alpha(theta)*sin(theta)*cos(theta)`` and the denominator integrates ``sin(theta)*cos(theta)`` over the
        configured angular range. The ``sin(theta)`` term represents the solid-angle density for an isotropic field over
        a hemisphere, while ``cos(theta)`` projects incident intensity onto the surface normal. This averaging produces
        a scalar absorption coefficient; it is not a definition of a diffuse complex impedance. Use
        ``field_impedance()`` when a scalar field-incidence impedance is needed.
        """
        if z_angle is None:
            self._raise_if_partial_z_angle("diffuse_absorption_coefficient()")
            z_angle = self.z_angle
        if angles is None:
            angles = self.incidence_angle

        angles = np.asarray(angles, dtype=float)
        z_angle = np.asarray(z_angle)
        if z_angle.ndim != 2 or z_angle.shape[1] != len(angles):
            raise ValueError(
                f"z_angle must have one column per angle: got shape {z_angle.shape} for {len(angles)} angles."
            )
        angles_rad = np.deg2rad(angles)
        weights = np.sin(angles_rad) * np.cos(angles_rad)
        denominator = self._trapezoidal_integral(weights, angles_rad)
        angle_alpha = []
        for angle_idx, angle in enumerate(angles):
            _, alpha = self.reflection_and_absorption_coefficient(z_angle[:, angle_idx], angle=angle)
            angle_alpha.append(np.asarray(alpha).reshape(-1))
        angle_alpha = np.column_stack(angle_alpha)
        return self._trapezoidal_integral(angle_alpha * weights[None, :], angles_rad, axis=1) / denominator

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _unit_gauss_legendre(n):
        """
        Return Gauss-Legendre nodes and weights on ``[0, 1]``. Cached, since numpy builds them by an eigenvalue problem.
        """
        nodes, weights = np.polynomial.legendre.leggauss(n)
        return 0.5 * (nodes + 1), 0.5 * weights

    @classmethod
    def _corner_graded_rule(cls, n, length):
        """
        Return a Gauss rule on ``[0, length]`` under ``u = length*t**2``, which clusters nodes at the corner ``u = 0``.
        """
        t, w = cls._unit_gauss_legendre(n)
        return length * t ** 2, 2 * length * w * t

    @classmethod
    @functools.lru_cache(maxsize=16)
    def _static_lag_integral(cls, a, b):
        """
        Return the integral of ``(a - u)*(b - v)/R`` over ``[0, a] x [0, b]``, with ``R = sqrt(u**2 + v**2)``.

        In polar coordinates the radial integral is a polynomial, which leaves a smooth integral over the polar angle.
        """
        x, w = cls._unit_gauss_legendre(400)
        total = 0.0
        corner = np.arctan2(b, a)
        for lower, upper in ((0.0, corner), (corner, np.pi / 2)):
            psi = lower + (upper - lower) * x
            c, s = np.cos(psi), np.sin(psi)
            length = a / c if upper <= corner else b / s
            radial = a * b * length - (a * s + b * c) * length ** 2 / 2 + c * s * length ** 3 / 3
            total += (upper - lower) * np.sum(w * radial)
        return total

    @classmethod
    def _thomasson_field_impedance_tensor(cls, wavenumbers, a, b, theta, phi):
        """
        Return ``Z_F`` with shape ``(len(wavenumbers), len(theta), len(phi))`` for every azimuth in ``phi``.

        In the lag variables ``u = x0 - x`` and ``v = y0 - y`` the fourfold integral becomes

        ``Z_F = 2j*k/(pi*S) * integral over [0, a] x [0, b] of (a - u)*(b - v)*exp(-1j*k*R)/R``
        ``* cos(k*mu_x*u)*cos(k*mu_y*v)``.

        A tensor Gauss rule graded toward ``u = v = 0`` evaluates it, and the direction enters only through the
        separable cosines, so all directions follow from one matrix product per wavenumber. The quadrature of the
        ``1/R`` part alone is replaced by its exact value, which leaves a bounded integrand for the rule.
        """
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        th, ph = np.meshgrid(theta, phi, indexing="ij")
        mu_x = (np.sin(th) * np.cos(ph)).ravel()
        mu_y = (np.sin(th) * np.sin(ph)).ravel()
        static = cls._static_lag_integral(float(a), float(b))
        z_f = np.empty((len(wavenumbers), th.size), dtype=complex)
        for index, k in enumerate(wavenumbers):
            # About 0.4 nodes per radian of the largest phase 2*k*side, rounded up to a multiple of 32 so that only a
            # few node sets are generated and cached.
            u, w_u = cls._corner_graded_rule(32 * int(np.ceil((16 + 0.8 * k * a) / 32)), a)
            v, w_v = cls._corner_graded_rule(32 * int(np.ceil((16 + 0.8 * k * b) / 32)), b)
            distance = np.hypot(u[:, None], v[None, :])
            weight = (w_u * (a - u))[:, None] * (w_v * (b - v))[None, :] / distance
            cos_u = np.cos((k * mu_x)[:, None] * u[None, :])
            cos_v = np.cos((k * mu_y)[:, None] * v[None, :])
            real = np.einsum("dj,dj->d", cos_u @ (weight * np.cos(k * distance)), cos_v)
            imag = np.einsum("dj,dj->d", cos_u @ (weight * np.sin(k * distance)), cos_v)
            z_f[index] = 2j * k / (np.pi * a * b) * (real - 1j * imag - weight.sum() + static)
        return z_f.reshape(len(wavenumbers), *th.shape)

    @classmethod
    def _thomasson_field_impedance(cls, k, a, b, theta, phi):
        """
        Return Thomasson's normalized field impedance ``Z_F`` of an ``a`` by ``b`` rectangle.

        ``k`` is a scalar wavenumber [1/m], ``a`` and ``b`` are the sample sides [m], and ``theta`` and ``phi`` are 1-D
        arrays [rad]. The result has shape ``(len(theta), len(phi))``.
        """
        return cls._thomasson_field_impedance_grid(float(a), float(b), (float(k),),
                                                   tuple(float(t) for t in np.atleast_1d(theta)),
                                                   tuple(float(p) for p in np.atleast_1d(phi)))[0]

    @classmethod
    @functools.lru_cache(maxsize=4)
    def _thomasson_field_impedance_grid(cls, a, b, wavenumbers, theta, phi):
        """
        Return ``Z_F`` with shape ``(len(wavenumbers), len(theta), len(phi))`` for hashable tuple inputs.

        ``Z_F`` depends only on the sample geometry, frequency, and direction, so results are cached for reuse across
        absorbers evaluated on the same grids. A square gives ``Z_F(phi) = Z_F(pi/2 - phi)``, so only the distinct
        folded azimuths are evaluated.
        """
        phi = np.asarray(phi, dtype=float)
        evaluated, inverse = phi, np.arange(phi.size)
        if a == b:
            evaluated, inverse = np.unique(np.round(np.minimum(phi, np.pi / 2 - phi), 12), return_inverse=True)
        grid = cls._thomasson_field_impedance_tensor(np.asarray(wavenumbers), a, b, theta, evaluated)[:, :, inverse]
        grid.flags.writeable = False
        return grid

    @classmethod
    @functools.lru_cache(maxsize=4)
    def _field_impedance_on_frequency_grid(cls, a, b, wavenumbers, theta, phi):
        """
        Return ``Z_F`` at every wavenumber of a frequency grid, shape ``(len(wavenumbers), len(theta), len(phi))``.

        ``Z_F`` varies on a wavenumber scale of ``1/max(a, b)``, so a denser grid is served by a cubic spline through
        ``Z_F`` evaluated at that spacing. Cached, so several variants on one grid interpolate once.
        """
        wavenumbers = np.asarray(wavenumbers, dtype=float)
        nodes = np.unique(np.r_[np.arange(wavenumbers.min(), wavenumbers.max(), 1 / max(a, b)), wavenumbers.max()])
        if nodes.size >= wavenumbers.size:
            return cls._thomasson_field_impedance_grid(a, b, tuple(wavenumbers), theta, phi)
        grid = CubicSpline(nodes, cls._thomasson_field_impedance_grid(a, b, tuple(nodes), theta, phi), axis=0)(
            wavenumbers)
        grid.flags.writeable = False
        return grid

    def _angle_weights(self, angle_weights, angles):
        """
        Return angular weights with shape ``(len(self.freq), len(angles))`` for ``finite_sample_absorption``.
        """
        n_freq, n_angles = len(self.freq), len(angles)
        if angle_weights is None:
            return np.ones((n_freq, n_angles))
        if isinstance(angle_weights, str):
            if angle_weights not in JEONG_ANGLE_WEIGHTS:
                raise ValueError(f"angle_weights preset must be one of {sorted(JEONG_ANGLE_WEIGHTS)}.")
            table = np.asarray(JEONG_ANGLE_WEIGHTS[angle_weights])
            rows = np.array([np.interp(angles, JEONG_WEIGHT_ANGLES, row) for row in table])
            # Monotone cubic in log2(frequency) through the octave-band centres: PCHIP slopes at the interior centres
            # and zero slope at 250 Hz and 4 kHz, where the weights continue as constants. The slope is continuous
            # everywhere and the curve does not overshoot neighbouring table values.
            nodes = np.arange(len(JEONG_WEIGHT_BANDS))
            slopes = PchipInterpolator(nodes, rows, axis=0).derivative()(nodes)
            slopes[[0, -1]] = 0
            octave = np.clip(np.log2(np.asarray(self.freq) / JEONG_WEIGHT_BANDS[0]), 0, len(JEONG_WEIGHT_BANDS) - 1)
            return CubicHermiteSpline(nodes, rows, slopes, axis=0)(octave)
        if callable(angle_weights):
            weights = angle_weights(np.asarray(angles)[None, :], np.asarray(self.freq)[:, None])
        else:
            weights = angle_weights
        weights = np.asarray(weights, dtype=float)
        if weights.shape not in {(n_angles,), (n_freq, n_angles)}:
            raise ValueError(
                f"angle_weights must have shape ({n_angles},) or ({n_freq}, {n_angles}); got {weights.shape}."
            )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("angle_weights must be finite and non-negative.")
        return np.broadcast_to(weights, (n_freq, n_angles))

    def finite_sample_absorption(self, sample_size, z_angle=None, angles=None, phi_points=16, normalization="incident",
                                 angle_weights=None):
        """
        Return the statistical absorption coefficient of a finite rectangular sample in a diffuse field.

        Thomasson [1] writes the pressure on a locally reacting sample of normalized impedance ``zeta`` as the incident
        pressure times ``2*zeta/(zeta + Z_F)``, where the field impedance ``Z_F(theta, phi)`` depends on the sample
        shape, size, frequency, and direction of incidence. For a rectangle of sides ``a`` and ``b`` and area
        ``S = a*b``, equation 18 gives

        ``Z_F = (1j*k/S) * integral over S of integral over S of G(R)*exp(-1j*k*(mu_x*(x0 - x) + mu_y*(y0 - y)))``,

        with ``G(R) = exp(-1j*k*R)/(2*pi*R)``, ``mu_x = sin(theta)*cos(phi)`` and ``mu_y = sin(theta)*sin(phi)``. For an
        infinite sample ``Z_F`` tends to ``1/cos(theta)``. The statistical absorption coefficient is equation 25,

        ``alpha = (4/pi) * integral over theta and phi of Re(zeta)*sin(theta)/abs(zeta + Z_F)**2``,

        with ``phi`` over ``[0, 2*pi]``. The method evaluates ``phi`` by Gauss-Legendre quadrature on ``[0, pi/2]``,
        which covers the full range through the symmetry of the rectangle, and ``theta`` by the trapezoidal rule over
        ``angles``. When the frequency grid is finer than a wavenumber step of ``1/max(a, b)``, ``Z_F`` is evaluated at
        that step and interpolated with a cubic spline, since it varies on that scale. ``zeta`` is ``z_angle/z0`` at
        each angle, so the surface impedance of a layered, non-locally reacting stack enters at its own angle. With
        ``normalization="incident"`` the absorbed power is divided by the projected incident power of the infinite
        sample, as above, and the edge effect can make ``alpha`` exceed 1, as in reverberation-room measurements.

        The two other options divide the same absorbed power by a reference power that includes the finite size, with
        both integrals over ``angles``. ``normalization="true"`` uses the power absorbed by a sample whose impedance is
        the infinite-sample value ``1/cos(theta)``, Allard and Atalla [2], equation 12.41,

        ``alpha = integral of 4*Re(zeta)*sin(theta)/abs(zeta + Z_F)**2``
        ``/ integral of 4*cos(theta)*sin(theta)/abs(1 + Z_F*cos(theta))**2``.

        This removes most of the edge-effect excess, but a sample that absorbs more than the ``1/cos(theta)`` reference
        can still exceed 1. ``normalization="available"`` uses the available power of Thomasson [1], equation 28, the
        largest power a sample can absorb, reached when ``zeta`` matches ``conj(Z_F)``, so ``1/Re(Z_F)`` replaces
        ``cos(theta)`` in the incident power and ``alpha`` cannot exceed 1,

        ``alpha = integral of 4*Re(zeta)*sin(theta)/abs(zeta + Z_F)**2 / integral of sin(theta)/Re(Z_F)``.

        ``angle_weights`` accounts for the non-diffuse angular distribution of the incident intensity in a real
        reverberation chamber, following Jeong [3], equation 8, and Jeong [4], equation 7, which multiply the ``theta``
        integrand by a weight ``w(theta)`` normalized to a mean of 1 over 0 to 90 degrees,

        ``alpha = 2 * integral of w(theta)*4*Re(zeta)*sin(theta)/abs(zeta + Z_F)**2``.

        With ``"true"`` or ``"available"`` the method also weights the reference-power integral, a package extension.
        The presets are the octave-band weights of [4], table II, which tabulates the phased beam-tracing results of
        [3]: ``"rectangular_floor"``, the floor of a rectangular room ([3], figure 7); ``"irregular_floor"``, the floor
        of an irregular room ([3], figure 6b); and ``"irregular_wall"``, a vertical wall of the irregular room ([3],
        figure 4b). The names follow the figure captions of [3]; table II of [4] swaps "rectangular" and "irregular" in
        its block labels. The method interpolates the weights in ``log2(f)`` through the octave-band centres with a
        monotone piecewise cubic (PCHIP slopes inside, zero slope at the end bands), which keeps the slope continuous
        without overshooting the tabulated values, and linearly in angle between the tabulated 15-degree steps;
        frequencies below 250 Hz use the 250 Hz row as in [3], and frequencies above 4 kHz use the 4 kHz row. The
        irregular-floor value at 4 kHz and 45 degrees is printed as 0.11 in [4] and taken as 1.11, as read from [3],
        figure 6b.

        Parameters
        ----------
        sample_size : float or tuple of float
            Sample side lengths ``(a, b)`` [m]. A single value gives a square.
        z_angle : array or None, optional
            Angle-dependent surface impedance with one column per angle [Pa*s/m]. If ``None``, ``self.z_angle`` is used.
        angles : array or None, optional
            Incidence angles of the ``z_angle`` columns [deg]. If ``None``, ``self.incidence_angle`` is used. The upper
            limit sets the integration range, for example 90 or 78 degrees.
        phi_points : int, optional
            Number of Gauss-Legendre points in azimuth on ``[0, pi/2]``.
        normalization : str, optional
            Reference power for the absorption coefficient: ``"incident"``, ``"true"``, or ``"available"``.
        angle_weights : None, str, array, or callable, optional
            Angular weighting of the incident intensity. ``None`` gives uniform weights. A string selects a preset:
            ``"rectangular_floor"``, ``"irregular_floor"``, or ``"irregular_wall"``. An array has one weight per
            angle, shape ``(len(angles),)``, or per frequency and angle, shape ``(len(freq), len(angles))``. A callable
            ``w(theta_deg, freq)`` is evaluated on the angle [deg] and frequency [Hz] grids.

        Returns
        -------
        array
            Statistical absorption coefficient for each frequency.

        References
        ----------
        [1] S.-I. Thomasson, "On the absorption coefficient", Acustica, vol. 44, pp. 265-273, 1980.
        [2] J. F. Allard and N. Atalla, "Propagation of Sound in Porous Media: Modelling Sound Absorbing
            Materials", 2nd ed., Wiley, 2009.
        [3] C.-H. Jeong, "Non-uniform sound intensity distributions when measuring absorption coefficients in
            reverberation chambers using a phased beam tracing", Journal of the Acoustical Society of America,
            vol. 127, pp. 3560-3568, 2010.
        [4] C.-H. Jeong, "Converting Sabine absorption coefficients to random incidence absorption coefficients",
            Journal of the Acoustical Society of America, vol. 133, pp. 3951-3962, 2013.
        """
        sides = np.atleast_1d(np.asarray(sample_size, dtype=float))
        if sides.size == 1:
            sides = np.repeat(sides, 2)
        if sides.size != 2 or np.any(~np.isfinite(sides)) or np.any(sides <= 0):
            raise ValueError("sample_size must be a positive side length or a pair of positive side lengths [m].")
        if int(phi_points) < 1:
            raise ValueError("phi_points must be a positive integer.")
        if normalization not in {"incident", "true", "available"}:
            raise ValueError("normalization must be 'incident', 'true', or 'available'.")
        if z_angle is None:
            self._raise_if_partial_z_angle("finite_sample_absorption()")
            z_angle = self.z_angle
        if angles is None:
            angles = self.incidence_angle
        angles = np.asarray(angles, dtype=float)
        z_angle = np.asarray(z_angle)
        if z_angle.ndim != 2 or z_angle.shape != (len(self.freq), len(angles)) or len(angles) < 2:
            raise ValueError(
                f"z_angle must have one row per frequency and one column per angle, with at least two angles: got "
                f"shape {z_angle.shape} for {len(self.freq)} frequencies and {len(angles)} angles."
            )

        a, b = sides
        theta = np.deg2rad(angles)
        nodes, weights = np.polynomial.legendre.leggauss(int(phi_points))
        phi = np.pi / 4 * (nodes + 1)
        phi_weights = weights / 2  # Mean over [0, pi/2].
        z_f = self._field_impedance_on_frequency_grid(float(a), float(b), tuple(float(k) for k in self.w0 / self.c0),
                                                      tuple(float(t) for t in theta), tuple(float(p) for p in phi))
        zeta = (z_angle / self.z0)[:, :, None]
        weighted_sin = self._angle_weights(angle_weights, angles) * np.sin(theta)[None, :]

        def angular_integral(values):
            return self._trapezoidal_integral((values @ phi_weights) * weighted_sin, theta, axis=1)

        absorbed = angular_integral(4 * zeta.real / np.abs(zeta + z_f) ** 2)
        if normalization == "incident":
            return 2 * absorbed  # Projected incident power over the hemisphere: integral of cos*sin = 1/2.
        if normalization == "true":
            cos_theta = np.cos(theta)[None, :, None]
            return absorbed / angular_integral(4 * cos_theta / np.abs(1 + z_f * cos_theta) ** 2)
        return absorbed / angular_integral(1 / z_f.real)

    def alpha_w(self, sample_size=None):
        """
        Return the weighted sound absorption coefficient ``alpha_w`` of ISO 11654 [1] with its shape indicators.

        The absorption coefficient is first averaged over the narrowband lines of each third-octave band from 200 Hz to
        5 kHz. The bands are centred on the exact base-10 mid-band frequencies ``1000*10**(x/10)``, ``x = -7`` to ``7``,
        with edges at ``10**(+-0.05)`` times the centre, and a line at ``f`` belongs to the band when
        ``f_low <= f < f_high``. The practical coefficient ``alpha_p`` of each octave band from 250 Hz to 4 kHz is the
        mean of its three third-octave values, rounded to the second decimal and then to a step of 0.05, and limited to
        1.00 ([1], clause 4.1). A negative octave mean is taken as 0. The standard does not say how a tie at the second
        decimal is rounded, and the method rounds it up. The reference curve 0.80, 1.00, 1.00, 1.00, 0.90 ([1],
        figure 1) is shifted down in steps of 0.05 until the sum of the unfavourable deviations, where ``alpha_p`` is
        below the curve, is at most 0.10 (clause 4.2). ``alpha_w`` is the shifted curve at 500 Hz. A shape indicator
        marks an excess of ``alpha_p`` over the shifted curve of 0.25 or more (clause 4.3): ``L`` at 250 Hz, ``M`` at
        500 Hz or 1 kHz, and ``H`` at 2 kHz or 4 kHz. The class follows the informative annex B, table B.1. The method
        counts in hundredths, so a deviation sum of exactly 0.10 passes.

        Parameters
        ----------
        sample_size : float or tuple of float, optional
            Side lengths ``(a, b)`` of a finite sample [m]. If given, the method rates
            ``finite_sample_absorption(sample_size)`` with its default settings instead of ``self.alpha``. It needs a
            treatment computed with ``incidence="diffuse"``.

        Returns
        -------
        dict
            ``alpha_w`` (float), ``shape`` (the indicator letters, empty when there are none), ``label`` (for example
            ``"0.70(H)"``), ``class`` (``"A"`` to ``"E"`` or ``"not classified"``), ``practical`` (``alpha_p`` at
            250 Hz to 4 kHz), ``reference`` (the shifted reference curve at the same bands), ``third_octave`` (the
            band means at 200 Hz to 5 kHz), and ``incidence`` (the absorption rated: ``"finite sample a x b m"``,
            ``"diffuse (field)"``, ``"diffuse (paris)"``, ``"normal"``, or ``"angle <theta> deg"``).

        Raises
        ------
        RuntimeError
            If the treatment has not been computed.
        ValueError
            If the frequency grid does not cover the 200 Hz to 5 kHz third-octave bands or leaves one of them empty, if
            the absorption is not finite in a band, or if ``sample_size`` is given without diffuse incidence.

        Notes
        -----
        ISO 11654 rates reverberation-room measurements made to ISO 354 on a finite sample. A finite sample absorbs
        more than the infinite one through its edges, and ``sample_size`` rates the absorption of a finite sample of
        that size. Without it, the
        method rates ``self.alpha`` as the last ``compute()`` left it: with ``incidence="diffuse"`` the absorption of
        the field-incidence impedance (``diffuse_method="field"``, the default) or the Paris average
        (``diffuse_method="paris"``), with ``incidence="normal"`` the normal-incidence absorption, and with
        ``incidence="angle"`` the absorption at that angle. The ``incidence`` entry of the result names the case. The
        standard considers the rating not appropriate below 250 Hz ([1], clause 1.2).

        References
        ----------
        [1] ISO 11654:1997, "Acoustics - Sound absorbers for use in buildings - Rating of sound absorption",
            International Organization for Standardization, 1997.
        """
        if not self._has_computed_results():
            raise RuntimeError("alpha_w() needs computed results. Call compute() first.")
        freq = np.asarray(self.freq, dtype=float)
        centres = 1000 * 10 ** (np.array(ALPHA_W_THIRD_OCTAVE_INDICES) / 10)
        low, high = centres * 10 ** -0.05, centres * 10 ** 0.05
        if freq.min() > low[0] or freq.max() < high[-1]:
            raise ValueError(f"alpha_w needs the third-octave bands from 200 Hz to 5 kHz: set fmin <= "
                             f"{np.floor(low[0] * 10) / 10} Hz and fmax >= {np.ceil(high[-1] * 10) / 10} Hz.")
        if sample_size is None:
            alpha = self.alpha
            if self.incidence == "diffuse":
                rated = f"diffuse ({self.diffuse_method})"
            elif self.incidence == "angle":
                rated = f"angle {float(self.incidence_angle[0]):g} deg"
            else:
                rated = "normal"
        else:
            if self.incidence != "diffuse":
                raise ValueError("alpha_w(sample_size=...) needs a treatment computed with incidence='diffuse'.")
            alpha = self.finite_sample_absorption(sample_size)
            sides = np.atleast_1d(np.asarray(sample_size, dtype=float))
            rated = "finite sample " + " x ".join(f"{side:g}" for side in np.repeat(sides, 3 - sides.size)) + " m"
        alpha = np.asarray(alpha, dtype=float)
        third_octave = []
        for f_low, f_high in zip(low, high):
            in_band = (freq >= f_low) & (freq < f_high)
            if not np.any(in_band):
                raise ValueError(f"alpha_w found no frequency line in the third-octave band {f_low:.1f} to "
                                 f"{f_high:.1f} Hz; refine the frequency grid.")
            if not np.all(np.isfinite(alpha[in_band])):
                raise ValueError(f"alpha_w found a non-finite absorption in the third-octave band {f_low:.1f} to "
                                 f"{f_high:.1f} Hz.")
            third_octave.append(alpha[in_band].mean())
        third_octave = np.array(third_octave)
        practical = []
        for mean in third_octave.reshape(5, 3).mean(axis=1):
            hundredths = max(int(np.floor(np.round(mean * 100, 9) + 0.5)), 0)
            practical.append(min(5 * int(round(hundredths / 5)), 100))  # hundredths / 5 never ends in .5
        for shift in range(0, 105, 5):
            reference = [value - shift for value in ALPHA_W_REFERENCE]
            if sum(max(r - p, 0) for r, p in zip(reference, practical)) <= 10:
                break
        excess = [p - r >= 25 for p, r in zip(practical, reference)]
        shape = "".join(letter for letter, hit in (("L", excess[0]), ("M", excess[1] or excess[2]),
                                                   ("H", excess[3] or excess[4])) if hit)
        value = reference[1]
        rating_class = next((name for lowest, name in ALPHA_W_CLASSES if value >= lowest), "not classified")
        return {"alpha_w": value / 100, "shape": shape, "label": f"{value / 100:.2f}" + (f"({shape})" if shape else ""),
                "class": rating_class, "practical": tuple(p / 100 for p in practical),
                "reference": tuple(r / 100 for r in reference), "third_octave": third_octave, "incidence": rated}

    @staticmethod
    def porous_presets():
        """
        Return the ``porous_layer(preset=...)`` parameter sets as a table.

        Each row holds the parameters in ``porous_layer()`` units, the table and printed page of Allard and Atalla
        (2009), Olny and Panneton (2008) or Kino and Ueno (2007) the values come from, and the source's description of
        the material and how it is used there.

        Returns
        -------
        pandas.DataFrame
            One row per preset, indexed by preset name. The table displays its ``source`` and ``description``
            columns in full; the global pandas display options are left unchanged.
        """
        return _FullTextFrame.from_dict(POROUS_PRESETS, orient="index").rename_axis("preset")

    @staticmethod
    def flow_resistivity(density, fibre_diameter=None, model="bies_hansen"):
        """
        Estimate the airflow resistivity of a fibrous material from its bulk density and fibre diameter.

        ``model="bies_hansen"`` is equation 5 of Bies and Hansen [1] for glass fibre, ``sigma*d**2*rho_m**-1.53 = K``
        with ``K = 3.18e-9``, the flow resistivity ``sigma`` [Pa*s/m²], the fibre diameter ``d`` [m] and the bulk
        density ``rho_m`` [kg/m³]. The source applies it to material of fairly uniform fibre diameter in which the
        binder plays no large part ([1], p. 367). ``model="garai_pompoli"`` is equation 2 of Garai and Pompoli [2] for
        polyester fibre, ``sigma = 25.989*rho_m**1.404``, fitted to samples of 12 to 60 kg/m³ made of fibres of 18 to
        48 µm. It has no diameter. With ``fibre_diameter`` the method uses equation 24 of Kino and Ueno [3],
        ``sigma*d**2*rho_m**-1.404 = 28.3e-9``, which recasts equation 2 about the 33 µm mean diameter of [2].

        Parameters
        ----------
        density : float or array
            Bulk density of the material [kg/m³].
        fibre_diameter : float or array, optional
            Mean fibre diameter [µm]. Required by ``"bies_hansen"``; optional for ``"garai_pompoli"``. The fibreglass
            products measured in [1], figure 2, range from 6.5 to 9.4 µm; the figure legend prints the unit as mm.
        model : str, optional
            ``"bies_hansen"`` or ``"garai_pompoli"``.

        Returns
        -------
        float or array
            Flow resistivity [k*Pa*s/m²], the unit of ``porous_layer(sigma=...)``.

        References
        ----------
        [1] D. A. Bies and C. H. Hansen, "Flow resistance information for acoustical design", Applied Acoustics,
            vol. 13, pp. 357-391, 1980.
        [2] M. Garai and F. Pompoli, "A simple empirical model of polyester fibre materials for acoustical
            applications", Applied Acoustics, vol. 66, pp. 1383-1398, 2005.
        [3] N. Kino and T. Ueno, "Experimental determination of the micro- and macrostructural parameters
            influencing the acoustical performance of fibrous media", Applied Acoustics, vol. 68, pp. 1439-1458, 2007.
        """
        if model not in ("bies_hansen", "garai_pompoli"):
            raise NameError(f"Unknown flow resistivity model '{model}'. Use 'bies_hansen' or 'garai_pompoli'.")
        density = np.asarray(density, dtype=float)
        if np.any(~np.isfinite(density)) or np.any(density <= 0):
            raise ValueError("density must be positive [kg/m³].")
        if fibre_diameter is None:
            if model == "bies_hansen":
                raise ValueError("model 'bies_hansen' needs fibre_diameter [µm].")
            sigma = 25.989 * density ** 1.404
        else:
            d = np.asarray(fibre_diameter, dtype=float)
            if np.any(~np.isfinite(d)) or np.any(d <= 0):
                raise ValueError("fibre_diameter must be positive [µm].")
            d = d * 1e-6
            sigma = (3.18e-9 * density ** 1.53 if model == "bies_hansen" else 28.3e-9 * density ** 1.404) / d ** 2
        sigma = sigma / 1000
        return float(sigma) if np.ndim(sigma) == 0 else sigma

    def equivalent_fluid_model(self, sigma, model="mac", fibre_type=1, porosity=0.95, tortuosity=1.0,
                               viscous_length=None, thermal_length=None, thermal_permeability=None,
                               static_tortuosity=None, static_thermal_tortuosity=None, frame="rigid",
                               bulk_density=None, pore_shape_factor=None, perforation_diameter=None,
                               perforation_spacing=None, thickness=None, rigid_backed=True):
        """
        Return the complex propagation constant ``kc`` and characteristic impedance ``zc`` for an equivalent-fluid
        porous material.

        The coefficient-set models use the common power-law form ``kc = k0*(1 + C1*X**(-C2) - 1j*C3*X**(-C4))`` and
        ``Zc = Z0*(1 + C5*X**(-C6) - 1j*C7*X**(-C8))``, where ``X = rho0*f/sigma`` and ``sigma`` is in SI units
        [Pa*s/m2] inside this method. This family includes ``db``, ``miki``, ``qunli``, ``mac``, ``mechel_gf_lowX``,
        ``mechel_gf_highX``, ``mechel_rf_lowX``, and ``mechel_rf_highX``. The implementation follows the generic
        empirical equivalent-fluid form presented by Cox and D'Antonio [1], section 6.5.1. The ``db``, ``miki``,
        ``qunli`` and ``mac`` rows follow the Delany-Bazley-Miki options of the COMSOL Acoustics Module documentation,
        https://doc.comsol.com/6.4/doc/com.comsol.help.aco/aco_ug_pressure.05.005.html#1226466. The four ``mechel_*``
        sets follow Table 8.2 of [11], whose glass-fibre ``E <= 0.025`` row gives ``C5 = 0.0668``. ``qunli`` takes
        Table 2 of Wu [12]; the paper's equation 2 prints ``0.212*E**-0.455`` for the real part of ``Zc/Z0 - 1`` and
        ``0.188*E**-0.544`` for that of ``kc/k0 - 1``. ``garai_pompoli`` takes the same form with the coefficients that
        Garai and Pompoli [10] fitted to polyester fibre blankets (equations 3-6 and Table 3, which lists the impedance
        coefficients first), here ``C1..C4 = 0.121, 0.530, 0.159, 0.571`` for ``kc`` and
        ``C5..C8 = 0.078, 0.623, 0.074, 0.660`` for ``Zc``.

        The remaining branches use separate equations. ``komatsu`` follows Komatsu [2], equations 15-18, using powers of
        ``2 - log10(f/sigma)``. ``mechel_1976`` follows Mechel's [3] low-frequency extension of the Delany-Bazley
        absorber formula and uses ``porosity`` as the open porosity ``h``. ``mechel_grundmann`` keeps the
        Mechel-Grundmann polynomial formulation from the reference implementation accompanying [1]. ``wilson_2015``
        implements Wilson [4], equations 24-25, with unit shape factors, ``porosity`` mapped to ``phi`` and
        ``tortuosity`` mapped as described under Parameters. ``wilson_db`` keeps the compact Wilson relaxation model
        from the MATLAB reference script accompanying [1].

        ``jca``, ``jcal`` and ``jcapl`` follow the summary of Allard and Atalla [5], section 5.8. The effective density
        is ``rho = rho0*alpha/phi`` with ``alpha = alpha_inf + nu*phi/(j*omega*q0)*G``, ``q0 = eta/sigma`` and
        ``nu = eta/rho0`` (equations 5.50 and 5.58). The bulk modulus is
        ``K = gamma*P0/(gamma - (gamma - 1)/alpha')/phi`` with ``alpha' = 1 + nu'*phi/(j*omega*q0')*G'`` and
        ``nu' = nu/Pr`` (equations 5.51-5.52 and 5.59). All three use
        ``G = 1 - b + b*sqrt(1 + (2*alpha_inf*q0/(phi*Lambda*b))**2*j*omega/nu)`` and
        ``G' = 1 - b' + b'*sqrt(1 + (2*q0'/(phi*Lambda'*b'))**2*j*omega/nu')`` (equations 5.53 and 5.56-5.57).
        ``jca`` (Johnson-Champoux-Allard) sets ``b = b' = 1`` and ``q0' = phi*Lambda'**2/8`` (equation 5.54).
        ``jcal`` (Lafarge et al. [9]) takes ``q0'`` as ``thermal_permeability`` with ``b = b' = 1``. ``jcapl``
        (Pride-Lafarge) sets ``b = 2*alpha_inf**2*q0/(phi*Lambda**2*(alpha_0 - alpha_inf))`` from the static
        tortuosity ``alpha_0`` (equation 5.33) and ``b' = 2*q0'/(phi*Lambda'**2*(alpha_0' - 1))``, its thermal
        homologue with ``alpha_inf = 1``. Then ``kc = omega*sqrt(rho/K)`` and ``Zc = sqrt(rho*K)``.

        ``frame`` sets how the solid frame moves. All models describe only the compression wave in the pore air; the
        frame-borne waves of Biot theory, and with them frame resonances, are outside the package. With
        ``frame="rigid"`` (the default, and the form every model above is written in) the frame does not move at
        all [6], section 2.2. This holds when the frame is held in place and stiff, heavy, or weakly coupled to the
        air. It also excludes rigid-body motion, so a light layer that is free to move is misrepresented, mostly at
        low frequency, where the rigid density grows without bound as ``sigma/(j*omega)`` ([5], equation 5.37).

        With ``frame="limp"`` the frame has no stiffness and is carried by the pore air, so its inertia joins the
        fluid's ([6], section 2.3). The bulk modulus is unchanged and the density is replaced by Panneton's
        equation 7, ``rho_limp = (M*rho - rho0**2)/(M + rho - 2*rho0)`` with ``M = rho1 + phi*rho0``, where
        ``rho1`` is ``bulk_density`` and ``rho = Zc*kc/omega`` is the rigid-frame density of the selected model
        (``K = Zc*omega/kc`` is kept). As ``rho1`` grows the rigid result returns. At low frequency ``rho_limp``
        tends to ``M`` instead of diverging, so a light frame moves with the air and loses less to viscosity; the
        two frames differ mainly at low frequency ([5], section 11.3.4). The limp form fits soft, light frames
        (light fibreglass, thin light foams), unbonded or free-hanging layers, and transmission through porous
        layers; [5] places the limit at a frame bulk modulus below about 20 kPa, mounting aside. Every model accepts
        it; for the empirical ones ``phi`` is ``porosity``, which then enters only through ``M``.
        ``double_porosity`` takes ``frame="rigid"`` only.

        ``attenborough`` follows Attenborough [7], equations 8-10, with ``tortuosity`` as his ``q**2`` and
        ``pore_shape_factor`` as ``s_f``: ``kc/k0 = q*sqrt(A*B)`` and ``Zc/Z0 = q**2*A/(phi*kc/k0)``, where
        ``A = 1/(1 - 2*T(x)/x)``, ``B = 1 + 2*(gamma - 1)*T(sqrt(Pr)*x)/(sqrt(Pr)*x)``, ``T = J1/J0``,
        ``x = lambda*sqrt(-j)`` and ``lambda = sqrt(8*rho0*q**2*omega/(sigma*phi))/s_f``. The source writes these
        for ``exp(-i*omega*t)``; ``sqrt(-j)`` gives the package's ``exp(j*omega*t)`` form. ``Pr`` is the air Prandtl
        number of the object, which the source leaves unspecified. The source bounds ``s_f`` between 0.6 (slit pores)
        and 1 (circular pores).

        ``double_porosity`` follows the analytical model of Sgard et al. [8], equations 1-17, for a JCAL microporous
        medium with straight circular meso-perforations of radius ``R`` on a square cell of side ``Lc``, so
        ``phi_p = pi*R**2/Lc**2``. The micropore permeability is ``eta/(j*omega*rho_m)`` from the ``jcal`` density;
        the perforation permeability and bulk modulus use the Zwikker-Kosten circular-duct functions (equations 6, 9
        and 11). Then ``rho_dp = eta/(j*omega*((1 - phi_p)*Pi_m + Pi_p))`` and
        ``K_dp = 1/(1/K_p + (1 - phi_p)*F_d(omega*P0/(phi*K_m))/K_m)`` (equations 2, 5 and 8), with the
        pressure-diffusion function of equations 12-16. The diffusion length is the finite-thickness form of
        equation 17, ``Lambda_d = 2*(Lc**2 - pi*R**2)*t/(2*pi*R*t + n*(Lc**2 - pi*R**2))``, with ``n = 1`` when the
        layer lies on the rigid wall and ``n = 2`` otherwise (section 2.4). ``compute()`` sets ``rigid_backed`` from
        the final stack. The model assumes a rigid skeleton and perforations far larger than the micropores. Its
        absorption gain needs a high permeability contrast ([8], section 3.1): the diffusion frequency
        ``omega_d = (1 - phi_p)*P0/(phi*sigma*D(0))`` must lie well below the micropores' viscous characteristic
        frequency ``sigma*phi/(rho0*alpha_inf)`` (equations 15 and 25-27), and the gain appears around ``omega_d``.
        A low ``omega_d`` calls for a porous, highly resistive, weakly tortuous material with large holes at a low
        perforation ratio; when the condition fails, perforating can lower the absorption.

        The model follows from the parameters known for the material. With ``sigma`` only, use the empirical sets:
        ``qunli`` for porous plastics and open foams, ``garai_pompoli`` for polyester fibre blankets, and ``db``,
        ``miki``, ``mac``, ``komatsu`` or the Mechel sets for glass and mineral wool. Each set holds within the range
        of its source, which ``POROUS_VALIDITY`` lists and ``show_layers(validity=True)`` checks against the stack.
        Use ``mechel_1976`` when open porosity is also known and ``wilson_2015`` when porosity and tortuosity are.
        ``jca`` needs both characteristic lengths, ``jcal`` adds the static thermal permeability and ``jcapl`` the two
        static tortuosities; ``porous_layer(preset=...)`` supplies measured sets.
        ``attenborough`` describes soils, sands and other granular media through the pore shape factor [7].
        ``double_porosity`` applies to a porous panel perforated with holes much larger than its pores, typically a
        dense, highly resistive material [8].
        ``frame`` is chosen separately from the model, as described above.

        Parameters
        ----------
        sigma : float
            Flow resistivity in SI units [Pa*s/m2]. ``porous_layer()`` accepts [kPa*s/m2] and converts before calling
            this method.
        model : str, optional
            Equivalent-fluid model name, one of the branches listed above.
        fibre_type : int, optional
            Fibre type for ``mechel_grundmann``. ``1`` is basalt/rock wool and ``2`` is glass fibre.
        porosity : float, optional
            Open pore volume fraction for branches that expose porosity. Used by ``mechel_1976``, ``wilson_2015``,
            ``wilson_db``, ``jca``, ``jcal``, ``jcapl``, ``attenborough``, ``double_porosity`` and ``frame="limp"``.
        tortuosity : float, optional
            High-frequency tortuosity ``alpha_inf``. Used by ``jca``, ``jcal``, ``jcapl``, ``double_porosity``,
            ``attenborough`` (as ``q**2``) and ``model="wilson_2015"``, where Wilson's internal ``q`` parameter is
            computed as ``sqrt(tortuosity)``.
        viscous_length : float, optional
            Viscous characteristic length ``Lambda`` for ``jca``, ``jcal``, ``jcapl`` and ``double_porosity`` [m].
        thermal_length : float, optional
            Thermal characteristic length ``Lambda'`` for ``jca``, ``jcal``, ``jcapl`` and ``double_porosity`` [m].
        thermal_permeability : float, optional
            Static thermal permeability ``q0'`` for ``jcal``, ``jcapl`` and ``double_porosity`` [m²].
        static_tortuosity : float, optional
            Static viscous tortuosity ``alpha_0`` for ``jcapl``. Must exceed ``tortuosity``.
        static_thermal_tortuosity : float, optional
            Static thermal tortuosity ``alpha_0'`` for ``jcapl``. Must exceed one.
        frame : str, optional
            ``"rigid"`` (default): the frame stays still. ``"limp"``: the frame has no stiffness and moves with the
            pore air, adding its mass through ``bulk_density``. See the frame paragraphs above.
        bulk_density : float, optional
            In-vacuo bulk density of the frame, the mass of solid per unit volume of material, for ``frame="limp"``
            [kg/m³].
        pore_shape_factor : float, optional
            Pore shape factor ratio ``s_f`` for ``model="attenborough"``.
        perforation_diameter, perforation_spacing : float, optional
            Diameter of the circular meso-perforations and side of their square cell for
            ``model="double_porosity"`` [m].
        thickness : float, optional
            Layer thickness for ``model="double_porosity"`` [m].
        rigid_backed : bool, optional
            ``True`` when the double-porosity layer lies on a rigid wall, ``False`` when air or another layer
            follows it.

        Returns
        -------
        kc, zc : ndarray
            Complex propagation constant and characteristic acoustic impedance.

        References
        ----------
        [1] T. J. Cox and P. D'Antonio, "Acoustic Absorbers and Diffusers: Theory, Design and Application", 3rd ed.,
            CRC Press, 2017.
        [2] T. Komatsu, "Improvement of the Delany-Bazley and Miki models for fibrous sound-absorbing materials",
            Acoustical Science and Technology, vol. 29, pp. 121-129, 2008.
        [3] F. P. Mechel, "Ausweitung der Absorberformel von Delany und Bazley zu tiefen Frequenzen", Acustica,
            vol. 35, pp. 210-213, 1976.
        [4] D. K. Wilson, "General models for acoustical properties of porous media, and reduction to simpler
            models", Proceedings of Inter-Noise 2015, San Francisco, 2015.
        [5] J. F. Allard and N. Atalla, "Propagation of Sound in Porous Media: Modelling Sound Absorbing
            Materials", 2nd ed., Wiley, 2009, chapter 5.
        [6] R. Panneton, "Comments on the limp frame equivalent fluid model for porous media", Journal of the
            Acoustical Society of America, vol. 122, pp. EL217-EL222, 2007.
        [7] K. Attenborough, "Acoustical impedance models for outdoor ground surfaces", Journal of Sound and
            Vibration, vol. 99, pp. 521-544, 1985.
        [8] F. C. Sgard, X. Olny, N. Atalla and F. Castel, "On the use of perforations to improve the sound
            absorption of porous materials", Applied Acoustics, vol. 66, pp. 625-651, 2005.
        [9] D. Lafarge, P. Lemarinier, J. F. Allard and V. Tarnow, "Dynamic compressibility of air in porous
            structures at audible frequencies", Journal of the Acoustical Society of America, vol. 102,
            pp. 1995-2006, 1997.
        [10] M. Garai and F. Pompoli, "A simple empirical model of polyester fibre materials for acoustical
            applications", Applied Acoustics, vol. 66, pp. 1383-1398, 2005.
        [11] K. Attenborough and I. L. Vér, "Sound-absorbing materials and sound absorbers", in I. L. Vér and
            L. L. Beranek (eds.), "Noise and Vibration Control Engineering: Principles and Applications", 2nd ed.,
            Wiley, 2006, chapter 8.
        [12] Q. Wu, "Empirical relations between acoustical properties and flow resistivity of porous plastic
            open-cell foam", Applied Acoustics, vol. 25, pp. 141-148, 1988.
        """
        if frame not in POROUS_FRAMES:
            raise ValueError(f"frame must be one of {POROUS_FRAMES}.")
        coefficients = {  # List of coefficients for each available model
            "db": [0.0978, 0.7, 0.189, 0.595, 0.0571, 0.754, 0.087, 0.732],  # Delaney-Bazley
            "miki": [0.122, 0.618, 0.18, 0.618, 0.079, 0.632, 0.12, 0.632],  # Miki
            "qunli": [0.188, 0.554, 0.163, 0.592, 0.209, 0.548, 0.105, 0.607],  # Qunli
            "mechel_gf_lowX": [0.135, 0.646, 0.396, 0.458, 0.0668, 0.707, 0.196, 0.549],  # Mechel, glass fiber, low X
            "mechel_gf_highX": [0.102, 0.705, 0.179, 0.674, 0.0235, 0.887, 0.0875, 0.77],  # Mechel, glass fiber, high X
            "mechel_rf_lowX": [0.136, 0.641, 0.322, 0.502, 0.081, 0.699, 0.191, 0.556],  # Mechel, rock fiber, low X
            "mechel_rf_highX": [0.103, 0.716, 0.179, 0.663, 0.0563, 0.725, 0.127, 0.655],  # Mechel, rock fiber, high X
            "mac": [0.0982, 0.685, 0.288, 0.526, 0.0729, 0.66228, 0.187, 0.5379],  # Modified Allard and Champoux
            "garai_pompoli": [0.121, 0.530, 0.159, 0.571, 0.078, 0.623, 0.074, 0.660],  # Garai-Pompoli, polyester
        }

        if model in coefficients.keys():
            """
            Empirical models based on linear regressions. The coefficient tables follow the COMSOL
            Poroacoustics Delany-Bazley-Miki constants; the implementation uses the generic equivalent-fluid
            power-law form also presented by Cox and D'Antonio, *Acoustic Absorbers and Diffusers*, 3rd ed.,
            section 6.5.1. For applicability ranges and comparisons between porous-model options, see Oliva and
            Hongisto, "Sound absorption of porous materials - Accuracy of prediction methods" (2013).
            """
            c = coefficients[model]
            X = self.rho0 * self.freq / sigma
            kc = self.k0 * (1 + c[0] * X ** -c[1] - 1j * c[2] * X ** -c[3])  # Wavenumber
            zc = self.z0 * (1 + c[4] * X ** -c[5] - 1j * c[6] * X ** -c[7])  # Characteristic impedance

        elif model == "komatsu":
            """
            Komatsu's empirical model for fibrous materials. Equations 15-18 in Komatsu, T. (2008),
            "Improvement of the Delany-Bazley and Miki models for fibrous sound-absorbing materials",
            give the characteristic-impedance resistance and reactance and the propagation attenuation and phase
            constants as powers of ``2 - log10(f/sigma)``. With the package sign convention this gives
            ``kc = beta - j*alpha`` and ``Zc = R + j*X``.
            """
            y = 2 - np.log10(self.freq / sigma)
            kc = self.k0 * (1 + 0.0004 * y ** 6.2 - 1j * 0.0069 * y ** 4.1)
            zc = self.z0 * (1 + 0.00027 * y ** 6.2 - 1j * 0.0047 * y ** 4.1)

        elif model == "mechel_1976":
            """
            Mechel's low-frequency extension of the Delany-Bazley absorber formula. The formulation uses
            ``C = sigma/(rho0*f)`` and switches at ``C = 60``. Below the switch, the complex normalized propagation
            factor and characteristic impedance are empirical power laws. Above it, the low-frequency expressions
            follow Mechel's Rayleigh-model asymptotes, with porosity entering as
            ``sqrt(-1.4664 + j*gamma_air*h*C/(2*pi))`` and ``(C/(2*pi) + j*4/(3*h))/gamma``. The package convention
            maps Mechel's propagation factor to ``kc = -j*k0*gamma`` and ``Zc = Z0*z``.

            Based on "Ausweitung der Absorberformel von Delany und Bazley zu tiefen Frequenzen", Mechel, F. P. (1976).
            """
            h = porosity
            if h <= 0 or h > 1:
                raise ValueError("porosity must be greater than zero and less than or equal to one.")

            c = sigma / (self.rho0 * self.freq)
            gamma_norm = np.empty_like(c, dtype=complex)
            z_norm = np.empty_like(c, dtype=complex)
            low_frequency_extension = c > 60.0
            empirical_range = ~low_frequency_extension

            gamma_norm[empirical_range] = (
                0.189 * c[empirical_range] ** 0.6185
                + 1j * (1 + 0.0978 * c[empirical_range] ** 0.6929)
            )
            z_norm[empirical_range] = (
                1
                + 0.04885 * c[empirical_range] ** 0.754
                - 1j * 0.087 * c[empirical_range] ** 0.7307
            )

            gamma_norm[low_frequency_extension] = np.sqrt(
                -1.4664 + 1j * (self.air_prop["specific_heat_ratio"] * h / (2 * np.pi)) *
                c[low_frequency_extension]
            )
            z_norm[low_frequency_extension] = (
                c[low_frequency_extension] / (2 * np.pi) + 1j * (4 / (3 * h))
            ) / gamma_norm[low_frequency_extension]

            kc = -1j * self.k0 * gamma_norm
            zc = self.z0 * z_norm

        elif model == "wilson_db":
            """
            Wilson's Delany-Bazley-equivalent relaxation model as transcribed in the Cox and D'Antonio
            ``wilson_db.m`` reference implementation. With ``porosity=1`` this branch is algebraically identical
            to that MATLAB source. TMM also allows a porosity factor in the characteristic impedance for diagnostic
            comparison with porous-model exports that expose porosity explicitly.
            """
            if porosity <= 0:
                raise ValueError("porosity must be greater than zero.")
            X = self.rho0 * self.freq / sigma  # Dimensionless quantity
            omega = 1
            gamma = 1.4  # Ratio of specific heats
            q = 1
            zc = (self.z0 / porosity) * (q / omega) / np.sqrt(
                (1 + (gamma - 1) / np.sqrt(1 + 1j * 19 * X)) * (1 - 1 / np.sqrt(1 + 1j * 13 * X)))
            kc = (q * self.k0) * np.sqrt(
                (1 + (gamma - 1) / np.sqrt(1 + 1j * 19 * X)) / (1 - 1 / np.sqrt(1 + 1j * 13 * X)))

        elif model == "wilson_2015":
            """
            Wilson's three-parameter relaxation model, specialized to unit shape factors. Equations 24-25 in
            Wilson, D. K. (2015), "General models for acoustical properties of porous media, and reduction to
            simpler models", express the normalized dynamic density and compressibility with one relaxation time
            ``tau0 = 2*rho0*q**2/(porosity*sigma)``. The public ``tortuosity`` input is interpreted as
            ``alpha_inf = q**2``; this branch therefore uses ``q = sqrt(tortuosity)``.
            """
            if porosity <= 0 or tortuosity <= 0:
                raise ValueError("porosity and tortuosity must be greater than zero.")
            gamma = self.air_prop["specific_heat_ratio"]
            prandtl = self.air_prop["prandtl_number"]
            q = np.sqrt(tortuosity)
            tau0 = 2 * self.rho0 * tortuosity / (porosity * sigma)
            omega_tau0 = self.w0 * tau0
            relax_vor = 1 / np.sqrt(1 + 1j * omega_tau0)
            relax_ent = 1 / np.sqrt(1 + 1j * prandtl * omega_tau0)
            dynamic_density_term = -2j / omega_tau0 + 1 + relax_vor
            compressibility_term = 1 + (gamma - 1) * relax_ent
            zc = self.z0 * (q / porosity) * np.sqrt(dynamic_density_term / compressibility_term)
            kc = self.k0 * q * np.sqrt(dynamic_density_term * compressibility_term)

        elif model == "mechel_grundmann":
            """
            Mechel and Grundmann polynomial formulation as implemented in the Cox and D'Antonio MATLAB
            reference ``mechel_grundmann.m``.
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
        elif model in JCA_MODELS:
            kc, zc = self._jca_equivalent_fluid(sigma, model, porosity, tortuosity, viscous_length, thermal_length,
                                                thermal_permeability, static_tortuosity, static_thermal_tortuosity)
        elif model == "attenborough":
            kc, zc = self._attenborough(sigma, porosity, tortuosity, pore_shape_factor)
        elif model == "double_porosity":
            if frame != "rigid":
                raise ValueError("model='double_porosity' supports frame='rigid' only.")
            kc, zc = self._double_porosity(sigma, porosity, tortuosity, viscous_length, thermal_length,
                                           thermal_permeability, perforation_diameter, perforation_spacing,
                                           thickness, rigid_backed)
        else:
            available_models = [key for key in coefficients.keys()]
            available_models.append("komatsu")
            available_models.append("mechel_1976")
            available_models.append("wilson_2015")
            available_models.append("wilson_db")
            available_models.append("mechel_grundmann")
            available_models.extend(JCA_MODELS)
            available_models.extend(["attenborough", "double_porosity"])
            raise NameError("Unidentified model. Choose between the available models: ", available_models)

        if frame == "limp":
            kc, zc = self._limp_frame(kc, zc, porosity, bulk_density)

        return kc, zc

    def _jca_equivalent_fluid(self, sigma, model, porosity, tortuosity, viscous_length, thermal_length,
                              thermal_permeability=None, static_tortuosity=None, static_thermal_tortuosity=None):
        """Return ``kc, zc`` of the JCA, JCAL or JCAPL rigid-frame equivalent fluid, with lengths in [m]."""
        rho_eq, bulk_eq = self._jca_density_bulk(sigma, model, porosity, tortuosity, viscous_length, thermal_length,
                                                 thermal_permeability, static_tortuosity, static_thermal_tortuosity)
        return self.w0 * np.sqrt(rho_eq / bulk_eq), np.sqrt(rho_eq * bulk_eq)

    def _jca_density_bulk(self, sigma, model, porosity, tortuosity, viscous_length, thermal_length,
                          thermal_permeability=None, static_tortuosity=None, static_thermal_tortuosity=None):
        """Return the equivalent density and bulk modulus of the JCA, JCAL or JCAPL fluid, with lengths in [m]."""
        required = {"viscous_length": viscous_length, "thermal_length": thermal_length}
        if model in ("jcal", "jcapl"):
            required["thermal_permeability"] = thermal_permeability
        if model == "jcapl":
            required["static_tortuosity"] = static_tortuosity
            required["static_thermal_tortuosity"] = static_thermal_tortuosity
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"model='{model}' needs {', '.join(missing)}.")
        if not 0 < porosity <= 1:
            raise ValueError("porosity must be greater than zero and less than or equal to one.")
        if tortuosity < 1:
            raise ValueError("tortuosity must be at least one.")
        if sigma <= 0 or viscous_length <= 0 or thermal_length <= 0:
            raise ValueError("sigma, viscous_length and thermal_length must be greater than zero.")

        eta = self.air_prop["air_viscosity"]
        gamma = self.air_prop["specific_heat_ratio"]
        nu = eta / self.rho0
        nu_thermal = nu / self.air_prop["prandtl_number"]
        p0 = self.air_prop["atmospheric_pressure"]
        omega = self.w0

        q0 = eta / sigma
        if model == "jca":
            q0_thermal = porosity * thermal_length ** 2 / 8
        else:
            if thermal_permeability <= 0:
                raise ValueError("thermal_permeability must be greater than zero.")
            q0_thermal = thermal_permeability

        if model == "jcapl":
            if static_tortuosity <= tortuosity or static_thermal_tortuosity <= 1:
                raise ValueError("static_tortuosity must exceed tortuosity and static_thermal_tortuosity must "
                                 "exceed one.")
            b = 2 * tortuosity ** 2 * q0 / (porosity * viscous_length ** 2 * (static_tortuosity - tortuosity))
            b_thermal = 2 * q0_thermal / (porosity * thermal_length ** 2 * (static_thermal_tortuosity - 1))
        else:
            b = b_thermal = 1.0

        g = 1 - b + b * np.sqrt(1 + (2 * tortuosity * q0 / (porosity * viscous_length * b)) ** 2 * 1j * omega / nu)
        g_thermal = 1 - b_thermal + b_thermal * np.sqrt(
            1 + (2 * q0_thermal / (porosity * thermal_length * b_thermal)) ** 2 * 1j * omega / nu_thermal)
        alpha = tortuosity + nu * porosity / (1j * omega * q0) * g
        alpha_thermal = 1 + nu_thermal * porosity / (1j * omega * q0_thermal) * g_thermal

        rho_eq = self.rho0 * alpha / porosity
        bulk_eq = gamma * p0 / (gamma - (gamma - 1) / alpha_thermal) / porosity
        return rho_eq, bulk_eq

    @staticmethod
    def _bessel_ratio_term(x):
        """Return ``2*J1(x)/(x*J0(x))``; scaled Bessel functions keep the ratio finite at large ``|x|``."""
        return 2 * special.jve(1, x) / (x * special.jve(0, x))

    def _attenborough(self, sigma, porosity, tortuosity, pore_shape_factor):
        """Return ``kc, zc`` of Attenborough's four-parameter rigid-porous model."""
        if pore_shape_factor is None:
            raise ValueError("model='attenborough' needs pore_shape_factor.")
        if not 0 < porosity <= 1:
            raise ValueError("porosity must be greater than zero and less than or equal to one.")
        if sigma <= 0 or tortuosity < 1 or pore_shape_factor <= 0:
            raise ValueError("sigma and pore_shape_factor must be greater than zero and tortuosity at least one.")
        gamma = self.air_prop["specific_heat_ratio"]
        root_prandtl = np.sqrt(self.air_prop["prandtl_number"])
        lam = np.sqrt(8 * self.rho0 * tortuosity * self.w0 / (sigma * porosity)) / pore_shape_factor
        x = lam * np.sqrt(-1j)
        viscous = 1 / (1 - self._bessel_ratio_term(x))
        thermal = 1 + (gamma - 1) * self._bessel_ratio_term(root_prandtl * x)
        kc_norm = np.sqrt(tortuosity * viscous * thermal)
        return self.k0 * kc_norm, self.z0 * tortuosity * viscous / (porosity * kc_norm)

    def _double_porosity(self, sigma, porosity, tortuosity, viscous_length, thermal_length, thermal_permeability,
                         perforation_diameter, perforation_spacing, thickness, rigid_backed):
        """Return ``kc, zc`` of a JCAL microporous medium with circular meso-perforations, lengths in [m]."""
        required = {"thermal_permeability": thermal_permeability, "perforation_diameter": perforation_diameter,
                    "perforation_spacing": perforation_spacing, "thickness": thickness}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"model='double_porosity' needs {', '.join(missing)}.")
        if thickness <= 0 or perforation_spacing <= 0:
            raise ValueError("thickness and perforation_spacing must be greater than zero.")
        radius = perforation_diameter / 2
        cell_area = perforation_spacing ** 2
        phi_p = np.pi * radius ** 2 / cell_area
        if not 0 < phi_p < 1:
            raise ValueError("perforation_diameter must be positive and leave material between the perforations.")
        rho_m, bulk_m = self._jca_density_bulk(sigma, "jcal", porosity, tortuosity, viscous_length, thermal_length,
                                               thermal_permeability)
        eta = self.air_prop["air_viscosity"]
        gamma = self.air_prop["specific_heat_ratio"]
        p0 = self.air_prop["atmospheric_pressure"]
        omega = self.w0

        # Perforations: Zwikker-Kosten viscous and thermal permeabilities of a circular duct.
        delta_v2 = eta / (self.rho0 * omega)
        delta_t2 = self.air_prop["air_thermal_conductivity"] / (
            self.rho0 * self.air_prop["constant_pressure_specific_heat"] * omega)
        perm_p = phi_p / 1j * delta_v2 * (1 - self._bessel_ratio_term(radius / np.sqrt(delta_v2) * np.sqrt(-1j)))
        thermal_p = phi_p / 1j * delta_t2 * (1 - self._bessel_ratio_term(radius / np.sqrt(delta_t2) * np.sqrt(-1j)))
        bulk_p = gamma * p0 / phi_p / (gamma - 1j * (gamma - 1) * thermal_p / (delta_t2 * phi_p))

        # Micropores: dynamic viscous permeability from the JCAL density.
        perm_m = eta / (1j * omega * rho_m)
        rho_dp = eta / (1j * omega * ((1 - phi_p) * perm_m + perm_p))

        # Pressure diffusion between perforations and micropores.
        d0 = cell_area / (4 * np.pi) * (np.log(1 / phi_p) - 1.5 + 2 * phi_p - phi_p ** 2 / 2)
        omega_d = (1 - phi_p) * p0 / (porosity * sigma * d0)
        solid = cell_area - np.pi * radius ** 2
        faces = solid if rigid_backed else 2 * solid
        length_d = 2 * solid * thickness / (2 * np.pi * radius * thickness + faces)
        m_d = 8 * d0 / (length_d ** 2 * (1 - phi_p))
        x = omega * p0 / (porosity * bulk_m) / omega_d
        f_d = 1 - 1j * x / (1j * x + np.sqrt(1 + 1j * m_d / 2 * x))
        bulk_dp = 1 / (1 / bulk_p + (1 - phi_p) * f_d / bulk_m)
        return omega * np.sqrt(rho_dp / bulk_dp), np.sqrt(rho_dp * bulk_dp)

    def _rigid_backed(self, layer):
        """Return whether ``layer`` is the last layer of a stack computed on a rigid backing."""
        keys = [key for key in self.matrix if isinstance(key, (int, np.integer))]
        is_last = not keys or layer >= max(keys)
        return bool(is_last and getattr(self, "_build_backing", "rigid") == "rigid")

    def _limp_frame(self, kc, zc, porosity, bulk_density):
        """Replace the effective density in ``kc, zc`` by Panneton's limp-frame density."""
        if bulk_density is None or bulk_density <= 0:
            raise ValueError("frame='limp' needs a positive bulk_density.")
        if not 0 < porosity <= 1:
            raise ValueError("porosity must be greater than zero and less than or equal to one.")
        omega = self.w0
        rho_eq = zc * kc / omega
        bulk_eq = zc * omega / kc
        total_density = bulk_density + porosity * self.rho0
        rho_limp = (total_density * rho_eq - self.rho0 ** 2) / (total_density + rho_eq - 2 * self.rho0)
        return omega * np.sqrt(rho_limp / bulk_eq), np.sqrt(rho_limp * bulk_eq)

    def porous_layer(self, sigma=None, t=5, model=None, fibre_type=1, porosity=None, tortuosity=None, layer=None, *,
                     viscous_length=None, thermal_length=None, thermal_permeability=None, static_tortuosity=None,
                     static_thermal_tortuosity=None, frame="rigid", bulk_density=None, pore_shape_factor=None,
                     perforation_diameter=None, perforation_spacing=None, preset=None):
        """
        Add a porous layer modelled as an equivalent fluid.

        The porous material is treated as an equivalent fluid with complex characteristic impedance ``Zc`` and complex
        wavenumber ``kc`` from the selected model. For oblique incidence, the tangential wavenumber is conserved at the
        air/material interface: ``kt = k0*sin(theta)`` and ``kz = sqrt(kc**2 - kt**2)``. The layer transfer matrix is
        then built with the normal wavenumber ``kz`` and the normal characteristic impedance ``Zc,n = Zc*kc/kz``:
        ``[[cos(kz*t), j*Zc,n*sin(kz*t)], [j*sin(kz*t)/Zc,n, cos(kz*t)]]``. This is the standard equivalent-fluid
        extended-reaction form; for notation, see Allard and Atalla [1].

        ``preset`` fills ``sigma``, ``porosity``, ``tortuosity``, ``viscous_length``, ``thermal_length``,
        ``bulk_density`` and, where given, ``thermal_permeability`` with the parameters of one characterized material,
        and any of these passed explicitly overrides the preset value. An override that differs from the preset raises
        one ``UserWarning`` listing each changed value, and the record keeps that list as ``preset_overrides``;
        ``rebuild()`` and ``load()`` do not repeat it. Without ``model``, a preset selects the model in its ``model``
        column: ``jcal`` for the sets that include a thermal permeability (``q0'``), ``jca`` otherwise. An explicit
        ``model`` takes only the parameters that model uses, so ``model="jca"`` on a ``jcal`` preset ignores its thermal
        permeability. Names give the material and its flow resistivity. Each set describes one measured sample, not a
        material class, and ``jcapl`` always needs its static parameters. The record stores the resolved model and
        values, so ``log_rebuild()`` does not depend on the preset table. ``porous_presets()`` lists each set with its
        source page and the source's description of the material.

        ======================= ===== ============ ====== ===== ====== ====== ======= ===== ========
        ``preset``              model Source       sigma  phi   alpha  Lambda Lambda' rho1  q0'
        ======================= ===== ============ ====== ===== ====== ====== ======= ===== ========
        ``foam_2.3k``           jcal  [3] I, II    2.3    0.96  1.28   202    376     60    1.67e-08
        ``polyester_4.1k``      jca   [6] 2 (14)   4.1    0.964 1.0354 269    541     49.9
        ``carpet_5k``           jca   [1] 11.7     5      0.99  1      23     28      60
        ``foam_5k``             jca   [1] 11.3     5      0.98  1.1    150    216     33
        ``polyester_5.7k``      jca   [6] 2 (15)   5.7    0.953 1.0464 206    399     64.4
        ``foam_6.6k``           jca   [1] 11.9     6.6    0.98  1.03   200    380     11.2
        ``polyester_9.8k``      jca   [6] 2 (12)   9.8    0.964 1.0328 152    292     49.8
        ``foam_10.9k``          jca   [1] 11.5 (3) 10.9   0.99  1.02   100    130     8.8
        ``glass_wool_11.9k``    jca   [6] 1 (1)    11.9   0.989 1.0108 143    302     28
        ``foam_12.6k``          jca   [1] 13.1     12.569 0.99  1.02   78     192     8.85
        ``glass_wool_16.8k``    jca   [6] 1 (2)    16.8   0.987 1.0093 132    237     31.8
        ``polyester_17.4k``     jca   [6] 2 (13)   17.4   0.947 1.0494 113    218     72.8
        ``polyester_19.7k``     jca   [6] 2 (10)   19.7   0.974 1.0206 125    221     35.9
        ``glass_wool_21.5k``    jca   [6] 1 (3)    21.5   0.983 1.0124 106    225     41.7
        ``foam_22k``            jca   [1] 11.6     22     0.98  1.9    87     146     30
        ``foam_22.1k``          jca   [1] 8.1 (2)  22.1   0.97  2.2    39     275     24.5
        ``felt_23k``            jca   [1] 11.5 (1) 23     0.99  1.4    64     131     66
        ``glass_wool_24.3k``    jcal  [3] I, II    24.3   0.988 1.01   63     137     53    3.1e-09
        ``soft_fibrous_25k``    jca   [1] 11.2     25     0.98  1.02   90     180     30
        ``glass_wool_29k``      jca   [6] 1 (4)    29     0.979 1.0144 87     184     52.6
        ``fibrous_layer_33k``   jca   [1] 11.7     33     0.98  1.1    50     110     60
        ``fibrous_blanket_34k`` jca   [1] 11.4 (1) 34     0.98  1.18   60     86      41
        ``mineral_wool_34k``    jca   [1] 12.2     34     0.95  1      40     80      90
        ``glass_wool_40k``      jca   [1] 11.8     40     0.94  1.06   56     110     130
        ``glass_wool_49k``      jca   [6] 1 (5)    49     0.968 1.0295 57     123     81.2
        ``foam_50k``            jca   [1] 8.1 (1)  50     0.98  1.4    50     150     25
        ``polyester_51k``       jca   [6] 2 (11)   51     0.947 1.0471 67     128     72.9
        ``rock_wool_51.2k``     jcal  [3] I, II    51.2   0.97  1.06   41     59      183   1.2e-09
        ``foam_65k``            jca   [1] 11.4 (4) 65     0.99  1.98   37     120     16
        ``glass_wool_69.9k``    jca   [6] 1 (6)    69.9   0.96  1.0358 48     97      101.2
        ``foam_87k``            jca   [1] 11.4 (3) 87     0.97  2.52   36     118     31
        ``rock_wool_135k``      jcal  [1] 13.2     135    0.94  2.1    49     166     130   3.3e-09
        ======================= ===== ============ ====== ===== ====== ====== ======= ===== ========

        Units are those of the Parameters below. The number in brackets after a table is its layer, material or sample
        number. ``soft_fibrous_25k`` is also the material of Panneton [2], Table 1. The [3] sets were measured in [3]:
        the viscous parameters with the inversion method of [4] and the thermal ones by fitting the Lafarge et al. model
        of [5] to measured bulk moduli. The [6] sets are six glass wool samples of nominal 7 µm fibres and polyester of
        three fibre diameters, two densities each. [6] measured the flow resistivity, tortuosity and both characteristic
        lengths, and calculated the porosity from the bulk density with fibre densities of 2500 kg/m³ (glass) and
        1380 kg/m³ (polyester). ``glass_wool_40k`` is the glass wool "Domisol Coffrage" of Saint-Gobain Isover ([1],
        section 6.5.4); the sources name no other product.

        Parameters
        ----------
        sigma : float, optional
            Flow resistivity of the porous material [k*Pa*s/m²]. Defaults to the preset value, or 27 without a preset.
        t : float or int, optional
            Thickness of the porous material [mm]
        model : str, optional
            Equivalent-fluid model name. See ``equivalent_fluid_model()`` for the options. Defaults to the preset's
            model (column ``model`` above), or ``"mac"`` without a preset.
        fibre_type : int, optional
            Fibre type for the Mechel-Grundmann model: ``1`` for basalt or rock wool, ``2`` for glass fibre.
        porosity : float, optional
            Open pore volume fraction for models that expose it. Used by ``model="wilson_2015"``, ``model="wilson_db"``,
            ``model="mechel_1976"``, ``jca``, ``jcal``, ``jcapl``, ``attenborough``, ``double_porosity`` and
            ``frame="limp"``. Defaults to the preset value, or 0.95 without a preset.
        tortuosity : float, optional
            High-frequency tortuosity. Used by ``jca``, ``jcal``, ``jcapl``, ``double_porosity``, ``attenborough`` (as
            Attenborough's ``q**2``) and ``model="wilson_2015"``, where Wilson's internal ``q`` is computed as
            ``sqrt(tortuosity)``. Defaults to the preset value, or 1 without a preset.
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.
        viscous_length : float, optional
            Viscous characteristic length for ``jca``, ``jcal``, ``jcapl`` and ``double_porosity`` [µm].
        thermal_length : float, optional
            Thermal characteristic length for ``jca``, ``jcal``, ``jcapl`` and ``double_porosity`` [µm].
        thermal_permeability : float, optional
            Static thermal permeability for ``model="jcal"``, ``model="jcapl"`` and ``model="double_porosity"``
            [m²].
        static_tortuosity : float, optional
            Static viscous tortuosity for ``model="jcapl"``.
        static_thermal_tortuosity : float, optional
            Static thermal tortuosity for ``model="jcapl"``.
        frame : str, optional
            ``"rigid"`` (default): the frame stays still, valid for held and stiff, heavy, or weakly coupled frames.
            ``"limp"``: the
            frame has no stiffness and moves with the pore air, adding its mass through Panneton's limp-frame density;
            suited to soft, light or unbonded layers, and it changes the result mainly at low frequency. See
            ``equivalent_fluid_model()`` for the equations and limits.
        bulk_density : float, optional
            In-vacuo bulk density of the frame, the mass of solid per unit volume of material, for ``frame="limp"``
            [kg/m³]. Every preset includes it.
        pore_shape_factor : float, optional
            Pore shape factor ratio for ``model="attenborough"``.
        perforation_diameter : float, optional
            Diameter of the circular meso-perforations for ``model="double_porosity"`` [mm].
        perforation_spacing : float, optional
            Side of the square cell holding one perforation for ``model="double_porosity"`` [mm].
        preset : str, optional
            Name of a material parameter set from the table above.

        References
        ----------
        [1] J. F. Allard and N. Atalla, "Propagation of Sound in Porous Media: Modelling Sound Absorbing
            Materials", 2nd ed., Wiley, 2009.
        [2] R. Panneton, "Comments on the limp frame equivalent fluid model for porous media", Journal of the
            Acoustical Society of America, vol. 122, pp. EL217-EL222, 2007.
        [3] X. Olny and R. Panneton, "Acoustical determination of the parameters governing thermal dissipation in
            porous media", Journal of the Acoustical Society of America, vol. 123, pp. 814-824, 2008.
        [4] R. Panneton and X. Olny, "Acoustical determination of the parameters governing viscous dissipation in
            porous media", Journal of the Acoustical Society of America, vol. 119, pp. 2027-2040, 2006.
        [5] D. Lafarge, P. Lemarinier, J. F. Allard and V. Tarnow, "Dynamic compressibility of air in porous
            structures at audible frequencies", Journal of the Acoustical Society of America, vol. 102,
            pp. 1995-2006, 1997.
        [6] N. Kino and T. Ueno, "Experimental determination of the micro- and macrostructural parameters
            influencing the acoustical performance of fibrous media", Applied Acoustics, vol. 68, pp. 1439-1458, 2007.
        """
        preset_overrides = None
        given = {"sigma": sigma, "porosity": porosity, "tortuosity": tortuosity, "viscous_length": viscous_length,
                 "thermal_length": thermal_length, "thermal_permeability": thermal_permeability,
                 "bulk_density": bulk_density}
        if preset is not None:
            if preset not in POROUS_PRESETS:
                raise ValueError(f"Unknown preset {preset!r}. Choose from {sorted(POROUS_PRESETS)}.")
            values = POROUS_PRESETS[preset]
            overrides = [f"{name} {values[name]:g} -> {value:g}" for name, value in given.items()
                         if value is not None and values[name] is not None and value != values[name]]
            if overrides:
                preset_overrides = ", ".join(overrides)
                warnings.warn(f"porous_layer(preset={preset!r}): overriding preset values {preset_overrides}.",
                              UserWarning, stacklevel=2)
            given = {name: values[name] if value is None else value for name, value in given.items()}
            if model is None:
                model = POROUS_PRESETS[preset]["model"]
        if model is None:
            model = "mac"
        for name, default in (("sigma", 27), ("porosity", 0.95), ("tortuosity", 1.0)):
            if given[name] is None:
                given[name] = default
        sigma, porosity, tortuosity = given["sigma"], given["porosity"], given["tortuosity"]
        viscous_length, thermal_length = given["viscous_length"], given["thermal_length"]
        thermal_permeability, bulk_density = given["thermal_permeability"], given["bulk_density"]

        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        sigma_k = sigma * 1000  # Convert kPa*s/m2 to Pa*s/m2
        if layer is None:
            layer = len(self.matrix)
        rigid_backed = self._rigid_backed(layer) if model == "double_porosity" else None

        kc, zc = self.equivalent_fluid_model(
            sigma_k,
            model=model,
            fibre_type=fibre_type,
            porosity=porosity,
            tortuosity=tortuosity,
            viscous_length=None if viscous_length is None else viscous_length * 1e-6,
            thermal_length=None if thermal_length is None else thermal_length * 1e-6,
            thermal_permeability=thermal_permeability,
            static_tortuosity=static_tortuosity,
            static_thermal_tortuosity=static_thermal_tortuosity,
            frame=frame,
            bulk_density=bulk_density,
            pore_shape_factor=pore_shape_factor,
            perforation_diameter=None if perforation_diameter is None else perforation_diameter / 1000,
            perforation_spacing=None if perforation_spacing is None else perforation_spacing / 1000,
            thickness=t_meters,
            rigid_backed=rigid_backed,
        )

        kc = np.asarray(kc, dtype=complex)[:, None]
        zc = np.asarray(zc, dtype=complex)[:, None]
        k0 = np.asarray(self.k0, dtype=float)[:, None]
        theta = np.deg2rad(np.asarray(self.incidence_angle, dtype=float))[None, :]
        kt = k0 * np.sin(theta)
        kz = np.sqrt(kc ** 2 - kt ** 2)
        zc_normal = zc * kc / kz

        Tp = np.array([[np.cos(kz * t_meters), 1j * zc_normal / self.s0 * np.sin(kz * t_meters)],
                       [1j * self.s0 / zc_normal * np.sin(kz * t_meters), np.cos(kz * t_meters)]])

        self.matrix[layer] = {"type": "porous_layer",
                              "flow_resistivity [k*Pa*s/m²]": sigma,
                              "thickness [mm]": t,
                              "model": model,
                              "fibre_type": fibre_type,
                              "porosity": porosity,
                              "tortuosity": tortuosity,
                              "viscous_length [µm]": viscous_length,
                              "thermal_length [µm]": thermal_length,
                              "thermal_permeability [m²]": thermal_permeability,
                              "static_tortuosity": static_tortuosity,
                              "static_thermal_tortuosity": static_thermal_tortuosity,
                              "frame": frame,
                              "bulk_density [kg/m³]": bulk_density,
                              "preset": preset,
                              "preset_overrides": preset_overrides,
                              "pore_shape_factor": pore_shape_factor,
                              "perforation_diameter [mm]": perforation_diameter,
                              "perforation_spacing [mm]": perforation_spacing,
                              "rigid_backed": rigid_backed,
                              # "kc": kc,
                              # "zc": zc,
                              "matrix": Tp,
                              }

    def air_layer(self, t=5, layer=None):
        """
        Add an air layer.

        For an inviscid air layer under oblique plane-wave incidence, the tangential wavenumber is conserved and the
        normal propagation constant is ``k0*cos(theta)``. The pressure/volume-velocity transfer matrix therefore uses
        the normal air impedance ``Z0/cos(theta)``: ``A = D = cos(k0*t*cos(theta))``,
        ``B = j*Z0/cos(theta)*sin(k0*t*cos(theta))`` and ``C = j*cos(theta)/Z0*sin(k0*t*cos(theta))`` before area
        scaling.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the air layer [mm]
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters

        theta = np.deg2rad(np.asarray(self.incidence_angle, dtype=float))[None, :]
        cos_theta = np.cos(theta)
        k0 = np.asarray(self.k0, dtype=float)[:, None] * cos_theta
        z0 = np.ones((len(self.freq), 1), dtype=float) * self.z0 / cos_theta

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
        Set a constant surface impedance over the current frequency and incidence grids.

        This is an auxiliary direct-assignment utility rather than a physical transfer-matrix layer. The supplied scalar
        is interpreted as a specific surface impedance in the same units as ``self.z``. It is copied to ``self.z`` with
        length ``len(self.freq)`` and to ``self.z_angle`` with shape ``(len(self.freq), len(self.incidence_angle))``.
        ``z=self.z0`` gives a matched surface, ``z=0`` a pressure-release surface and a very large value a rigid wall.
        For angle incidence and ``diffuse_method="paris"``, the cached absorption coefficient is calculated from the
        configured incidence angles.

        Parameters
        ----------
        z : float or complex, optional
            Specific surface impedance value to assign at every frequency and configured incidence angle.
        """
        dtype = np.result_type(z, float)
        self.z = np.full(len(self.freq), z, dtype=dtype)
        self.z_angle = np.full((len(self.freq), len(self.incidence_angle)), z, dtype=dtype)
        if self.incidence == "angle":
            _, alpha = self.reflection_and_absorption_coefficient(self.z, angle=self.incidence_angle[0])
            self._alpha = np.asarray(alpha, dtype=float).reshape((len(self.freq),))
        elif self.incidence == "diffuse" and self.diffuse_method == "paris":
            self._alpha = self.diffuse_absorption_coefficient(self.z_angle)
        self.matrix = {"termination": {"type": "constant_z"}}

    def membrane_layer(self, t=1, rho=8050, layer=None, *, mechanical_resistance=0):
        """
        Add a limp membrane or mass-sheet layer.

        The membrane is modeled as a local pressure-jump impedance with surface mass ``m_s = rho*t`` and specific
        impedance ``Z_m = r_m*rho0*c0 + 1j*omega*m_s``, where ``t`` is converted from millimeters to meters and ``r_m``
        is the mechanical loss resistance normalized by ``rho0*c0``. The transfer matrix is ``[[1, Z_m/S0], [0, 1]]``
        because this class uses pressure and volume velocity as the matrix state; ``S0`` is the front surface area. The
        resulting surface impedance recovered by ``compute()`` is scaled back to specific impedance.

        With ``r_m = 0`` this is the mass term of the classical limp-membrane absorber model. Brandão [1], section
        2.3.2, writes ``Z_m = 1j*omega*m_s`` after neglecting membrane resistance and combines it in series with the
        surface impedance of the backing cavity, which here comes from the layers behind the membrane. Kang and Fuchs
        [2], equation 1, keep the resistance in the normalized impedance ``z_M = r' + 1j*omega*m'/(rho0*c0)``, where
        ``r'`` "depends mainly on mounting conditions" and is around 1 in their calculations. Sakagami et al. [3] call
        it the mechanical loss resistance. ``r_m`` is ``r'``.

        ``r_m`` is the only loss of the sheet's motion. It sets the height and width of the absorption peak at the
        resonance of the membrane mass with the layers behind it. At that resonance a membrane over a rigidly backed air
        space has the impedance ``r_m*rho0*c0``, so ``r_m = 1`` gives the highest peak at normal incidence ([3]).
        Without it, a membrane over a lossless air space absorbs nothing.

        The model neglects bending stiffness and finite-panel modal behavior. It is appropriate for a limp
        local-reaction mass sheet, not a full structural panel model. ``thin_plate_layer`` adds the bending stiffness of
        an infinite plate, which changes the impedance at oblique incidence near and above the critical frequency.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the membrane [mm]
        rho : float or int, optional
            Material density [kg/m³]
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the membrane motion, mainly from mounting losses, normalized by
            ``rho0*c0`` [-].

        References
        ----------
        [1] E. Brandão, "Acústica de Salas: Projeto e Modelagem", Blucher, São Paulo, 2018.
        [2] J. Kang and H. V. Fuchs, "Predicting the absorption of open weave textiles and micro-perforated
            membranes backed by an air space", Journal of Sound and Vibration, vol. 220, pp. 905-920, 1999.
        [3] K. Sakagami, M. Morimoto and M. Yairi, "A note on the relationship between the sound absorption by
            microperforated panels and panel/membrane-type absorbers", Applied Acoustics, vol. 70, pp. 1131-1136, 2009.
        """
        if mechanical_resistance < 0:
            raise ValueError("mechanical_resistance must not be negative.")
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters

        zc = (mechanical_resistance * self.rho0 * self.c0 + 1j * self.w0 * rho * t_meters) / self.s0
        zc = self._broadcast_to_incidence_angles(zc)

        ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

        Tm = np.array([[ones, zc],
                       [zeros, ones]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "membrane_layer",
                              "thickness [mm]": t,
                              "density [kg/m³]": rho,
                              "mechanical_resistance [-]": mechanical_resistance or None,
                              "matrix": Tm,
                              }

    def thin_plate_layer(self, t=12.5, rho=800, E=4.1, nu=0.3, loss_factor=0.01, layer=None, *,
                         mechanical_resistance=0):
        """
        Add an infinite thin plate with bending stiffness as a series-impedance layer.

        The plate is a Kirchhoff plate with surface mass ``m = rho*t`` and bending stiffness
        ``D = E*t**3/(12*(1 - nu**2))``, where ``t`` and ``E`` are converted to meters and pascals. At the trace
        wavenumber ``kx = k0*sin(theta)`` of the incident wave its specific impedance is
        ``Z_w = eta*A + 1j*(omega*m - A)`` with ``A = D*kx**4/omega`` and ``eta`` the loss factor, Vigran [1], section
        6.5.1.2, equation 6.90 with the complex stiffness ``D*(1 + 1j*eta)``. With the critical frequency
        ``f_c = c0**2/(2*pi)*sqrt(m/D)`` this is ``1j*omega*m*(1 - (f/f_c)**2*(1 + 1j*eta)*sin(theta)**4)``, Vigran [1],
        equation 6.91, and Mechel [2], section I.8, equation 11.
        The transfer matrix is ``[[1, Z/S0], [0, 1]]`` with ``Z = Z_w + r_m*rho0*c0``. ``kx`` is the same in every
        layer, so ``A`` does not depend on the media next to the plate. The mounting resistance ``r_m`` is the membrane
        resistance ``r'`` of Kang and Fuchs [3], equation 1, added to the plate impedance as a package extension.

        Compared with ``membrane_layer``, the plate adds the bending term ``A``. At normal incidence ``A = 0`` and the
        layer is identical to ``membrane_layer`` with the same ``t``, ``rho`` and ``mechanical_resistance``. At oblique
        incidence the stiffness lowers the mass reactance by the fraction ``(f/f_c)**2*sin(theta)**4``, which is small
        well below ``f_c``. At ``f = f_c/sin(theta)**2`` the reactance vanishes (coincidence) and the impedance is
        ``eta*omega*m + r_m*rho0*c0``. Above it the stiffness dominates. ``eta`` acts only through ``A``, so it does not
        damp the plate at normal incidence or well below ``f_c``. A resonant panel absorber still needs
        ``mechanical_resistance`` or a damped cavity. Neither layer models finite size, edge conditions or panel modes.

        Parameters
        ----------
        t : float, optional
            Plate thickness [mm]
        rho : float, optional
            Plate density [kg/m³]
        E : float, optional
            Young's modulus [GPa]
        nu : float, optional
            Poisson's ratio [-]
        loss_factor : float, optional
            Loss factor ``eta`` of the bending stiffness [-]
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the plate motion, mainly from mounting losses, normalized by
            ``rho0*c0`` [-].

        Notes
        -----
        Thin-plate theory holds while the bending wavelength is longer than six plate thicknesses ([1], section
        3.7.3.1; [4], section 5.2.7.1). A ``UserWarning`` is raised when the frequency range goes above this limit and
        an incidence angle is oblique. At normal incidence the bending term is zero and the limit does not apply.

        With a small loss factor and no other loss in the stack, the mass-spring resonance behind the plate is narrow
        and moves with the angle of incidence. Diffuse results then depend on the ``incidence_angle`` step, as they do
        for ``membrane_layer`` with a small resistance. A finer step or a porous layer in the cavity resolves the
        resonance, and a larger ``mechanical_resistance`` widens it.

        References
        ----------
        [1] T. E. Vigran, "Building Acoustics", Taylor & Francis, 2008.
        [2] F. P. Mechel (ed.), "Formulas of Acoustics", 2nd edition, Springer, Berlin, 2008.
        [3] J. Kang and H. V. Fuchs, "Predicting the absorption of open weave textiles and micro-perforated
            membranes backed by an air space", Journal of Sound and Vibration, vol. 220, pp. 905-920, 1999.
        [4] T. E. Vigran, "Manual for NorFLAG, version 4.0", Trondheim, 2018.
        """
        values = (t, rho, E, nu, loss_factor, mechanical_resistance)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("thin_plate_layer needs finite t, rho, E, nu, loss_factor and mechanical_resistance.")
        if t <= 0 or rho <= 0 or E <= 0:
            raise ValueError("thin_plate_layer needs t > 0, rho > 0 and E > 0.")
        if not -1 < nu < 0.5:
            raise ValueError("nu must lie between -1 and 0.5.")
        if loss_factor < 0:
            raise ValueError("loss_factor must not be negative.")
        if mechanical_resistance < 0:
            raise ValueError("mechanical_resistance must not be negative.")

        t_meters = t / 1000
        mass = rho * t_meters
        stiffness = E * 1e9 * t_meters ** 3 / (12 * (1 - nu ** 2))
        # Bending wavelength 2*pi/k_B equals 6*t where k_B**2 = omega*sqrt(m/D).
        f_thin = (2 * np.pi / (6 * t_meters)) ** 2 * np.sqrt(stiffness / mass) / (2 * np.pi)
        # The bending term is zero at normal incidence, so the limit only matters when an oblique angle is computed.
        # checked_fmax records the highest frequency the limit was checked against (0 when no angle was oblique).
        checked_fmax = float(np.max(self.freq)) if self._has_oblique_angle() else 0.0
        if checked_fmax > f_thin:
            warnings.warn(f"thin_plate_layer: the bending wavelength is shorter than 6 plate thicknesses above "
                          f"{f_thin:.0f} Hz, where thin-plate theory loses accuracy.", UserWarning, stacklevel=2)

        omega = self.w0[:, None]
        sin_theta = np.sin(np.deg2rad(np.asarray(self.incidence_angle, dtype=float)))[None, :]
        # A = D*kx**4/omega with kx = omega*sin(theta)/c0, written without the division so that f = 0 stays finite.
        bending = stiffness * omega ** 3 * (sin_theta / self.c0) ** 4
        zc = (mechanical_resistance * self.rho0 * self.c0 + loss_factor * bending
              + 1j * (omega * mass - bending)) / self.s0
        zc = np.broadcast_to(zc, (len(self.freq), len(self.incidence_angle)))

        ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

        Tm = np.array([[ones, zc],
                       [zeros, ones]])

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "thin_plate_layer",
                              "thickness [mm]": t,
                              "density [kg/m³]": rho,
                              "young_modulus [GPa]": E,
                              "poisson_ratio [-]": nu,
                              "loss_factor [-]": loss_factor,
                              "mechanical_resistance [-]": mechanical_resistance or None,
                              "critical_frequency [Hz]": float(self.c0 ** 2 / (2 * np.pi) * np.sqrt(mass / stiffness)),
                              "thin_plate_limit [Hz]": float(f_thin),
                              "checked_fmax [Hz]": checked_fmax,
                              "matrix": Tm,
                              }

    def porous_facing_layer(self, t=1, surface_mass=None, resistance=None, porosity=0.7, tortuosity=1.0,
                            method="rebillard", free=True, rho=None, d=None, s=None, open_area=None,
                            layer=None, *, in_contact=False):
        """
        Add a thin porous-facing or fabric sheet as a local pressure-jump layer.

        The layer is represented by a zero-thickness transfer matrix ``[[1, Zf/S0], [0, 1]]``, where ``Zf`` is the
        specific sheet impedance and ``S0`` is the front surface area used by the volume-velocity matrix state.

        ``method="rebillard"`` implements the unbonded thin porous-facing expression from Rebillard et al. [1]. The
        implementation uses the negligible-flexural-stiffness limit. With open porosity ``h``, tortuosity ``tau``,
        facing thickness ``e``, surface mass ``m``, and flow resistance ``R = sigma*e``, the coupling term is
        ``I = 1j*omega*(tau - 1)*h*rho0*e + R*h**2``.

        ``method="pieren"`` implements the thin woven-fabric sheet impedance described by Pieren [2]. With
        ``free=True``, the sheet resistance and limp surface mass are combined in parallel,
        ``Zf = 1j*omega*m*R / (1j*omega*m + R)``. With ``free=False``, the motionless fabric limit ``Zf = R`` is used.
        The aliases ``method="pieren_free"`` and ``method="pieren_fixed"`` set this option explicitly.

        Measured ``surface_mass`` and ``resistance`` are preferred for real fabrics. If ``surface_mass`` is ``None`` and
        ``rho`` is supplied, it is estimated as ``rho*t``; if ``resistance`` is ``None``, it is estimated from the ideal
        circular-pore expression ``32*eta*t/(open_area*d**2)``. For ``method="rebillard"``, omitting both uses a nominal
        thin facing with ``surface_mass=0.25`` kg/m2 and ``resistance=200`` Pa*s/m.

        By default the sheet acts at the mean surface velocity. This describes a sheet further than about one hole
        diameter from the openings of a perforated plate, as in Cox and D'Antonio [3], equation 7.18. A sheet in contact
        with the plate, within about one hole diameter of the openings ([3], section 7.3.3.1), carries the flow through
        the openings only. Its impedance then acts as ``Zf/phi`` for open-area ratio ``phi`` ([3], equation 7.17; Vigran
        [4], section 5.4.1.5, for a fabric in contact with the panel).

        With ``in_contact=True`` the layer's own matrix is the identity, and ``compute()`` adds ``Zf`` to the opening
        branch of the plate directly before or after it: ``perforated_panel_layer``, ``slotted_panel_layer`` or
        ``rectangular_slot_layer``. There ``Zf/phi`` enters before any plate-mass coupling. [3] and [4] give the
        resistance part; dividing the full sheet impedance, mass included, by ``phi`` is a package extension. On the
        exit face of a plate with ``method="conical"``, ``phi`` is the open area of that face (see the plate methods).
        ``compute()`` raises if no such plate lies next to the sheet or if plates lie on both sides.

        Parameters
        ----------
        t : float or int, optional
            Facing thickness [mm].
        surface_mass : float or None, optional
            Mass per unit area [kg/m2]. If ``None`` and ``rho`` is supplied, it is estimated as ``rho*t``.
        resistance : float or None, optional
            Specific airflow resistance [Pa*s/m]. If ``None``, it is estimated from ``d`` and ``open_area``.
        porosity : float, optional
            Open porosity used by the Rebillard model.
        tortuosity : float, optional
            Tortuosity used by the Rebillard model.
        method : string, optional
            ``"rebillard"``, ``"pieren"``, ``"pieren_free"``, or ``"pieren_fixed"``.
        free : bool, optional
            For ``method="pieren"``, choose the free-moving fabric expression when ``True`` or the motionless
            resistance-only limit when ``False``.
        rho : float or None, optional
            Apparent fabric density [kg/m3], used only to estimate ``surface_mass`` when needed.
        d : float or None, optional
            Equivalent pore diameter [mm], used only to estimate ``resistance``.
        s : float or None, optional
            Equivalent pore spacing [mm], used to estimate ``open_area`` when needed.
        open_area : float or None, optional
            Open-area ratio used for the geometric resistance estimate.
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.
        in_contact : bool, optional
            If ``True``, the sheet rests on the plate directly before or after it, within about one hole diameter of its
            openings, and acts at the opening velocity, as described above.

        References
        ----------
        [1] P. Rebillard, J.-F. Allard, C. Depollier, P. Guignouard, W. Lauriks, C. Verhaegen and A. Cops, "The
            effect of a porous facing on the impedance and the absorption coefficient of a layer of porous material",
            Journal of Sound and Vibration, vol. 156, pp. 541-555, 1992.
        [2] R. Pieren, "Sound absorption modeling of thin woven fabrics backed by an air cavity", Textile Research
            Journal, vol. 82, pp. 864-874, 2012.
        [3] T. J. Cox and P. D'Antonio, "Acoustic Absorbers and Diffusers: Theory, Design and Application", 3rd ed.,
            CRC Press, 2017.
        [4] T. E. Vigran, "Building Acoustics", Taylor & Francis, 2008.
        """
        method = method.lower()
        if method == "pieren_free":
            method = "pieren"
            free = True
        elif method == "pieren_fixed":
            method = "pieren"
            free = False
        elif method not in {"rebillard", "pieren"}:
            raise ValueError("porous_facing_layer method must be 'rebillard', 'pieren', 'pieren_free', "
                             "or 'pieren_fixed'.")

        t_meters = t / 1000
        if t_meters <= 0:
            raise ValueError("Facing thickness must be greater than zero.")

        if method == "rebillard":
            if surface_mass is None and rho is None:
                surface_mass = 0.25
            if resistance is None and d is None:
                resistance = 200

        if surface_mass is None and rho is not None:
            surface_mass = rho * t_meters

        if resistance is None:
            if d is None:
                raise ValueError("resistance must be supplied, or d and open_area/s must be supplied to estimate it.")
            d_meters = d / 1000
            if d_meters <= 0:
                raise ValueError("Equivalent pore diameter must be greater than zero.")
            if open_area is None:
                if s is None:
                    raise ValueError("open_area or s must be supplied when estimating resistance.")
                s_meters = s / 1000
                open_area = np.pi / ((2 * s_meters / d_meters) ** 2)
            resistance = 32 * self.air_prop["air_viscosity"] * t_meters / (open_area * d_meters**2)
        elif open_area is None and d is not None and s is not None:
            d_meters = d / 1000
            s_meters = s / 1000
            open_area = np.pi / ((2 * s_meters / d_meters) ** 2)

        if open_area is not None and not 0 < open_area <= 1:
            raise ValueError("Open area must be between 0 and 1.")
        if resistance < 0:
            raise ValueError("Facing resistance cannot be negative.")

        if method == "pieren":
            if free and surface_mass is None:
                raise ValueError("surface_mass or rho must be supplied for a free Pieren fabric.")
        else:
            if surface_mass is None:
                raise ValueError("surface_mass or rho must be supplied for the Rebillard porous-facing model.")
            if not 0 < porosity <= 1:
                raise ValueError("porosity must be greater than zero and less than or equal to one.")
            if tortuosity <= 0:
                raise ValueError("tortuosity must be greater than zero.")

        ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        if in_contact:
            # The sheet acts on the openings of the plate it rests on, which adds it in compute().
            Tf = np.array([[ones, zeros],
                           [zeros, ones]])
        else:
            z_sheet = self._porous_facing_impedance(t_meters, surface_mass, resistance, porosity, tortuosity, method,
                                                    free)
            z_sheet = self._broadcast_to_incidence_angles(z_sheet / self.s0)
            Tf = np.array([[ones, z_sheet],
                           [zeros, ones]])

        if layer is None:
            layer = len(self.matrix)

        matrix_data = {"type": "porous_facing_layer",
                       "thickness [mm]": t,
                       "surface_mass [kg/m2]": surface_mass,
                       "resistance [Pa*s/m]": resistance,
                       "porosity": porosity,
                       "tortuosity": tortuosity,
                       "method": method,
                       "rho [kg/m3]": rho,
                       "pore_diameter [mm]": d,
                       "pore_spacing [mm]": s,
                       "open_area [%]": None if open_area is None else open_area * 100,
                       "in_contact": True if in_contact else None,
                       "matrix": Tf,
                       }
        if method == "pieren":
            matrix_data["free"] = free
        self.matrix[layer] = matrix_data

    def _porous_facing_impedance(self, t_meters, surface_mass, resistance, porosity, tortuosity, method, free):
        """
        Return the specific sheet impedance ``Zf`` [Pa*s/m] of ``porous_facing_layer`` on the frequency grid.
        """
        if method == "pieren":
            if not free:
                return np.full_like(self.freq, resistance, dtype=complex)
            jw_mass = 1j * self.w0 * surface_mass
            return jw_mass * resistance / (jw_mass + resistance)
        rho2 = porosity * self.rho0
        rho_a = (tortuosity - 1.0) * rho2
        coupling = 1j * self.w0 * rho_a * t_meters + resistance * porosity**2
        c_term = 1j * self.w0 * rho2 * t_meters + coupling
        b_term = 1j * self.w0 * surface_mass + coupling
        denominator = porosity * (coupling * (1.0 - porosity) + b_term * porosity) + (
            1.0 - porosity
        ) * (coupling * porosity + c_term * (1.0 - porosity))
        return (c_term * b_term - coupling**2) / denominator

    def _contact_facings(self, layer):
        """Return ``(side, record)`` for each in-contact ``porous_facing_layer`` directly before (-1) or after (+1)."""
        found = []
        for side in (-1, 1):
            record = self.matrix.get(layer + side)
            if isinstance(record, dict) and record.get("type") == "porous_facing_layer" and record.get("in_contact"):
                found.append((side, record))
        return found

    def _contact_facing_impedance(self, layer, side=None):
        """
        Return the summed sheet impedance ``Zf`` [Pa*s/m] of the in-contact facings next to ``layer``, or 0.

        With ``side`` set to -1 or +1, only the facing on that side is included.
        """
        total = 0.0
        for facing_side, record in self._contact_facings(layer):
            if side is None or side == facing_side:
                total = total + self._porous_facing_impedance(
                    record["thickness [mm]"] / 1000, record["surface_mass [kg/m2]"], record["resistance [Pa*s/m]"],
                    record["porosity"], record["tortuosity"], record["method"], record.get("free", True))
        return total

    def _contact_facing_plates(self):
        """
        Return the keys of the plates that carry an in-contact ``porous_facing_layer``.

        Raises if such a sheet has no plate directly before or after it, or has plates on both sides.
        """
        aperture_types = {"perforated_panel_layer", "slotted_panel_layer", "rectangular_slot_layer"}
        plates = set()
        for key, record in self.matrix.items():
            if not (isinstance(record, dict) and record.get("type") == "porous_facing_layer"
                    and record.get("in_contact")):
                continue
            neighbours = [n for n in (key - 1, key + 1)
                          if isinstance(self.matrix.get(n), dict) and self.matrix[n].get("type") in aperture_types]
            if not neighbours:
                raise ValueError(f"Layer {key} is a porous_facing_layer with in_contact=True, but no "
                                 "perforated_panel_layer, slotted_panel_layer or rectangular_slot_layer lies directly "
                                 "before or after it.")
            if len(neighbours) > 1:
                raise ValueError(f"Layer {key} is a porous_facing_layer with in_contact=True between two plates; place "
                                 "it directly next to the one plate it rests on.")
            plates.add(neighbours[0])
        return plates

    @staticmethod
    def _validate_mechanical_resistance(mechanical_resistance, rho):
        """Raise if the mechanical resistance is negative or set on a rigid plate."""
        if mechanical_resistance < 0:
            raise ValueError("mechanical_resistance must not be negative.")
        if mechanical_resistance and not rho:
            raise ValueError("mechanical_resistance acts on the plate motion and needs rho; a rigid plate does not "
                             "move.")

    @staticmethod
    def _circular_hole_end_length(d_meters, open_area, end_correction):
        """
        Return the total end-correction length of both hole ends [m] for ``perforated_panel_layer``.
        """
        xi = np.sqrt(open_area)
        if end_correction == "ingard":
            # Vigran's fit to Ingard's circular aperture in a square tube, both ends: 0.96*sqrt(A0)*(...).
            return 0.96 * np.sqrt(np.pi * d_meters ** 2 / 4) * (1 - 1.24 * xi + 0.27 * xi ** 2)
        if end_correction == "nesterov":
            # Cox and D'Antonio, Table 7.1: circular holes in a circular pattern.
            return 0.85 * d_meters * (1 - 1.47 * xi + 0.47 * xi ** 3)
        if end_correction in {"jb", "jaouen_becot"}:
            # Cox and D'Antonio, Table 7.1: circular holes in a square pattern.
            return 0.85 * d_meters * (1 - 1.13 * xi - 0.09 * xi ** 2 + 0.27 * xi ** 3)
        if end_correction == "beranek":
            # Single circular hole in an infinite baffle.
            return 0.85 * d_meters
        raise ValueError(
            "perforated_panel_layer end_correction must be 'ingard', 'jb', 'jaouen_becot', 'nesterov', or 'beranek'."
        )

    @staticmethod
    def _validate_porous_contact(porous_contact):
        """Raise if the porous-contact factor lies outside ``[0, 1]``."""
        if not 0 <= porous_contact <= 1:
            raise ValueError("porous_contact must be between 0 and 1.")

    def _porous_contact_resistance(self, layer, porous_contact, end_length):
        """
        Return the porous-contact resistance ``K*sigma*delta`` [Pa*s/m] of the sheet stored at ``layer``.

        ``sigma`` sums the flow resistivity of the ``porous_layer`` entries directly before and after ``layer``, and
        ``end_length`` is the one-side end correction ``delta`` [m].
        """
        if not porous_contact:
            return 0.0
        sigma = 0.0
        for neighbour in (layer - 1, layer + 1):
            record = self.matrix.get(neighbour)
            if isinstance(record, dict) and record.get("type") == "porous_layer":
                sigma += record["flow_resistivity [k*Pa*s/m²]"] * 1000
        return porous_contact * sigma * end_length

    def _bessel_ingard_circular_sheet_impedance(self, t_meters, d_meters, open_area, end_length, rho=None,
                                                resistance=0.0, mechanical_resistance=0.0):
        """
        Return the local sheet impedance for ``perforated_panel_layer(method="bessel_ingard")``.

        The Bessel short-tube impedance is evaluated over the hole length extended by ``end_length``. The series term
        ``resistance`` [Pa*s/m], the porous-contact resistance plus any in-contact facing impedance and so possibly
        complex, is added as ``resistance/phi``. The optional full-surface plate mass, with ``mechanical_resistance``
        (normalized by ``rho0*c0``) in series, is coupled in parallel.
        """
        radius = d_meters / 2
        viscosity = self.air_prop["air_viscosity"]
        kd = radius * np.sqrt(self.rho0 * self.w0 / viscosity)
        bessel_argument = kd * np.sqrt(-1j)
        aperture_factor = 1 - 2 * jv(1, bessel_argument) / (bessel_argument * jv(0, bessel_argument))
        z_sheet = 1j * self.w0 * self.rho0 * (t_meters + end_length) / (open_area * aperture_factor)
        z_sheet = z_sheet + resistance / open_area

        if rho is not None:
            plate_mass = mechanical_resistance * self.rho0 * self.c0 + 1j * self.w0 * rho * t_meters
            z_sheet = 1 / (1 / z_sheet + 1 / plate_mass)

        return z_sheet

    @staticmethod
    def _mouth_radiation(shape, x):
        """
        Return the radiation impedance of a baffled aperture mouth, normalized by ``rho0*c0``.

        ``shape="hole"`` is the circular piston of Vigran (2004), equation 9, and ``shape="slot"`` the strip of equation
        20, both at ``x = 2*k*a`` for radius or half-width ``a``. A complex ``x`` (a porous medium) must have
        ``Re(x) > 0`` and ``Im(x) <= 0``, as a passive wavenumber gives. It uses the power series of the Struve
        functions ``S0`` and ``S1`` below ``|x| = 18`` (F. W. J. Olver, D. W. Lozier, R. F. Boisvert and C. W. Clark,
        eds., "NIST Handbook of Mathematical Functions", Cambridge University Press, 2010, equation 11.2.1, where
        ``S_n`` is written ``H_n``). Above it the Bessel and Struve terms grow as ``exp(|Im(x)|)`` and cancel, so
        ``J1`` is written as ``H1 + 1j*Y1``, with ``H1`` the Hankel function of the second kind, and the bounded
        differences ``K_n = S_n - Y_n`` (equation 11.2.5) are taken from their asymptotic series (equation 11.6.1),
        truncated at the smallest term. For the strip, ``H1*Y0 - H0*Y1 = J1*Y0 - J0*Y1 = 2/(pi*x)``, the Wronskian of
        equation 10.5.2.
        """
        x = np.asarray(x)
        if not (np.iscomplexobj(x) and np.any(x.imag != 0)):
            x = x.real
            if shape == "hole":
                return 1 - 2 * jv(1, x) / x + 2j * special.struve(1, x) / x
            h0, h1 = special.hankel2(0, x), special.hankel2(1, x)
            return (x * (h0 + np.pi / 2 * (h1 * special.struve(0, x) - h0 * special.struve(1, x))) - h1
                    + 2j / (np.pi * x))
        x = x.astype(complex)
        if np.any(x.real <= 0) or np.any(x.imag > 0):
            raise ValueError("The mouth radiation is evaluated for Re(x) > 0 and Im(x) <= 0 only.")
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        out = np.empty_like(x)
        small = np.abs(x) < 18
        z = x[small]
        if shape == "hole":
            out[small] = 1 - 2 * jv(1, z) / z + 2j * TMM._struve_series(1, z) / z
        else:
            h0, h1 = special.hankel2(0, z), special.hankel2(1, z)
            out[small] = (z * (h0 + np.pi / 2 * (h1 * TMM._struve_series(0, z) - h0 * TMM._struve_series(1, z))) - h1
                          + 2j / (np.pi * z))
        z = x[~small]
        h0, h1 = special.hankel2(0, z), special.hankel2(1, z)
        if shape == "hole":
            out[~small] = 1 - 2 * h1 / z + 2j * TMM._struve_minus_y(1, z) / z
        else:
            bracket = h1 * TMM._struve_minus_y(0, z) - h0 * TMM._struve_minus_y(1, z) + 2 / (np.pi * z)
            out[~small] = z * (h0 + np.pi / 2 * bracket) - h1 + 2j / (np.pi * z)
        return out[0] if scalar else out

    @staticmethod
    def _struve_series(n, z, terms=80):
        """Return the Struve function ``S_n(z)`` (``n`` = 0 or 1) from its power series, NIST Handbook eq. 11.2.1."""
        out = np.zeros_like(z)
        half = z / 2
        power = half ** (n + 1)
        for k in range(terms):
            out = out + (-1) ** k * power / (special.gamma(k + 1.5) * special.gamma(k + n + 1.5))
            power = power * half ** 2
        return out

    @staticmethod
    def _struve_minus_y(n, z, terms=40):
        """
        Return ``K_n(z) = S_n(z) - Y_n(z)`` (``n`` = 0 or 1) from the asymptotic series of NIST Handbook eq. 11.6.1,
        ``(1/pi)*sum(Gamma(k + 1/2)*(z/2)**(n - 2*k - 1)/Gamma(n + 1/2 - k))``, truncated at its smallest term.
        """
        out = np.zeros_like(z)
        smallest = np.full(z.shape, np.inf)
        stopped = np.zeros(z.shape, dtype=bool)
        for k in range(terms):
            term = special.gamma(k + 0.5) * (z / 2) ** (n - 2 * k - 1) / special.gamma(n + 0.5 - k) / np.pi
            stopped |= np.abs(term) > smallest
            out = np.where(stopped, out, out + term)
            smallest = np.where(stopped, smallest, np.abs(term))
        return out

    def _tapered_aperture_impedance(self, shape, t_meters, a, b, zb):
        """
        Return the input impedance ``Zi`` [Pa*s/m] at the narrow mouth of a conical hole or wedge-shaped slot.

        Closed form of Vigran (2004), equations 16 and 17 (holes) and 23 and 25 (slots). The exit rows of the system
        fix the ratio ``B2/A2`` of the backward and forward waves from the normalized exit load ``zb``, and ``Zi``
        follows at ``x1 = a*h/(b - a)``. The entrance radiation does not enter ``Zi``.
        """
        k = self.k0
        z0 = self.rho0 * self.c0
        x1, x2 = a * t_meters / (b - a), b * t_meters / (b - a)
        if shape == "hole":
            e1, e2 = np.exp(-1j * k * x1), np.exp(-1j * k * x2)
            ratio = -e2 ** 2 * ((1 + 1j * k * x2) * zb - 1j * k * x2) / ((1 - 1j * k * x2) * zb - 1j * k * x2)
            pressure = e1 + ratio / e1
            velocity = (1 + 1j * k * x1) * e1 + ratio * (1 - 1j * k * x1) / e1
            return 1j * z0 * k * x1 * pressure / velocity
        h0m2, h1m2 = special.hankel2(0, k * x2), special.hankel2(1, k * x2)
        h0p2, h1p2 = special.hankel1(0, k * x2), special.hankel1(1, k * x2)
        ratio = (1j * h0m2 - zb * h1m2) / (1j * h0p2 - zb * h1p2)
        h0m1, h1m1 = special.hankel2(0, k * x1), special.hankel2(1, k * x1)
        h0p1, h1p1 = special.hankel1(0, k * x1), special.hankel1(1, k * x1)
        return 1j * z0 * (h0m1 - ratio * h0p1) / (h1m1 - ratio * h1p1)

    def _porous_record_properties(self, record, layer=None):
        """
        Return ``(kc, zc)`` of a stored ``porous_layer`` record on the frequency grid, as ``porous_layer`` does.

        For a double-porosity record at ``layer``, whether it lies on the rigid wall is taken from its position in the
        current stack, since the stored value may predate the backing of this ``compute()``.
        """
        def scaled(key, factor):
            value = record.get(key)
            return None if value is None else value * factor

        kc, zc = self.equivalent_fluid_model(
            record["flow_resistivity [k*Pa*s/m²]"] * 1000,
            model=record["model"],
            fibre_type=record.get("fibre_type", 1),
            porosity=record.get("porosity"),
            tortuosity=record.get("tortuosity"),
            viscous_length=scaled("viscous_length [µm]", 1e-6),
            thermal_length=scaled("thermal_length [µm]", 1e-6),
            thermal_permeability=record.get("thermal_permeability [m²]"),
            static_tortuosity=record.get("static_tortuosity"),
            static_thermal_tortuosity=record.get("static_thermal_tortuosity"),
            frame=record.get("frame", "rigid"),
            bulk_density=record.get("bulk_density [kg/m³]"),
            pore_shape_factor=record.get("pore_shape_factor"),
            perforation_diameter=scaled("perforation_diameter [mm]", 1e-3),
            perforation_spacing=scaled("perforation_spacing [mm]", 1e-3),
            thickness=record["thickness [mm]"] / 1000,
            rigid_backed=(self._rigid_backed(layer) if layer is not None and record["model"] == "double_porosity"
                          else record.get("rigid_backed")),
        )
        return np.asarray(kc, dtype=complex).reshape(-1), np.asarray(zc, dtype=complex).reshape(-1)

    def _exit_porous_layer(self, layer):
        """
        Return the key of the ``porous_layer`` behind the plate at ``layer``, or ``None``.

        An in-contact ``porous_facing_layer`` on the exit face is a sheet resting on the plate, so the lookup steps
        over it to the layer behind.
        """
        behind = layer + 1
        record = self.matrix.get(behind)
        if isinstance(record, dict) and record.get("type") == "porous_facing_layer" and record.get("in_contact"):
            behind += 1
            record = self.matrix.get(behind)
        return behind if isinstance(record, dict) and record.get("type") == "porous_layer" else None

    @staticmethod
    def _validate_exit_radiation(exit_radiation, method):
        """Raise for an unknown ``exit_radiation`` or one set on a method without a conical aperture."""
        if exit_radiation not in {"air", "porous", "mean"}:
            raise ValueError("exit_radiation must be 'air', 'porous', or 'mean'.")
        if method != "conical" and exit_radiation != "air":
            raise ValueError("exit_radiation is only used by method='conical'.")

    @staticmethod
    def _warn_cone_angle(t, entrance, exit_size, layer_type):
        """Warn when the cone or wedge angle, taken as the half-angle, exceeds the about 30 degrees of the source."""
        half_angle = np.degrees(np.arctan((exit_size - entrance) / (2 * t)))
        if half_angle > 30:
            warnings.warn(f"{layer_type}(method='conical'): the aperture half-angle is {half_angle:.0f} degrees; the "
                          "source limits the cone angle, taken here as the half-angle, to about 30 degrees.",
                          UserWarning, stacklevel=3)

    def _conical_sheet_impedance(self, shape, layer, t_meters, a, b, open_area, open_area_exit, open_area_mean,
                                 exit_radiation, rho, mechanical_resistance):
        """
        Return the specific sheet impedance [Pa*s/m] of a plate with conical holes or wedge-shaped slots.

        ``Zp = (Zi + Zf_entrance)/phi + Zf_exit/phi_exit`` (Vigran 2004, equation 27), with the in-contact facing
        impedances ``Zf`` of each face. With ``exit_radiation="porous"`` or ``"mean"`` and a ``porous_layer`` behind the
        plate (see ``_exit_porous_layer``), the exit mouth radiates into that medium (characteristic impedance ``zc``,
        wavenumber ``kc``) or into the mean of it and air; without that neighbour the air load is used here and
        ``compute()`` raises. The optional plate mass uses the mean open area through the plate.
        """
        z0 = self.rho0 * self.c0
        zb = self._mouth_radiation(shape, 2 * self.k0 * b)
        neighbour = self._exit_porous_layer(layer) if exit_radiation != "air" else None
        if neighbour is not None:
            kc, zc = self._porous_record_properties(self.matrix[neighbour], neighbour)
            zb_porous = zc / z0 * self._mouth_radiation(shape, 2 * kc * b)
            zb = zb_porous if exit_radiation == "porous" else (zb + zb_porous) / 2
        zi = self._tapered_aperture_impedance(shape, t_meters, a, b, zb)
        z_sheet = ((zi + self._contact_facing_impedance(layer, side=-1)) / open_area
                   + self._contact_facing_impedance(layer, side=1) / open_area_exit)
        if rho:
            # Package extension: plate motion with the solid fraction of the mean open area through the plate
            plate = mechanical_resistance * z0 + 1j * self.w0 * rho * t_meters * (1 - open_area_mean)
            z_sheet = 1 / (1 / z_sheet + 1 / plate)
        return z_sheet

    def perforated_panel_layer(self, t=19, d=8, s=16, open_area=None, rho=None, end_correction=None, method="barrier",
                               porous_contact=0, layer=None, *, mechanical_resistance=0, d_exit=None,
                               exit_radiation="air"):
        """
        Add a perforated plate with circular holes as a transfer-matrix layer.

        The available formulations fall into two matrix topologies. ``"barrier"``, ``"barrier_mpp"``,
        ``"bessel_ingard"``, and ``"conical"`` model the perforated plate as a local surface impedance using a
        pressure-jump transfer matrix. Their aperture and optional plate-mass terms are first combined as specific
        impedances; the matrix entry is then ``Zp/S0`` because the package matrices use volume velocity, where ``S0`` is
        ``self.s0``. ``"eq_fluid"`` treats the holes as finite viscothermal ducts and uses a finite-length propagation
        matrix. When ``rho`` is supplied, a plate-mass path is coupled in parallel with the perforation impedance, with
        the mass convention given under ``rho`` below.

        ``method="barrier"`` implements the Helmholtz-resonator perforated-sheet impedance of Cox and D'Antonio [1],
        equation 7.21 (section 7.3.3.3), which combines the acoustic mass of equation 7.6 with the viscous resistance of
        equation 7.12. The sheet impedance is ``Zp = Rm + 1j*omega*rho0*t_eff/phi``, where ``phi`` is the open-area
        ratio, ``t_eff = 2*delta*a + t`` for hole radius ``a`` and end-correction factor ``delta``, and
        ``Rm = sqrt(8*eta*rho0*omega)*(1 + t/(2*a))/phi``. The book writes ``Rm`` with the kinematic viscosity as
        ``(rho0/phi)*sqrt(8*nu*omega)*(1 + t/(2*a))``; the two forms are equal because ``nu = eta/rho0``, and
        ``air_prop`` supplies the dynamic viscosity ``eta``. The remaining ``z_s2`` term of equation 7.21 is the
        impedance below the sheet, which the pressure-jump matrix supplies, and the small viscous mass term of equation
        7.6 is omitted, as it is in equation 7.21 itself.

        If ``rho`` is supplied, a solid-fraction plate-mass approximation ``1j*omega*rho*t*(1 - phi)`` is coupled in
        parallel with the perforation impedance, following the topology of equation 7.16. The book's ``z_mem`` there is
        the full panel mass per unit area; the ``(1 - phi)`` weighting is a package convention.

        ``method="barrier_mpp"`` implements Maa's approximate microperforated-panel sheet impedance for circular
        submillimetre perforations. For the pure aperture branch (``rho=None``), the implementation follows Maa [2],
        equations 5a, 5b, and 6. Maa writes the impedance normalized by ``rho0*c0``; this method converts it to
        dimensional surface impedance as ``Zp = 32*eta*t*kr/(phi*d**2) + 1j*omega*rho0*t*km/phi``, with
        ``k = d*sqrt(omega*rho0/(4*eta))``, ``kr = sqrt(1 + k**2/32) + sqrt(2)*k*d/(32*t)``, and
        ``km = 1 + 1/sqrt(1 + k**2/2) + 0.85*d/t``. ``end_correction="maa1987"`` uses the earlier forms of Maa [7],
        equations 10 and 11 (in numerical form, equations 13 and 14), ``kr = sqrt(1 + k**2/32) + sqrt(2)*k*d/(8*t)`` and
        ``km = 1 + 1/sqrt(9 + k**2/2) + 0.85*d/t``, which Kang and Fuchs [5], equations 3-7, also use. The resistive end
        term ``sqrt(2)*k*d/(32*t)`` adds ``Rs/phi``, with surface resistance ``Rs = 0.5*sqrt(2*eta*rho0*omega)``. The
        earlier term ``sqrt(2)*k*d/(8*t)`` adds ``4*Rs/phi``, which equals the end resistance
        ``sqrt(8*eta*rho0*omega)/phi`` of ``method="barrier"`` ([1], equation 7.12). If ``rho`` is supplied, the
        solid-fraction plate mass ``1j*omega*rho*t*(1 - phi)`` is added in parallel as a package extension; it is not
        part of Maa's aperture impedance.

        ``method="eq_fluid"`` treats each circular hole as a finite viscothermal duct. The circular-pore propagation
        constant and characteristic impedance are returned by ``viscothermal_circular()``; the characteristic impedance
        is then scaled by ``1/phi`` to homogenize the pore over the full panel area. The corrected length ``t_eff`` is
        used in the finite transfer matrix
        ``[[cos(kc*t_eff), 1j*Zc/phi*sin(kc*t_eff)], [1j*phi/Zc*sin(kc*t_eff), cos(kc*t_eff)]]``, with the package area
        scaling applied in the same way as other layer matrices.

        ``method="bessel_ingard"`` follows the general circular-aperture model of Vigran [4], section 5.2.5.1 and annex
        J, which combines the exact viscous short-tube solution with Ingard's [3] end correction. With the short-tube
        impedance of Maa [2], equation 2, for hole radius ``a``, angular frequency ``omega``, air density ``rho0``,
        dynamic viscosity ``eta``, and open-area ratio ``phi``,

        ``s = a*sqrt(-1j*omega*rho0/eta)``

        ``F = 1 - 2*J1(s)/(s*J0(s))``

        ``Zp = 1j*omega*rho0*(t + Delta)/(phi*F)``,

        where ``Delta`` is the end-correction length of both hole ends. Placing ``Delta`` inside the Bessel factor gives
        the end mass the same viscous correction ``1/F`` as the air in the hole, and no separate end resistance is
        added. If ``rho`` is supplied, the full-surface plate mass ``1j*omega*rho*t`` is coupled in parallel with
        ``Zp``, as in the membrane and aperture circuit of Kang and Fuchs [5], equations 1 and 9.

        ``method="conical"`` implements the conical holes of Vigran [8]. The hole diameter grows from ``d`` on the
        incident face to ``d_exit`` on the other face, so the narrow end must face the incident side. The air in the
        hole carries spherical waves from the virtual apex of the cone, which lies ``x1 = a*t/(b - a)`` before the
        entrance and ``x2 = b*t/(b - a)`` before the exit for the radii ``a = d/2`` and ``b = d_exit/2``. The exit mouth
        radiates as a baffled piston,

        ``Zb = 1 - 2*J1(2*k*b)/(2*k*b) + 2j*S1(2*k*b)/(2*k*b)``,

        normalized by ``rho0*c0``, with the Struve function ``S1`` ([8], equation 9). The exit conditions of [8],
        equations 12 and 13, fix the ratio of the backward and forward waves in the hole,

        ``r = -exp(-2j*k*x2)*((1 + 1j*k*x2)*Zb - 1j*k*x2)/((1 - 1j*k*x2)*Zb - 1j*k*x2)``,

        and equations 11 and 16 give the input impedance at the entrance mouth, with ``E1 = exp(-1j*k*x1)``,

        ``Zi = 1j*rho0*c0*k*x1*(E1 + r/E1)/((1 + 1j*k*x1)*E1 + r*(1 - 1j*k*x1)/E1)``.

        The entrance radiation does not enter ``Zi``, and the sheet impedance is ``Zp = Zi/phi`` ([8], equation 27). The
        model has no viscous losses, so the resistance comes from a fabric (``porous_facing_layer`` with
        ``in_contact=True``) or a porous layer behind the plate. [8] assumes a low open area and holes small compared
        with the wavelength, and applies the design to panels thicker than about 9-10 mm. The cone angle is limited to
        about 30 degrees ([4], section 5.2.5.2). The method takes it as the half-angle and warns above it.

        ``exit_radiation`` sets the medium the exit mouths radiate into. ``"air"`` is equation 9 of [8]. With
        ``"porous"``, the ``porous_layer`` behind the plate replaces the air, and
        ``Zb = (Zc/(rho0*c0))*Zpiston(2*kc*b)``, where ``Zpiston`` is the piston expression above and ``Zc`` and ``kc``
        are that layer's characteristic impedance and wavenumber. Kirby and Cummings [9], equation 18, citing Ingard and
        Bolt, obtain the mass end correction next to a porous medium by this substitution; applying it to the full
        piston impedance is a package extension. They also find that, in most cases, it over-predicts the measured end
        correction. ``"mean"`` averages the ``"air"`` and ``"porous"`` values, the empirical choice of [4], sections
        5.2.4.3 and 5.2.5.2. The ``porous_layer`` must follow the plate directly or follow an in-contact facing on the
        exit face, and ``compute()`` raises otherwise.

        If ``open_area`` is not supplied, the method assumes a square pitch and computes ``phi = pi*d**2/(4*s**2)``. If
        ``open_area`` is supplied, ``s`` is replaced by the equivalent square pitch for metadata.

        The ``end_correction`` options set ``Delta``, which ``method="barrier"`` and ``method="eq_fluid"`` use as
        ``t_eff = t + Delta`` and ``method="bessel_ingard"`` places inside the Bessel factor. ``"ingard"`` is Vigran's
        fit [4], annex J, to Ingard's end correction for a circular aperture in a square tube,
        ``Delta = 0.96*sqrt(A0)*(1 - 1.24*sqrt(phi) + 0.27*phi)`` with ``A0 = pi*a**2``. ``"nesterov"`` is the
        circular-pattern correction ``Delta = 0.85*d*(1 - 1.47*sqrt(phi) + 0.47*phi**1.5)`` listed in [1], table 7.1;
        ``"jb"`` / ``"jaouen_becot"`` is the square-pattern Jaouen-Becot correction
        ``Delta = 0.85*d*(1 - 1.13*sqrt(phi) - 0.09*phi + 0.27*phi**1.5)`` from the same table; and ``"beranek"`` is the
        single-hole infinite-baffle approximation ``Delta = 0.85*d``.

        A porous layer in contact with the plate also adds resistance, because the flow converging on each hole passes
        through the porous material. Ingard [6], equation 3, gives this term as ``sigma*delta/phi`` for the one-side end
        correction ``delta = Delta/2`` and flow resistivity ``sigma``. With ``porous_contact`` set to ``K > 0``,
        ``method="barrier"`` and ``method="bessel_ingard"`` add ``K*sigma*delta/phi`` to the aperture impedance before
        the plate mass is coupled, where ``sigma`` is the summed flow resistivity of the ``porous_layer`` entries
        directly before and after the plate. Ingard's result is ``K = 1``; the scaling ``0 <= K <= 1`` for partial
        contact follows [4], section 5.2.4.1. ``compute()`` re-evaluates the term from the neighbours present at that
        time.

        A ``porous_facing_layer`` with ``in_contact=True`` directly before or after the plate adds its sheet impedance
        ``Zf`` to the aperture impedance as ``Zf/phi``, before the plate mass is coupled (see ``porous_facing_layer``).
        ``method="eq_fluid"`` has no plate-mass branch, so there ``Zf/phi`` is a pressure jump at the face of the holes
        on the side of the sheet. With ``method="conical"``, a sheet after the plate acts as ``Zf/phi_exit`` with the
        exit open area ``phi_exit = phi*(d_exit/d)**2``, which is ``q*Zf/phi`` with ``q = (d/d_exit)**2`` of [8],
        equation 27.

        If ``rho`` is supplied, the mechanical loss resistance ``r_m`` (``mechanical_resistance``) is added in series
        with the plate mass. The plate branch becomes ``r_m*rho0*c0 + 1j*omega*rho*t*(1 - phi)``, or
        ``r_m*rho0*c0 + 1j*omega*rho*t`` for ``method="bessel_ingard"`` and ``r_m*rho0*c0 + 1j*omega*rho*t*(1 - phi_m)``
        for ``method="conical"``. ``r_m`` enters as ``r'`` in the membrane branch of Kang and Fuchs [5], equations 1 and
        9, where it mainly reflects mounting losses; the ``(1 - phi)`` weighting is the package convention named above.
        It damps only the plate motion, not the flow through the holes. It matters for light sheets and textiles, and
        near the resonance of the plate mass with the layers behind it. ``method="eq_fluid"`` does not accept it.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the perforated plate [mm].
        d : float or int, optional
            Hole diameter [mm].
        s : float or int, optional
            Hole spacing from the center of one hole to the next [mm].
        open_area : float, optional
            Ratio of open area. If set to ``None`` it is calculated from the hole spacing ``s``. If a value is supplied,
            the equivalent spacing is calculated and stored.
        rho : float, int or None, optional
            Plate density [kg/m3]. If ``None`` is passed, a fully rigid plate is assumed. If supplied,
            ``method="bessel_ingard"`` uses the full-surface mass ``1j*omega*rho*t`` while ``method="barrier"`` and
            ``method="barrier_mpp"`` use the solid-fraction mass ``1j*omega*rho*t*(1 - phi)``. ``method="conical"`` uses
            ``1j*omega*rho*t*(1 - phi_m)`` with the mean open area through the plate
            ``phi_m = phi*(1 + d_exit/d + (d_exit/d)**2)/3``, a package extension.
        end_correction : string or None, optional
            End-correction model. Options are ``"ingard"``, ``"jb"`` / ``"jaouen_becot"``, ``"nesterov"``, and
            ``"beranek"``. If ``None``, ``method="bessel_ingard"`` uses ``"ingard"`` and ``method="barrier"`` and
            ``method="eq_fluid"`` use ``"jb"``. ``method="barrier_mpp"`` accepts only ``"maa1998"`` (default) and
            ``"maa1987"``. ``method="conical"`` takes none.
        method : string, optional
            Chooses the perforated-plate formulation. Available options are ``"barrier"``, ``"barrier_mpp"``,
            ``"eq_fluid"``, ``"bessel_ingard"``, and ``"conical"``.
        porous_contact : float, optional
            Porous-contact factor ``K`` from 0 to 1. ``0`` disables the term. Only ``method="barrier"`` and
            ``method="bessel_ingard"`` accept a nonzero value. ``compute()`` warns when a nonzero value has no
            ``porous_layer`` directly before or after the sheet, since the term is then zero.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be added to the existing ones.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the plate motion, in series with the plate mass, normalized by
            ``rho0*c0`` [-]. Requires ``rho``; not accepted by ``method="eq_fluid"``.
        d_exit : float or None, optional
            Hole diameter on the exit face for ``method="conical"`` [mm]. Must be larger than ``d`` and not larger
            than ``s``.
        exit_radiation : string, optional
            Medium the exit mouths radiate into for ``method="conical"``: ``"air"`` (default), ``"porous"``, or
            ``"mean"``.

        References
        ----------
        [1] T. J. Cox and P. D'Antonio, "Acoustic Absorbers and Diffusers: Theory, Design and Application", 3rd ed.,
            CRC Press, 2017.
        [2] D.-Y. Maa, "Potential of microperforated panel absorber", Journal of the Acoustical Society of America,
            vol. 104, pp. 2861-2866, 1998.
        [3] U. Ingard, "On the theory and design of acoustic resonators", Journal of the Acoustical Society of
            America, vol. 25, pp. 1037-1061, 1953.
        [4] T. E. Vigran, "Manual for NorFLAG, version 4.0", Trondheim, 2018.
        [5] J. Kang and H. V. Fuchs, "Predicting the absorption of open weave textiles and micro-perforated
            membranes backed by an air space", Journal of Sound and Vibration, vol. 220, pp. 905-920, 1999.
        [6] U. Ingard, "Perforated facing and sound absorption", Journal of the Acoustical Society of America, vol. 26,
            pp. 151-154, 1954.
        [7] D.-Y. Maa, "Microperforated-panel wideband absorbers", Noise Control Engineering Journal, vol. 29,
            pp. 77-84, 1987.
        [8] T. E. Vigran, "Conical apertures in panels; sound transmission and enhanced absorption in resonator
            systems", Acta Acustica united with Acustica, vol. 90, pp. 1170-1177, 2004.
        [9] R. Kirby and A. Cummings, "The impedance of perforated plates subjected to grazing gas flow and backed by
            porous media", Journal of Sound and Vibration, vol. 217, pp. 619-636, 1998.
        """
        valid_methods = {"barrier", "barrier_mpp", "eq_fluid", "bessel_ingard", "conical"}
        if method not in valid_methods:
            raise ValueError("perforated_panel_layer method must be 'barrier', 'barrier_mpp', 'eq_fluid', "
                             "'bessel_ingard', or 'conical'.")
        self._validate_exit_radiation(exit_radiation, method)
        if method == "conical":
            if d_exit is None:
                raise ValueError("perforated_panel_layer(method='conical') needs the exit hole diameter d_exit.")
            if not (np.isfinite(t) and np.isfinite(d) and np.isfinite(d_exit) and t > 0 and d > 0):
                raise ValueError("perforated_panel_layer(method='conical') needs finite t > 0, d > 0 and d_exit.")
            if d_exit <= d:
                raise ValueError("perforated_panel_layer(method='conical') needs d_exit > d: the narrow end of the "
                                 "holes faces the incident side.")
            if end_correction is not None:
                raise ValueError("method='conical' takes no end_correction; its only end term is the exit radiation.")
            self._warn_cone_angle(t, d, d_exit, "perforated_panel_layer")
        elif d_exit is not None:
            raise ValueError("d_exit is only used by method='conical'.")
        self._validate_porous_contact(porous_contact)
        if porous_contact and method not in {"barrier", "bessel_ingard"}:
            raise ValueError("porous_contact is only supported by method='barrier' and method='bessel_ingard'.")
        self._validate_mechanical_resistance(mechanical_resistance, rho)
        if mechanical_resistance and method == "eq_fluid":
            raise ValueError("mechanical_resistance needs a plate-mass branch, which method='eq_fluid' does not have.")
        if layer is None:
            layer = len(self.matrix)

        uses_end_correction = method in {"barrier", "eq_fluid", "bessel_ingard"}
        effective_end_correction = None
        if uses_end_correction:
            default_end_correction = "ingard" if method == "bessel_ingard" else "jb"
            effective_end_correction = default_end_correction if end_correction is None else end_correction
        elif method == "barrier_mpp":
            # barrier_mpp carries its end corrections inside Maa's resistance and mass factors.
            effective_end_correction = "maa1998" if end_correction is None else end_correction
            if effective_end_correction not in {"maa1998", "maa1987"}:
                raise ValueError("perforated_panel_layer(method='barrier_mpp') end_correction must be 'maa1998' or "
                                 f"'maa1987', not {end_correction!r}.")

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
        if method == "conical" and d_exit > s:
            raise ValueError("The exit holes of method='conical' overlap: d_exit must not exceed the hole spacing s.")

        end_length = None
        t_corr = None
        if uses_end_correction:
            end_length = self._circular_hole_end_length(d_meters, open_area, effective_end_correction)
            t_corr = t_meters + end_length
        contact_resistance = self._porous_contact_resistance(layer, porous_contact, (end_length or 0.0) / 2)
        facing_impedance = self._contact_facing_impedance(layer)
        contact_facing = bool(self._contact_facings(layer))
        vis = self.air_prop["air_viscosity"]

        Tp = None
        if method == "barrier":
            """
            Lumped perforated-sheet impedance from Cox and D'Antonio, section 7.3.1. The transfer matrix is a
            zero-thickness pressure-jump element, so the perforated screen is represented by its surface impedance
            rather than by propagation through a finite duct.
            """
            # Equation 7.12, the resistive term of equation 7.21. The book writes it with the kinematic
            # viscosity nu, while air_prop["air_viscosity"] is the dynamic viscosity eta. Since nu = eta/rho0,
            # rho0 belongs inside the root multiplying vis.
            rm = (1 / open_area) * np.sqrt(8 * vis * self.rho0 * self.w0) * (
                        1 + t_meters / (d_meters))  # Surface resistance
            zpp = (1j / open_area) * t_corr * self.w0 * self.rho0 + rm  # Impedance of perforated plate
            zpp = zpp + (contact_resistance + facing_impedance) / open_area
            if rho:
                # Specific impedance of the plate motion: mechanical loss resistance and solid-fraction mass
                mip = mechanical_resistance * self.rho0 * self.c0 + 1j * self.w0 * rho * t_meters * (1 - open_area)
                zpp = 1 / (1 / zpp + 1 / mip)
            zpp = zpp / self.s0
            zpp = self._broadcast_to_incidence_angles(zpp)

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Tp = np.array([[ones, zpp],
                           [zeros, ones]])

        elif method == "barrier_mpp":
            """
            Maa's microperforated-plate impedance from "Potential of microperforated panel absorber" (1998).

            The expression is kept in dimensional surface-impedance form: ``rm`` is not normalized by ``rho0*c0`` and
            the acoustic mass term is multiplied by ``rho0`` rather than divided by ``c0``.
            """
            cis = d_meters * np.sqrt(self.w0 * self.rho0 / (4 * vis))
            maa1987 = effective_end_correction == "maa1987"
            # Resistive end term: sqrt(2)/32 (Maa 1998, one surface resistance) or sqrt(2)/8 (Maa 1987, four)
            kr = np.sqrt(1 + cis ** 2 / 32) + np.sqrt(2) / (8 if maa1987 else 32) * cis * d_meters / t_meters
            rm = 32 * vis / open_area * t_meters / d_meters ** 2 * kr
            km = 1 + 1 / np.sqrt((9 if maa1987 else 1) + cis ** 2 / 2) + 0.85 * d_meters / t_meters
            m = self.rho0 * t_meters / open_area * km

            zpp = rm + 1j * self.w0 * m + facing_impedance / open_area
            if rho:
                # Specific impedance of the plate motion: mechanical loss resistance and solid-fraction mass
                mip = mechanical_resistance * self.rho0 * self.c0 + 1j * self.w0 * rho * t_meters * (1 - open_area)
                zpp = 1 / (1 / zpp + 1 / mip)
            zpp = zpp / self.s0
            zpp = self._broadcast_to_incidence_angles(zpp)

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Tp = np.array([[ones, zpp],
                           [zeros, ones]])

        elif method == "bessel_ingard":
            """
            Local circular-aperture Bessel sheet over the end-corrected hole length, with optional full-surface
            plate-mass coupling.
            """
            zpp = self._bessel_ingard_circular_sheet_impedance(t_meters, d_meters, open_area, end_length, rho=rho,
                                                               resistance=contact_resistance + facing_impedance,
                                                               mechanical_resistance=mechanical_resistance)
            zpp = zpp / self.s0
            zpp = self._broadcast_to_incidence_angles(zpp)

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Tp = np.array([[ones, zpp],
                           [zeros, ones]])

        elif method == "eq_fluid":
            """
            Finite circular-duct transfer matrix using viscothermal equivalent-fluid properties.

            Each perforation is treated as a circular pore with propagation constant and characteristic impedance
            from ``viscothermal_circular()``. The pore impedance is homogenized over the full panel area by the
            ``1/open_area`` factor below, and ``end_correction`` is applied as an effective added duct length. These
            two perforated-panel steps are separate from the circular-pore viscothermal solution itself.
            """
            kc, zc = self.viscothermal_circular(d_meters)
            zc = zc / open_area
            kc = self._broadcast_to_incidence_angles(kc)
            zc = self._broadcast_to_incidence_angles(zc)

            Tp = np.array([[np.cos(kc * t_corr), 1j * zc / self.s0 * np.sin(kc * t_corr)],
                           [1j * self.s0 / zc * np.sin(kc * t_corr), np.cos(kc * t_corr)]])
            # A facing in contact sits at the face of the holes it covers: a pressure jump Zf/phi on that side.
            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            for side in (-1, 1):
                z_face = self._contact_facing_impedance(layer, side=side)
                if np.any(z_face != 0):
                    z_face = self._broadcast_to_incidence_angles(z_face / open_area / self.s0)
                    T_face = np.array([[ones, z_face], [zeros, ones]])
                    Tp = (np.einsum("ijfa,jkfa->ikfa", T_face, Tp) if side < 0
                          else np.einsum("ijfa,jkfa->ikfa", Tp, T_face))

        elif method == "conical":
            """
            Conical holes of Vigran (2004): wave solution in the tapered hole, exit radiation, panel equation 27.
            """
            ratio = d_exit / d
            zpp = self._conical_sheet_impedance("hole", layer, t_meters, d_meters / 2, d_exit / 2000, open_area,
                                                open_area * ratio ** 2, open_area * (1 + ratio + ratio ** 2) / 3,
                                                exit_radiation, rho, mechanical_resistance)
            zpp = self._broadcast_to_incidence_angles(zpp / self.s0)

            ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
            zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

            Tp = np.array([[ones, zpp],
                           [zeros, ones]])

        self.matrix[layer] = {"type": "perforated_panel_layer",
                              "thickness [mm]": t,
                              "hole_diameter [mm]": d,
                              "exit_hole_diameter [mm]": d_exit,
                              "hole_spacing [mm]": s,
                              "open_area [%]": open_area * 100,
                              "end_correction": effective_end_correction,
                              "exit_radiation": exit_radiation if method == "conical" else None,
                              "rho [kg/m3]": rho,
                              "method": method,
                              "porous_contact": porous_contact or None,
                              "contact_resistance [Pa*s/m]": contact_resistance if porous_contact else None,
                              "mechanical_resistance [-]": mechanical_resistance or None,
                              "contact_facing": contact_facing or None,
                              "matrix": Tp,
                              }

    def slotted_panel_layer(self, t=19, w=8, s=16, open_area=None, rho=None, method="barrier", porous_contact=0,
                            layer=None, *, mechanical_resistance=0, w_exit=None, exit_radiation="air"):
        """
        Add a regular long-slot panel as a lumped transfer-matrix layer.

        This method models a plate with infinitely long, periodically spaced slits as a local surface impedance with
        transfer matrix ``[[1, Zs], [0, 1]]``, with ``method="barrier"`` for slits of constant width and
        ``method="conical"`` for wedge-shaped slits.

        The open-area ratio is ``phi = w/s`` unless ``open_area`` is supplied directly. For ``method="barrier"``, the
        slot end correction follows the long-slot expression of Kristiansen and Vigran [1],
        ``t_eff = t + 2*w*(-1/pi)*log(sin(pi*phi/2))``, and the slot impedance is assembled as
        ``Zs = (Rp + 1j*omega*rho0*t_eff) / phi``, where ``Rp = 0.5*sqrt(2*eta*rho0*omega)*(4 + 2*t/w)`` represents
        viscous losses in the slot. If ``rho`` is supplied, the solid-fraction plate mass ``1j*omega*rho*t*(1 - phi)``
        is coupled in parallel with the slot impedance, so the mass path scales with the closed area fraction of the
        sheet.

        With ``porous_contact`` set to ``K > 0``, the resistance ``K*sigma*delta`` is added to ``Rp``, where
        ``delta = -(w/pi)*log(sin(pi*phi/2))`` is the one-side end correction and ``sigma`` sums the flow resistivity
        of the ``porous_layer`` entries directly before and after the plate. Ingard [2], equation 3, derived the term
        (``K = 1``) for circular holes in contact with a porous layer; applying it to slots and scaling it by
        ``0 <= K <= 1`` follows [3], section 5.2.4.1. ``compute()`` re-evaluates the term from the neighbours present
        at that time.

        A ``porous_facing_layer`` with ``in_contact=True`` directly before or after the plate adds its sheet impedance
        ``Zf`` to the slot impedance as ``Zf/phi``, before the plate mass is coupled (see ``porous_facing_layer``). If
        ``rho`` is supplied, the mechanical loss resistance ``r_m`` (``mechanical_resistance``) is added in series with
        the plate mass, so the plate branch is ``r_m*rho0*c0 + 1j*omega*rho*t*(1 - phi)``. ``r_m`` enters as ``r'`` in
        the membrane branch of Kang and Fuchs [4], equations 1 and 9, which uses the full surface mass; the
        ``(1 - phi)`` weighting is this layer's solid-fraction mass described above. It damps only the plate motion, not
        the flow through the slots.

        ``method="conical"`` implements the wedge-shaped slots of Vigran [5], the slots of a panel of beams with a
        trapezoidal cross-section ([3], section 5.2.4.3). The slot width grows from ``w`` on the incident face to
        ``w_exit`` on the other face, so the narrow end must face the incident side. The air in the slot carries
        cylindrical waves from the virtual apex of the wedge, which lies ``x1 = a*t/(b - a)`` before the entrance and
        ``x2 = b*t/(b - a)`` before the exit for the half-widths ``a = w/2`` and ``b = w_exit/2``. The exit mouth
        radiates as a baffled strip,

        ``Zb = 2*k*b*(H0(2*k*b) + (pi/2)*(H1(2*k*b)*S0(2*k*b) - H0(2*k*b)*S1(2*k*b))) - H1(2*k*b) + 2j/(pi*2*k*b)``,

        normalized by ``rho0*c0``, with Hankel functions of the second kind ``H0`` and ``H1`` and Struve functions
        ``S0`` and ``S1`` ([5], equation 20). With Hankel functions of the first kind ``G0`` and ``G1``, the exit rows
        of [5], equation 23, fix the ratio of the backward and forward waves,

        ``r = (1j*H0(k*x2) - Zb*H1(k*x2))/(1j*G0(k*x2) - Zb*G1(k*x2))``,

        and equation 25 gives the input impedance at the entrance mouth,

        ``Zi = 1j*rho0*c0*(H0(k*x1) - r*G0(k*x1))/(H1(k*x1) - r*G1(k*x1))``.

        The sheet impedance is ``Zs = Zi/phi``; [5] gives equation 27 for holes and states that the slot form follows
        from it. The model has no viscous losses, so the resistance comes from a fabric or a porous layer behind the
        plate. The wedge angle is limited to about 30 degrees ([3], section 5.2.4.3). The method takes it as the
        half-angle and warns above it.

        ``exit_radiation`` works as in ``perforated_panel_layer``. ``"porous"`` evaluates the strip impedance with the
        characteristic impedance and wavenumber of the ``porous_layer`` behind the plate. This is the substitution of
        Kirby and Cummings [6], equation 18, citing Ingard and Bolt, applied to the full strip impedance as a package
        extension. ``"mean"`` averages it with ``"air"`` ([3], section 5.2.4.3). A ``porous_facing_layer`` with
        ``in_contact=True`` acts as ``Zf/phi`` before the plate and as ``Zf/phi_exit`` after it, with
        ``phi_exit = w_exit/s``. If ``rho`` is supplied, the plate branch is
        ``r_m*rho0*c0 + 1j*omega*rho*t*(1 - phi_m)`` with the mean open area ``phi_m = (w + w_exit)/(2*s)``, a package
        extension.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the slotted plate [mm]
        w : float or int, optional
            Slit width [mm]
        s : float or int, optional
            Slit spacing from the center of one slit to the next [mm]
        open_area : float, optional
            Ratio of open area. If set to ``None`` it is calculated with the slit spacing ``s``. If a value is supplied,
            the equivalent spacing is calculated and stored.
        rho : float, int or None, optional
            Plate density [kg/m3]. If ``None`` is passed, a fully rigid plate is assumed. If supplied, the
            solid-fraction mass ``1j*omega*rho*t*(1 - phi)`` is coupled in parallel with the slot impedance, or
            ``1j*omega*rho*t*(1 - phi_m)`` for ``method="conical"``.
        method : string, optional
            Slotted-panel calculation method: ``"barrier"`` or ``"conical"``.
        porous_contact : float, optional
            Porous-contact factor ``K`` from 0 to 1. ``0`` disables the term. Only ``method="barrier"`` accepts a
            nonzero value. ``compute()`` warns when a nonzero value has no ``porous_layer`` directly before or after the
            sheet, since the term is then zero.
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the plate motion, in series with the plate mass, normalized by
            ``rho0*c0`` [-]. Requires ``rho``.
        w_exit : float or None, optional
            Slit width on the exit face for ``method="conical"`` [mm]. Must be larger than ``w`` and not larger than
            ``s``.
        exit_radiation : string, optional
            Medium the exit mouths radiate into for ``method="conical"``: ``"air"`` (default), ``"porous"``, or
            ``"mean"``.

        References
        ----------
        [1] U. R. Kristiansen and T. E. Vigran, "On the design of resonant absorbers using a slotted plate", Applied
            Acoustics, vol. 43, pp. 39-48, 1994.
        [2] U. Ingard, "Perforated facing and sound absorption", Journal of the Acoustical Society of America, vol. 26,
            pp. 151-154, 1954.
        [3] T. E. Vigran, "Manual for NorFLAG, version 4.0", Trondheim, 2018.
        [4] J. Kang and H. V. Fuchs, "Predicting the absorption of open weave textiles and micro-perforated
            membranes backed by an air space", Journal of Sound and Vibration, vol. 220, pp. 905-920, 1999.
        [5] T. E. Vigran, "Conical apertures in panels; sound transmission and enhanced absorption in resonator
            systems", Acta Acustica united with Acustica, vol. 90, pp. 1170-1177, 2004.
        [6] R. Kirby and A. Cummings, "The impedance of perforated plates subjected to grazing gas flow and backed by
            porous media", Journal of Sound and Vibration, vol. 217, pp. 619-636, 1998.
        """
        if method not in {"barrier", "conical"}:
            raise ValueError("slotted_panel_layer method must be 'barrier' or 'conical'.")
        self._validate_exit_radiation(exit_radiation, method)
        if method == "conical":
            if w_exit is None:
                raise ValueError("slotted_panel_layer(method='conical') needs the exit slot width w_exit.")
            if not (np.isfinite(t) and np.isfinite(w) and np.isfinite(w_exit) and t > 0 and w > 0):
                raise ValueError("slotted_panel_layer(method='conical') needs finite t > 0, w > 0 and w_exit.")
            if w_exit <= w:
                raise ValueError("slotted_panel_layer(method='conical') needs w_exit > w: the narrow end of the slots "
                                 "faces the incident side.")
            if porous_contact:
                raise ValueError("porous_contact is only supported by method='barrier'; method='conical' takes the "
                                 "porous layer behind the plate through exit_radiation.")
            self._warn_cone_angle(t, w, w_exit, "slotted_panel_layer")
        elif w_exit is not None:
            raise ValueError("w_exit is only used by method='conical'.")
        self._validate_porous_contact(porous_contact)
        self._validate_mechanical_resistance(mechanical_resistance, rho)
        if layer is None:
            layer = len(self.matrix)

        if open_area is None:
            open_area = w / s
        else:
            s = w / open_area

        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        w_meters = w / 1000

        if open_area > 1:
            raise ValueError("Slit spacing must be larger than slit width.")

        contact_resistance = 0.0
        if method == "conical":
            # Wedge-shaped slots of Vigran (2004): wave solution in the slot, exit strip radiation, equation 27.
            if w_exit > s:
                raise ValueError("Slit spacing must not be smaller than the exit slot width.")
            ratio = w_exit / w
            zs = self._conical_sheet_impedance("slot", layer, t_meters, w_meters / 2, w_exit / 2000, open_area,
                                               open_area * ratio, open_area * (1 + ratio) / 2, exit_radiation, rho,
                                               mechanical_resistance)
        else:
            end_length = w_meters * (-1 / np.pi) * np.log(np.sin(0.5 * np.pi * open_area))  # One side
            t_corr = t_meters + 2 * end_length
            contact_resistance = self._porous_contact_resistance(layer, porous_contact, end_length)
            facing_impedance = self._contact_facing_impedance(layer)
            vis = self.air_prop["air_viscosity"]
            Rp = 0.5 * np.sqrt(2 * vis * self.rho0 * self.w0) * (4 + (2 * t) / w) + contact_resistance
            Xp = self.rho0 * t_corr
            zs = (Rp + 1j * self.w0 * Xp + facing_impedance) / open_area
            if rho:
                # Specific impedance of the plate motion: mechanical loss resistance and solid-fraction mass
                mip = mechanical_resistance * self.rho0 * self.c0 + 1j * self.w0 * rho * t_meters * (1 - open_area)
                zs = 1 / (1 / zs + 1 / mip)
        zs = zs / self.s0

        zs = self._broadcast_to_incidence_angles(zs)

        ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))

        Ts = np.array([[ones, zs],
                       [zeros, ones]])

        self.matrix[layer] = {"type": "slotted_panel_layer",
                              "thickness [mm]": t,
                              "slot_width [mm]": w,
                              "exit_slot_width [mm]": w_exit,
                              "slot_spacing [mm]": s,
                              "open_area [%]": open_area * 100,
                              "exit_radiation": exit_radiation if method == "conical" else None,
                              "rho [kg/m3]": rho,
                              "method": method,
                              "porous_contact": porous_contact or None,
                              "contact_resistance [Pa*s/m]": contact_resistance if porous_contact else None,
                              "mechanical_resistance [-]": mechanical_resistance or None,
                              "contact_facing": bool(self._contact_facings(layer)) or None,
                              "matrix": Ts,
                              }

    def _rectangular_pore_density(self, w_meters, l_meters):
        """
        Return the Stinson-Champoux complex density of air in a rectangular pore of sides ``w`` and ``l``.

        The sum over ``n`` is evaluated in closed form. With ``b_n = (2n + 1)*pi/w`` and
        ``q**2 = a_k**2 + 1j*omega*rho0/eta``, the identities ``sum_n 1/b_n**2 = w**2/8`` and
        ``sum_n 1/(b_n**2 + q**2) = w*tanh(q*w/2)/(4*q)`` give
        ``sum_n 1/(b_n**2*(b_n**2 + q**2)) = (w**2/8 - w*tanh(q*w/2)/(4*q))/q**2``. The remaining sum over ``k`` falls
        off as ``1/a_k**4``.
        """
        eta = self.air_prop["air_viscosity"]
        omega = np.asarray(self.w0, dtype=float)[:, None]
        a2 = ((np.arange(RECT_SLOT_DENSITY_TERMS) + 0.5) * 2 * np.pi / l_meters) ** 2
        q2 = a2[None, :] + 1j * omega * self.rho0 / eta
        q = np.sqrt(q2)
        decay = np.exp(-q * w_meters)  # tanh(q*w/2) written for Re(q) > 0 so that large arguments cannot overflow
        inner = (w_meters ** 2 / 8 - w_meters * (1 - decay) / (1 + decay) / (4 * q)) / q2
        series = (inner / a2[None, :]).sum(axis=1)
        f_function = 64j * omega[:, 0] * self.rho0 / (eta * l_meters ** 2 * w_meters ** 2) * series
        return self.rho0 / f_function

    @staticmethod
    def _ingard_rectangular_end_correction(w_meters, l_meters, sw_meters, sl_meters):
        """Return Ingard's one-sided end correction [m] of a rectangular aperture in a rectangular tube."""
        m = np.arange(RECT_SLOT_END_CORRECTION_TERMS + 1, dtype=float)[:, None]
        n = np.arange(RECT_SLOT_END_CORRECTION_TERMS + 1, dtype=float)[None, :]
        open_area = (w_meters * l_meters) / (sw_meters * sl_meters)
        g = np.sinc(m * l_meters / sl_meters) ** 2 * np.sinc(n * w_meters / sw_meters) ** 2 * open_area
        weight = np.where((m == 0) ^ (n == 0), 0.5, 1.0)
        with np.errstate(divide="ignore"):
            terms = g * weight / np.sqrt((m / sl_meters) ** 2 + (n / sw_meters) ** 2)
        terms[0, 0] = 0.0
        return float(2 / np.pi * terms.sum())

    def rectangular_slot_layer(self, t=10, w=5, l=20, sw=10, sl=30, rho=None, porous_contact=0, layer=None, *,
                               mechanical_resistance=0):
        """
        Add a plate perforated by a regular pattern of finite rectangular slots as a lumped transfer-matrix layer.

        The plate is a local sheet with transfer matrix ``[[1, Zs], [0, 1]]``, following the rectangular-aperture model
        of Vigran [1]. Air in a slot of width ``w`` and length ``l`` has the complex density ``rho_r = rho0/F``, where
        ``F`` is the Stinson and Champoux [2] series given in [1], equations 1, 3 and 4,

        ``F = 64j*omega*rho0/(eta*l**2*w**2) * sum_k sum_n 1/(a_k**2*b_n**2*(a_k**2 + b_n**2 + 1j*omega*rho0/eta))``

        with ``a_k = (k + 1/2)*2*pi/l`` and ``b_n = (n + 1/2)*2*pi/w``. The sum over ``n`` is evaluated in closed form,
        so only the sum over ``k`` is truncated. The one-sided end correction ``delta`` is Ingard's [3] (appendix I)
        result for a rectangular aperture in a rectangular tube whose sides are the centre-to-centre spacings ``sl`` and
        ``sw``, given in [1], equations 18 and 19,

        ``delta = (2/pi) * sum_m sum_n g_mn*nu_mn/sqrt((m/sl)**2 + (n/sw)**2)``

        over all ``(m, n)`` except ``(0, 0)``, with ``g_mn = sinc(m*l/sl)**2 * sinc(n*w/sw)**2 * phi`` and
        ``nu_mn = 1/2`` when exactly one index is zero. With open-area ratio ``phi = w*l/(sw*sl)``, the slot impedance
        follows the slotted-plate form of Kristiansen and Vigran [4], equations 7 and 9, with ``rho_r`` in place of
        ``rho0`` for the air inside the slot,

        ``Z_ap = (R + 1j*omega*rho_r*t + 1j*omega*rho0*2*delta)/phi``,

        where ``R`` is a series impedance at the slots, in the place of the fabric resistance of [4], equation 7. ``R``
        collects the porous-contact term and the impedance of a facing in contact with the plate, described below.
        Without them it is zero, and it is complex when the facing carries mass. The term ``1j*omega*rho_r*t`` treats
        the slot air as a lumped mass, which holds while the plate is much thinner than the wavelength. If ``rho`` is
        supplied, the plate branch ``r_m*rho0*c0 + 1j*omega*rho*t`` is coupled in parallel with ``Z_ap``, as in the
        membrane and aperture circuit of Kang and Fuchs [5], equations 1 and 9. ``r_m`` is the mechanical loss
        resistance (``mechanical_resistance``), normalized by ``rho0*c0``.

        With ``porous_contact`` set to ``K > 0``, the resistance ``K*sigma*delta`` is added to ``R``, where ``sigma``
        sums the flow resistivity of the ``porous_layer`` entries directly before and after the plate. Ingard [6],
        equation 3, derived the term (``K = 1``) for circular holes in contact with a porous layer; applying it to
        rectangular slots and scaling it by ``0 <= K <= 1`` follows [7], section 5.2.4.1. ``compute()`` re-evaluates
        the term from the neighbours present at that time.

        A ``porous_facing_layer`` with ``in_contact=True`` directly before or after the plate adds its sheet impedance
        ``Zf`` to ``R`` (see ``porous_facing_layer``). ``r_m`` damps only the plate motion, not the flow through the
        slots.

        Parameters
        ----------
        t : float, optional
            Plate thickness [mm].
        w : float, optional
            Slot width, the short side of the aperture [mm].
        l : float, optional
            Slot length, the long side of the aperture [mm].
        sw : float, optional
            Centre-to-centre spacing between slots across their width [mm].
        sl : float, optional
            Centre-to-centre spacing between slots along their length [mm].
        rho : float or None, optional
            Plate density [kg/m³]. If ``None``, the plate is rigid.
        porous_contact : float, optional
            Porous-contact factor ``K`` from 0 to 1. ``0`` disables the term. ``compute()`` warns when a nonzero
            value has no ``porous_layer`` directly before or after the sheet, since the term is then zero.
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the plate motion, in series with the plate mass, normalized by
            ``rho0*c0`` [-]. Requires ``rho``.

        References
        ----------
        [1] T. E. Vigran, "The acoustic properties of panels with rectangular apertures", Journal of the Acoustical
            Society of America, vol. 135, pp. 2777-2784, 2014.
        [2] M. R. Stinson and Y. Champoux, "Propagation of sound and the assignment of shape factors in model porous
            materials having simple pore geometries", Journal of the Acoustical Society of America, vol. 91,
            pp. 685-695, 1992.
        [3] U. Ingard, "On the theory and design of acoustic resonators", Journal of the Acoustical Society of
            America, vol. 25, pp. 1037-1061, 1953.
        [4] U. R. Kristiansen and T. E. Vigran, "On the design of resonant absorbers using a slotted plate", Applied
            Acoustics, vol. 43, pp. 39-48, 1994.
        [5] J. Kang and H. V. Fuchs, "Predicting the absorption of open weave textiles and micro-perforated
            membranes backed by an air space", Journal of Sound and Vibration, vol. 220, pp. 905-920, 1999.
        [6] U. Ingard, "Perforated facing and sound absorption", Journal of the Acoustical Society of America, vol. 26,
            pp. 151-154, 1954.
        [7] T. E. Vigran, "Manual for NorFLAG, version 4.0", Trondheim, 2018.
        """
        if min(t, w, l, sw, sl) <= 0:
            raise ValueError("Rectangular-slot dimensions must be greater than zero.")
        if w > sw or l > sl:
            raise ValueError("Slot width and length must not exceed their centre-to-centre spacings.")
        self._validate_porous_contact(porous_contact)
        self._validate_mechanical_resistance(mechanical_resistance, rho)
        if layer is None:
            layer = len(self.matrix)

        t_meters, w_meters, l_meters, sw_meters, sl_meters = (value / 1000 for value in (t, w, l, sw, sl))
        open_area = (w_meters * l_meters) / (sw_meters * sl_meters)
        end_correction = self._ingard_rectangular_end_correction(w_meters, l_meters, sw_meters, sl_meters)
        rho_r = self._rectangular_pore_density(w_meters, l_meters)
        contact_resistance = self._porous_contact_resistance(layer, porous_contact, end_correction)
        facing_impedance = self._contact_facing_impedance(layer)

        zs = (contact_resistance + facing_impedance + 1j * self.w0 * rho_r * t_meters
              + 1j * self.w0 * self.rho0 * 2 * end_correction) / open_area
        if rho:
            plate_mass = mechanical_resistance * self.rho0 * self.c0 + 1j * self.w0 * rho * t_meters
            zs = 1 / (1 / zs + 1 / plate_mass)
        zs = self._broadcast_to_incidence_angles(zs / self.s0)

        ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        Ts = np.array([[ones, zs],
                       [zeros, ones]])

        self.matrix[layer] = {"type": "rectangular_slot_layer",
                              "thickness [mm]": t,
                              "slot_width [mm]": w,
                              "slot_length [mm]": l,
                              "slot_spacing_width [mm]": sw,
                              "slot_spacing_length [mm]": sl,
                              "open_area [%]": open_area * 100,
                              "end_correction [mm]": end_correction * 1000,
                              "rho [kg/m3]": rho,
                              "porous_contact": porous_contact or None,
                              "contact_resistance [Pa*s/m]": contact_resistance if porous_contact else None,
                              "mechanical_resistance [-]": mechanical_resistance or None,
                              "contact_facing": bool(self._contact_facings(layer)) or None,
                              "matrix": Ts,
                              }

    def viscothermal_circular(self, d, open_area=None):
        """
        Return viscothermal propagation properties for a circular pore.

        This implements the low-reduced-frequency circular-tube solution for a rigid, isothermal-wall duct. The shear
        wave number is ``s = R*sqrt(omega*rho0/eta)``. The dynamic density uses the viscous Bessel correction
        ``J1(s*sqrt(-j))/J0(s*sqrt(-j))`` and the dynamic bulk modulus uses the thermal correction
        ``J1(s*sqrt(-j*Pr))/J0(s*sqrt(-j*Pr))``. The returned values are the pore propagation constant
        ``kc = omega*sqrt(rho_eff/K_eff)`` and pore characteristic impedance ``zc = sqrt(rho_eff*K_eff)``.

        The formulation follows the Zwikker-Kosten/Tijdeman circular tube solution, with notation consistent with later
        equivalent-fluid treatments such as Stinson. It describes propagation inside the pore only. Perforated-panel
        open-area scaling and end corrections are applied by ``perforated_panel_layer(method="eq_fluid")``.

        Parameters
        ----------
        d : int or float
            Hole diameter [m]
        open_area : int or float, optional
            Deprecated compatibility argument. If supplied, ``zc`` is scaled by ``1/open_area`` to reproduce the
            previous panel-scaled return value. New code should omit this argument and apply panel scaling in the layer
            builder.

        Returns
        -------
        Pore propagation constant array and pore characteristic acoustic impedance array.
        """

        vis = self.air_prop["air_viscosity"]
        gam = self.air_prop["specific_heat_ratio"]
        pn = self.air_prop["prandtl_number"]
        p0 = self.air_prop["atmospheric_pressure"]

        beta = d / 2 * np.sqrt(self.w0 * self.rho0 / vis)
        rhoef = self.rho0 / (1 - 2 / (beta * np.sqrt(-1j)) * jv(1, beta * np.sqrt(-1j)) /
                             jv(0, beta * np.sqrt(-1j)))
        kef = p0 * gam / (1 + (gam - 1) * 2 / (beta * np.sqrt(-1j * pn)) * jv(1, beta * np.sqrt(-1j * pn)) /
                          jv(0, beta * np.sqrt(-1j * pn)))
        kc = self.w0 * np.sqrt(rhoef / kef)
        zc = np.sqrt(rhoef * kef)
        if open_area is not None:
            warnings.warn(
                "Passing open_area to viscothermal_circular() is deprecated. The method now returns pure pore "
                "properties; apply perforated-panel open-area scaling in the layer builder.",
                DeprecationWarning,
                stacklevel=2,
            )
            zc = zc / open_area

        return kc, zc

    @staticmethod
    def _load_gras_bands(csv_name):
        """
        Read a bundled GRAS surface-description CSV file.

        Each file holds three rows: band centre frequencies [Hz], random-incidence absorption coefficients and
        scattering coefficients. The rows are returned as float arrays in that order.
        """
        data = pandas.read_csv(database_path() + "_csv" + os.sep + csv_name, header=None).to_numpy(dtype=float)
        return data[0], data[1], data[2]

    @staticmethod
    def _material_model_record(type, params, bands, absorption, scattering):
        """
        Return the ``matrix["material_model"]`` entry for a fitted material model.

        The band data are stored as lists of floats so the record survives HDF5 checkpoints unchanged. ``scattering``
        may be ``None`` for models without scattering data.
        """
        return {"type": type,
                "params": params,
                "measured_bands_hz": [float(f) for f in np.asarray(bands, dtype=float).ravel()],
                "measured_absorption": [float(a) for a in np.asarray(absorption, dtype=float).ravel()],
                "measured_scattering": None if scattering is None
                else [float(s) for s in np.asarray(scattering, dtype=float).ravel()]}

    def material_model(self, type="door", params=None):
        """
        Set an empirical boundary impedance model adapted from the GRAS material helpers distributed with Hargreaves,
        Rendell, and Lam [1].

        This method is separate from the transfer-matrix layer builders. It directly assigns ``self.z`` and, where
        available, ``self.scat`` from empirical room-acoustics boundary models used in the GRAS auralization examples.

        The ``floor``, ``ceiling``, ``concrete``, ``plaster``, ``door``, and ``window`` branches are Python adaptations
        of the Scene 9 MATLAB helpers ``MaterialModel_Scene9*.m``. The ``mdf`` branch follows
        ``MaterialModel_Scene3MDF.m``. The resistive-data branches load GRAS surface-description absorption data,
        convert random-incidence absorption to a purely real normalized admittance with the 55 degree rule,
        ``Y = cos(55 deg)*(1 - sqrt(1 - alpha))/(1 + sqrt(1 - alpha))``, and interpolate the admittance onto
        ``self.freq``. TMM uses natural cubic spline interpolation for these fits rather than the MATLAB helpers'
        endpoint-slope spline, which avoids unstable endpoint artifacts in sparse data. Where the bundled CSV files
        carry scattering data, ``self.scat`` is interpolated the same way; that output is a package addition, since the
        MATLAB helpers return admittance only. ``door`` has no such data, so ``self.scat`` stays ``None``.

        The ``self.matrix["material_model"]`` record keeps, next to ``type`` and ``params``, the band center
        frequencies, absorption coefficients and scattering coefficients that were fitted, as lists of floats under
        ``measured_bands_hz``, ``measured_absorption`` and ``measured_scattering``. ``door`` stores its summed
        absorption and transmission coefficients and ``None`` for scattering. ``rebuild()`` refits from the bundled
        data, so the record documents the fit and is never read back.

        The ``door`` and ``window`` branches use the hybrid construction: a resistive absorption-data fit combined with
        a reactive mass-spring-damper panel admittance through a Linkwitz-Riley-style crossover, which the source
        helpers call the non-linear crossover method of Aretz et al. [2]. TMM uses the current object's air properties
        rather than the helpers' hard-coded ``rho0=1.21 kg/m3`` and ``c0=343 m/s``. The optional ``smooth`` parameter is
        a package extension.

        GRAS database: https://depositonce.tu-berlin.de//handle/11303/7506
        Supplemental data: https://asa.scitation.org/doi/suppl/10.1121/1.5096171

        Parameters
        ----------
        type : str, optional
            One of ``"floor"``, ``"ceiling"``, ``"door"``, ``"concrete"``, ``"plaster"``, ``"mdf"`` or ``"window"``.
        params : dict, optional
            Calculation parameters for ``"door"`` and ``"window"``: ``sample_rate``, ``crossover_frequency``, ``rho_m``,
            ``d``, ``area``, ``f_res`` and ``smooth``. Missing keys take the branch defaults.

        References
        ----------
        [1] J. A. Hargreaves, L. R. Rendell and Y. W. Lam, "A framework for auralization of boundary element method
            simulations including source and receiver directivity", Journal of the Acoustical Society of America,
            vol. 145, pp. 2625-2637, 2019.
        [2] M. Aretz, R. Nöthen, M. Vorländer and D. Schröder, "Combined broadband impulse responses using FEM and
            hybrid ray-based methods", EAA Symposium on Auralization, Espoo, 2009.
        """
        params = None if params is None else dict(params)
        self._scat = None

        def material_spline(x, y):
            return CubicSpline(x, y, bc_type="natural")

        resistive_files = {"floor": "mat_scene09_floor.csv",
                           "ceiling": "mat_scene09_ceiling.csv",
                           "concrete": "mat_scene09_concrete.csv",
                           "plaster": "mat_scene09_plaster.csv",
                           "mdf": "mat_MDF25mmA_plane_00deg.csv"}

        if type in resistive_files:
            # Purely real (resistive) admittance from the GRAS random-incidence absorption data. ``floor``,
            # ``ceiling``, ``concrete`` and ``plaster`` are the Scene 9 materials; ``mdf`` follows
            # ``MaterialModel_Scene3MDF.m`` for the Scene 3 MDF panel.
            fMeas, aMeas, sMeas = self._load_gras_bands(resistive_files[type])

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            self.z = 1 / material_spline(fMeas, YsMeas)(self.freq)
            self.scat = material_spline(fMeas, sMeas)(self.freq)

        elif type == "door":
            """
            Python adaptation of ``MaterialModel_Scene9Door.m`` from the Hargreaves, Rendell, and Lam GRAS Scene 9
            helper code. No direct GRAS door material data were provided, so this branch preserves the source
            helper's constructed boundary model and caveat: it is intended as a practical room-boundary condition
            where insufficient data exist, not as a general door material law.

            It comprises two approaches:

            1) area purely resistive fit to octave-band summed absorption and transmission coefficient data. 
               Both absorption and transmission coefficients were used since the former did not rise at low
               frequencies,
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

            These are combined using the Linkwitz-Riley-style crossover implemented in the original helper and
            described there as the non-linear crossover method of Aretz et al.

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
                params = {}
                
            if "sample_rate" not in params:
                params["sample_rate"] = 44100
            if "crossover_frequency" not in params:
                params["crossover_frequency"] = 250
            if "rho_m" not in params:
                params["rho_m"] = 375
            if "d" not in params:
                params["d"] = 0.043
            if "area" not in params:
                params["area"] = 2.2 * 0.97
            if "f_res" not in params:
                params["f_res"] = 95
            if "smooth" not in params:
                params["smooth"] = False

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
            sMeas = None  # No scattering data were provided for the door

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = material_spline(fMeas, YsMeas)
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

        elif type == "window":
            """
            Python adaptation of ``MaterialModel_Scene9Windows.m`` from the Hargreaves, Rendell, and Lam GRAS
            Scene 9 helper code. It combines two approaches:

            1) area purely resistive fit to the third-octave band absorption coefficient data provided with the 
               GRAS dataset.

            2) area reactive Mass-Spring-Damper model of the assumed fundamental resonance of the window panels. 
               This was included since such effects are well known be reactive, and this affects room modal
               frequencies.
               It was also deemed necessary since the fundamental resonance of the panels appeared to be lower than the 
               bandwidth the measured dataset extended to (absorption rose quite sharply at the lowest frequencies). 
               The Mass value was chosen to be consistent with the assumed material. Stiffness and Damping values were 
               tuned to the desired absorption peak frequency and bandwidth. This did not however produce sufficient 
               absorption to join with the trend in 1, so an additional amount of purely resistive absorption was also 
               added.

            These are combined using the Linkwitz-Riley-style crossover implemented in the original helper and
            described there as the non-linear crossover method of Aretz et al.

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
                params = {}

            if "sample_rate" not in params:
                params["sample_rate"] = 44100
            if "crossover_frequency" not in params:
                params["crossover_frequency"] = 200
            if "rho_m" not in params:
                params["rho_m"] = 2500
            if "d" not in params:
                params["d"] = 0.0067
            if "area" not in params:
                params["area"] = 5.33
            if "f_res" not in params:
                params["f_res"] = 6.66
            if "smooth" not in params:
                params["smooth"] = False

            sample_rate = params["sample_rate"]
            crossover_frequency = params["crossover_frequency"]
            rho_m = params["rho_m"]
            d = params["d"]
            area = params["area"]
            f_res = params["f_res"]
            smooth = params["smooth"]

            # Load the random incidence absorption coefficient data included in the GRAS database:
            fMeas, aMeas, sMeas = self._load_gras_bands("mat_scene09_windows.csv")

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = material_spline(fMeas, YsMeas)
            Sf = material_spline(fMeas, sMeas)
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
        self.matrix = {"material_model": self._material_model_record(type, params, fMeas, aMeas, sMeas)}

    def field_impedance(self, z):
        """
        Compute the field-incidence impedance from a set of angle-dependent impedances.

        This method averages surface admittance rather than absorption. Given angle-dependent impedance ``Zs(theta)``,
        it evaluates ``Y_f = integral((1 / Zs(theta)) sin(theta) dtheta) / integral(sin(theta) dtheta)`` over
        ``self.incidence_angle`` and returns ``Z_f = 1 / Y_f``. This produces a scalar field-incidence impedance that
        can be passed through the usual reflection relation. It is mathematically distinct from statistical diffuse
        absorption, which averages ``alpha(theta)`` with ``sin(theta)*cos(theta)`` weighting. See Aretz [1], p. 86.

        Because this is a complex admittance average, the result depends on the impedance reference plane: a lossless
        top air layer changes it even though the angle-wise absorption is unchanged. See ``compute()``.

        Parameters
        ----------
        z : array
            Multidimensional array with angle dependent set of impedances.

        Returns
        -------
        Field impedance array.

        References
        ----------
        [1] M. Aretz, "Combined Wave and Ray Based Room Acoustic Simulations of Small Rooms", Ph.D. thesis, RWTH
            Aachen University, Logos Verlag, 2012.
        """
        z = np.asarray(z)
        if z.ndim != 2 or z.shape[1] != len(self.incidence_angle):
            self._raise_if_partial_z_angle("field_impedance()")
            raise ValueError(
                f"z must have one column per angle in incidence_angle: got shape {z.shape} for "
                f"{len(self.incidence_angle)} angles."
            )
        A = 1 / z
        Af1 = A * np.sin(np.deg2rad(self.incidence_angle))
        Af2 = np.sin(np.deg2rad(self.incidence_angle))
        Af1 = integrate.simpson(Af1, x=np.deg2rad(self.incidence_angle))
        Af2 = integrate.simpson(Af2, x=np.deg2rad(self.incidence_angle))
        Af = Af1 / Af2
        z_field = 1 / Af
        return z_field

    def _unflanged_pipe_radiation_load(self):
        """
        Return the low-frequency unflanged circular-pipe radiation load.

        ``srad`` is interpreted as the circular radiating area, so ``a = sqrt(srad/pi)`` and ``ka = k0*a``. The specific
        radiation impedance is ``Zrad/Z0 ~= 0.25*(ka)**2 + 1j*0.6133*ka``. This follows from the low-frequency
        unflanged-pipe relation summarized by Silva et al. [1], equations 2-3 and 10-11, with Levine and Schwinger's
        static end correction ``L/a = 0.6133`` under this package's positive-reactance convention.

        The returned load is divided by ``srad`` because the transfer matrices use volume velocity.

        References
        ----------
        [1] F. Silva, Ph. Guillemain, J. Kergomard, B. Mallaroni and A. N. Norris, "Approximation formulae for the
            acoustic radiation impedance of a cylindrical pipe", Journal of Sound and Vibration, vol. 322,
            pp. 255-263, 2009.
        """
        radius = np.sqrt(self.srad / np.pi)
        ka = self.k0 * radius
        zrad = self.z0 * (0.25 * ka ** 2 + 1j * 0.6133 * ka)
        return (zrad / self.srad).reshape((len(zrad), 1))

    def compute(self, rigid_backing=True, backing=None, conj=False, show_layers=True):
        """
        Compute the global transfer matrix for the existing layers.

        The layer matrices are multiplied from incident side to termination side. Termination is selected with
        ``backing``: ``'rigid'`` for a hard wall, ``'air'`` for an oblique plane-wave air load, or ``'radiation'`` for a
        low-frequency unflanged circular-pipe radiation load.

        Parameters
        ----------
        rigid_backing : bool, optional
            Backing selector retained for the original Boolean API. If True, use a rigid backing. If False and
            ``backing`` is omitted, this is exactly equivalent to ``backing='air'``.
        backing : string, optional
            Explicit backing mode: ``'rigid'``, ``'air'``, or ``'radiation'``. ``backing='air'`` is the same
            no-rigid-wall air termination selected by ``rigid_backing=False``.
        conj : bool, optional
            Option to conjugate the imaginary part of the impedance.
        show_layers : bool, optional
            Option to display the layers and their details.

        Notes
        -----
        Finite backing loads use the loaded transfer-matrix relation ``Zs = S0*(A*ZL + B)/(C*ZL + D)``.
        ``backing='air'`` is the default no-rigid-wall termination when ``rigid_backing=False`` and is distinct from the
        analytical ``backing='radiation'`` option.

        If ``incidence='diffuse'`` and ``diffuse_method='field'`` are used with a top air layer, this method emits a
        ``RuntimeWarning`` because field-incidence impedance averaging is reference-plane sensitive. This warning does
        not apply to ``diffuse_method='paris'``, which averages angle-wise absorption coefficients.
        """
        if getattr(self, "_layers_stale", False):
            reasons = "; ".join(getattr(self, "_stale_reasons", []))
            details = f" Reasons: {reasons}." if reasons else ""
            raise RuntimeError(f"compute() cannot use stale layer matrices. Call rebuild() first.{details}")

        previous_backing = self._backing_record()[1]

        backing_was_omitted = backing is None
        if backing is None and previous_backing is not None:
            backing = previous_backing.get("backing")
            if backing is None:
                backing = "rigid" if previous_backing.get("rigid_backing", True) else "air"
            if not conj and previous_backing.get("impedance_conjugate", False):
                conj = True
        elif backing is None:
            backing = "rigid" if rigid_backing else "air"
        if backing not in {"rigid", "radiation", "air"}:
            raise ValueError("backing must be 'rigid', 'radiation', or 'air'.")

        if list(self.matrix.keys())[0] != "termination" and "material_model" not in self.filename:
            # The old backing entry is removed below and only re-added once the computation succeeds, so a
            # failure in between would otherwise leave the treatment without its termination metadata and
            # make a later rebuild() silently fall back to an air backing.
            matrix_before_compute = dict(self.matrix)
            try:
                self._compute_global_matrix(backing, backing_was_omitted, conj)
            except BaseException:
                self.matrix = matrix_before_compute
                raise

            if show_layers:
                self.show_layers()

    def _backing_record(self, matrix=None):
        """Return the key and record of the stored backing in ``matrix`` (default ``self.matrix``), or ``None``s."""
        matrix = self.matrix if matrix is None else matrix
        for key, record in matrix.items():
            if isinstance(record, dict) and record.get("type") == "backing":
                return key, record
        return None, None

    def _compute_global_matrix(self, backing, backing_was_omitted, conj):
        """Multiply the layer matrices, apply the termination, and store the resulting impedances."""
        self.matrix = dict(collections.OrderedDict(sorted(self.matrix.items())))

        backing_key, backing_record = self._backing_record()
        if backing_record is not None:
            self.matrix.pop(backing_key)
            if backing_was_omitted and backing is None:
                backing = "rigid"
        # The backing record of an earlier compute() sits under a layer index, so layers added afterwards follow it.
        # Renumber the layers in order so the product below and the neighbour lookups see consecutive indices.
        self.matrix = {index: record for index, record in enumerate(self.matrix.values())}

        # Sheets with porous contact read their neighbours when built, and a double-porosity layer reads whether it
        # lies on the rigid wall. Layers may be added in any order, so these are rebuilt here against the final stack.
        self._build_backing = backing
        contact_plates = self._contact_facing_plates()
        for key, record in self.matrix.items():
            if (isinstance(record, dict) and record.get("exit_radiation") in {"porous", "mean"}
                    and self._exit_porous_layer(key) is None):
                raise ValueError(f"Layer {key} ({record['type']}) has exit_radiation={record['exit_radiation']!r} but "
                                 "no porous_layer directly after it or after its in-contact facing.")
        for key, record in list(self.matrix.items()):
            # A plate that folded in a facing at an earlier compute() is rebuilt too, in case that facing has gone.
            if isinstance(record, dict) and (record.get("porous_contact") or record.get("model") == "double_porosity"
                                             or key in contact_plates or record.get("contact_facing")
                                             or record.get("exit_radiation") in {"porous", "mean"}):
                self._add_layer_from_record(key, record)
        for key, record in self.matrix.items():
            if isinstance(record, dict) and record.get("porous_contact") and not any(
                    self.matrix.get(neighbour, {}).get("type") == "porous_layer" for neighbour in (key - 1, key + 1)):
                warnings.warn(f"Layer {key} ({record['type']}) has porous_contact={record['porous_contact']} but no "
                              "porous_layer directly before or after it, so its contact resistance is zero.",
                              UserWarning, stacklevel=3)

        first_layer = self.matrix[list(self.matrix.keys())[0]]
        if (
            self.incidence == "diffuse"
            and self.diffuse_method == "field"
            and isinstance(first_layer, dict)
            and first_layer.get("type") == "air_layer"
        ):
            warnings.warn(
                "diffuse_method='field' was computed with a top air layer. Field-incidence impedance averaging "
                "is sensitive to the impedance reference plane, so a lossless top air layer can change the "
                "field-incidence absorption curve even when angle-wise absorption is unchanged. If you want "
                "statistical diffuse absorption, consider diffuse_method='paris'. If the top air layer is only "
                "a reference-plane offset, remove it before using diffuse_method='field'.",
                RuntimeWarning,
                stacklevel=3,
            )

        Tg = self.matrix[0]["matrix"]
        for matrix in range(len(self.matrix) - 1):
            Tg = np.einsum("ijna,jkna->ikna", Tg, self.matrix[matrix + 1]["matrix"])

        Ag = Tg[0, 0]
        Bg = Tg[0, 1]
        Cg = Tg[1, 0]
        Dg = Tg[1, 1]

        if backing == "rigid":
            # Rigid termination: rear particle velocity is zero, so Zs = S0*A/C.
            self.z_angle = self.s0 * Ag / Cg
        elif backing == "radiation":
            # Low-ka unflanged circular-pipe radiation load.
            zload = self._unflanged_pipe_radiation_load()
            self.z_angle = self.s0 * (Ag * zload + Bg) / (Cg * zload + Dg)
        else:
            # Air termination used by rigid_backing=False: oblique plane-wave load.
            theta = np.deg2rad(np.asarray(self.incidence_angle, dtype=float))[None, :]
            zload = self.z0 / np.cos(theta)
            zload = zload / self.srad
            self.z_angle = self.s0 * (Ag * zload + Bg) / (Cg * zload + Dg)

        if self.incidence == "diffuse":
            zc = self.field_impedance(self._z_angle)
            alpha = self.diffuse_absorption_coefficient(self._z_angle) if self.diffuse_method == "paris" else None
        else:
            zc = self._z_angle[:, 0]
            if self.incidence == "angle":
                _, alpha = self.reflection_and_absorption_coefficient(zc, angle=self.incidence_angle[0])
            else:
                alpha = None

        if not conj:
            self.z = zc
        else:
            self.z = np.conj(zc)
            self.z_angle = np.conj(self._z_angle)
        self._alpha = None if alpha is None else np.asarray(alpha, dtype=float).reshape((len(self.freq),))

        self.matrix[len(self.matrix)] = {"type": "backing",
                                         "rigid_backing": backing == "rigid",
                                         "backing": backing,
                                         "diffuse_method": self.diffuse_method,
                                         "impedance_conjugate": conj,
                                         }
        self._clear_stale_state()

    def clear_matrix(self):
        """Remove the cached layer matrices from ``self.matrix`` to reduce file size."""
        for matrix in self.matrix.keys():
            if "matrix" in list(self.matrix[matrix].keys()):
                self.matrix[matrix]["matrix"] = None

    def reduce_size(self):
        """Drop the cached layer matrices and ``z_angle`` to reduce file size."""
        self.clear_matrix()
        self._z_angle = None
        self._z_angle_angles = None

    def reduced_copy(self, keep_angles=(0.0,)):
        """
        Return a lightweight copy of this treatment for storage or transport.

        The copy carries the layer metadata needed by ``rebuild()`` but not the cached layer transfer matrices, and
        retains only the selected columns of ``z_angle``. Scalar results are untouched, so ``z``, ``alpha``, and
        ``z_norm`` stay exact for both diffuse methods; ``alpha_angle()`` stays exact for every retained angle. This
        object is not modified.

        Parameters
        ----------
        keep_angles : sequence of float or None, optional
            Incidence angles in degrees whose ``z_angle`` columns are retained, in the same units as the
            ``incidence_angle`` constructor argument. Each value must be one of ``self.incidence_angle``; retained
            columns are returned in ascending angle order. The default ``(0.0,)`` keeps normal incidence, which is the
            first angle of both default diffuse ranges. ``None`` discards ``z_angle`` entirely, matching
            ``reduce_size()``.

        Returns
        -------
        TMM
            A new treatment holding the reduced data.

        Raises
        ------
        ValueError
            If ``keep_angles`` is empty, or names an angle this treatment did not compute. Diffuse ranges that start
            above zero have no normal-incidence column, so they need an explicit ``keep_angles``.

        Notes
        -----
        For ``incidence='normal'`` and ``incidence='angle'`` there is a single angle, so that column always spans
        ``incidence_angle``, ``keep_angles`` is not consulted, and the copy is a fully functional treatment; the size
        reduction there comes from dropping the layer matrices alone.

        For ``incidence='diffuse'`` with a proper subset of angles the copy cannot reproduce quantities that integrate
        over the hemisphere, so ``field_impedance()``, ``diffuse_absorption_coefficient()``, and
        ``save2sheet(export_all=True)`` raise a descriptive error rather than returning a wrong result. Call
        ``rebuild()`` to restore the full angular data; it recomputes ``z_angle`` for every angle and clears the reduced
        marker.

        ``compute()`` cannot run directly on the copy because the layer matrices were dropped. Use ``rebuild()``, which
        reconstructs them from the layer metadata first.
        """
        self._raise_if_stale_results("reduced_copy()")

        source_z_angle = self._z_angle
        source_angles = self._stored_z_angle_angles()

        if keep_angles is None or source_z_angle is None:
            new_z_angle = None
            new_angles = None
        elif source_z_angle.shape[1] == 1:
            # A single stored column already spans the angular range, so there is nothing to reduce.
            new_z_angle = source_z_angle.copy()
            new_angles = source_angles.copy() if self._has_partial_z_angle() else None
        else:
            requested = np.asarray(keep_angles, dtype=float).reshape(-1)
            if requested.size == 0:
                raise ValueError(
                    "keep_angles must name at least one incidence angle in degrees, or be None to drop z_angle."
                )
            indices = self._match_stored_angles(requested, source_angles)
            new_z_angle = source_z_angle[:, indices].copy()
            retained = source_angles[indices]
            spans_all = indices.size == source_angles.size
            new_angles = None if spans_all and not self._has_partial_z_angle() else retained

        # Copy without duplicating the heavy arrays: blank them out, deepcopy the remaining metadata, then
        # restore this object and attach the reduced data to the copy.
        stashed_matrices = {}
        for key, layer in self._matrix.items():
            if "matrix" in layer:
                stashed_matrices[key] = layer["matrix"]
                layer["matrix"] = None
        self._z_angle = None
        try:
            reduced = copy.deepcopy(self)
        finally:
            self._z_angle = source_z_angle
            for key, matrix in stashed_matrices.items():
                self._matrix[key]["matrix"] = matrix

        reduced._z_angle = new_z_angle
        reduced._z_angle_angles = new_angles
        return reduced

    def _has_oblique_angle(self):
        """Return True when any computed incidence angle is not zero."""
        return bool(np.any(np.asarray(self.incidence_angle, dtype=float) != 0))

    def _add_layer_from_record(self, key, value):
        """Re-create one stored layer at ``key`` from its ``self.matrix`` metadata."""
        # The record holds checked values, so a cone-angle warning was already given when the layer was made. The
        # thin-plate warning depends on the frequency range, so it is given again only when the range now goes above
        # the plate's limit and the range checked when the layer was made did not.
        limit = value.get("thin_plate_limit [Hz]") if value.get("type") == "thin_plate_layer" else None
        checked = value.get("checked_fmax [Hz]")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*method='conical'\): the aperture half-angle")
            warnings.filterwarnings("ignore", message=r"thin_plate_layer: the bending wavelength")
            self._add_layer_from_record_inner(key, value)
        if (limit is not None and self._has_oblique_angle() and np.max(self.freq) > limit
                and (checked is None or checked <= limit)):
            warnings.warn(f"thin_plate_layer: the bending wavelength is shorter than 6 plate thicknesses above "
                          f"{limit:.0f} Hz, where thin-plate theory loses accuracy.", UserWarning, stacklevel=3)

    def _add_layer_from_record_inner(self, key, value):
        """Re-create one stored layer at ``key``; see ``_add_layer_from_record``."""
        if value["type"] == "porous_layer":
            # The record holds the resolved values, so any preset override was already reported when it was made.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"porous_layer\(preset=")
                self.porous_layer(sigma=value["flow_resistivity [k*Pa*s/m²]"],
                                  t=value["thickness [mm]"],
                                  model=value["model"],
                                  fibre_type=value.get("fibre_type", 1),
                                  porosity=value.get("porosity", 0.95),
                                  tortuosity=value.get("tortuosity", 1.0),
                                  viscous_length=value.get("viscous_length [µm]"),
                                  thermal_length=value.get("thermal_length [µm]"),
                                  thermal_permeability=value.get("thermal_permeability [m²]"),
                                  static_tortuosity=value.get("static_tortuosity"),
                                  static_thermal_tortuosity=value.get("static_thermal_tortuosity"),
                                  frame=value.get("frame", "rigid"),
                                  bulk_density=value.get("bulk_density [kg/m³]"),
                                  pore_shape_factor=value.get("pore_shape_factor"),
                                  perforation_diameter=value.get("perforation_diameter [mm]"),
                                  perforation_spacing=value.get("perforation_spacing [mm]"),
                                  preset=value.get("preset"),
                                  layer=key)
        elif value["type"] == "air_layer":
            self.air_layer(t=value["thickness [mm]"],
                           layer=key)
        elif value["type"] == "perforated_panel_layer":
            self.perforated_panel_layer(t=value["thickness [mm]"],
                                        d=value["hole_diameter [mm]"],
                                        s=value["hole_spacing [mm]"],
                                        end_correction=value.get("end_correction"),
                                        rho=value["rho [kg/m3]"],
                                        method=value["method"],
                                        porous_contact=value.get("porous_contact") or 0,
                                        layer=key,
                                        mechanical_resistance=value.get("mechanical_resistance [-]") or 0,
                                        d_exit=value.get("exit_hole_diameter [mm]"),
                                        exit_radiation=value.get("exit_radiation") or "air")
        elif value["type"] == "slotted_panel_layer":
            self.slotted_panel_layer(t=value["thickness [mm]"],
                                     w=value["slot_width [mm]"],
                                     s=value["slot_spacing [mm]"],
                                     rho=value["rho [kg/m3]"],
                                     method=value["method"],
                                     porous_contact=value.get("porous_contact") or 0,
                                     layer=key,
                                     mechanical_resistance=value.get("mechanical_resistance [-]") or 0,
                                     w_exit=value.get("exit_slot_width [mm]"),
                                     exit_radiation=value.get("exit_radiation") or "air")
        elif value["type"] == "rectangular_slot_layer":
            self._check_record_supported(value)
            self.rectangular_slot_layer(t=value["thickness [mm]"],
                                        w=value["slot_width [mm]"],
                                        l=value["slot_length [mm]"],
                                        sw=value["slot_spacing_width [mm]"],
                                        sl=value["slot_spacing_length [mm]"],
                                        rho=value["rho [kg/m3]"],
                                        porous_contact=value.get("porous_contact") or 0,
                                        layer=key,
                                        mechanical_resistance=value.get("mechanical_resistance [-]") or 0)
        elif value["type"] == "membrane_layer":
            self.membrane_layer(t=value["thickness [mm]"],
                                rho=value["density [kg/m³]"],
                                layer=key,
                                mechanical_resistance=value.get("mechanical_resistance [-]") or 0)
        elif value["type"] == "thin_plate_layer":
            self.thin_plate_layer(t=value["thickness [mm]"],
                                  rho=value["density [kg/m³]"],
                                  E=value["young_modulus [GPa]"],
                                  nu=value["poisson_ratio [-]"],
                                  loss_factor=value["loss_factor [-]"],
                                  layer=key,
                                  mechanical_resistance=value.get("mechanical_resistance [-]") or 0)
        elif value["type"] == "porous_facing_layer":
            open_area = value["open_area [%]"]
            self.porous_facing_layer(t=value["thickness [mm]"],
                                     surface_mass=value["surface_mass [kg/m2]"],
                                     resistance=value["resistance [Pa*s/m]"],
                                     porosity=value["porosity"],
                                     tortuosity=value["tortuosity"],
                                     method=value["method"],
                                     free=value.get("free", True),
                                     rho=value["rho [kg/m3]"],
                                     d=value["pore_diameter [mm]"],
                                     s=value["pore_spacing [mm]"],
                                     open_area=None if open_area is None else open_area / 100,
                                     layer=key,
                                     in_contact=bool(value.get("in_contact")))

    @staticmethod
    def _check_record_supported(value):
        """Raise for a stored layer record that the current layer methods cannot rebuild."""
        if value.get("type") == "rectangular_slot_layer" and value.get("resistance [Pa*s/m]"):
            raise ValueError("This rectangular_slot_layer record has a slot resistance, which the layer no longer "
                             "takes; add porous_facing_layer(method='pieren_fixed', resistance=R, in_contact=True) "
                             "next to the plate instead.")

    def rebuild(self):
        """Rebuild the layer matrices from the stored metadata and recompute the results."""
        matrix = self.matrix.copy()
        # Check before the layers are cleared, so a record that cannot be rebuilt leaves the object as it was.
        for value in matrix.values():
            if isinstance(value, dict):
                self._check_record_supported(value)
        preserve_diffuse_method = (
            getattr(self, "_results_stale", False)
            and any("diffuse_method was changed" in reason for reason in getattr(self, "_stale_reasons", []))
        )
        current_diffuse_method = self.diffuse_method
        self._clear_stale_state()
        self.matrix = {}
        for key, value in matrix.items():
            if key == "termination":
                if value["type"] == "constant_z":
                    self.constant_z(np.asarray(self._z).reshape(-1)[0])
            elif key != "material_model":
                self._add_layer_from_record(key, value)
            else:
                self.material_model(value["type"], params=value["params"])

        backing_data = self._backing_record(matrix)[1]
        if backing_data is not None:
            if "diffuse_method" in backing_data:
                self._diffuse_method = (current_diffuse_method if preserve_diffuse_method
                                        else backing_data["diffuse_method"])
            compute_args = dict(rigid_backing=backing_data["rigid_backing"], backing=backing_data.get("backing"),
                                conj=backing_data["impedance_conjugate"])
        else:
            compute_args = dict(rigid_backing=False)
        # A missing porous neighbour was reported when the stack was first computed.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Layer \d+ .* has porous_contact=")
            self.compute(show_layers=False, **compute_args)
        self._clear_stale_state()

    def _describe_frequency_vector(self):
        """Return source text that rebuilds the explicit frequency vector.

        Inlining a few thousand floats is unreadable, so a vector that came from ``linspace`` or ``geomspace`` is
        emitted as the call that produced it. Anything else falls back to the literal values, which is long but exact. A
        reproduction log that cannot reproduce the grid is worse than a long line.
        """
        freq = self._freq
        first, last, points = float(freq[0]), float(freq[-1]), freq.size

        if points > 2:
            if np.allclose(freq, np.linspace(first, last, points)):
                return f"np.linspace({first!r}, {last!r}, {points})"
            if first > 0 and np.allclose(freq, np.geomspace(first, last, points)):
                return f"np.geomspace({first!r}, {last!r}, {points})"

        return f"np.array({freq.tolist()!r})"

    def log_rebuild(self):
        """Return the list of calls needed to recreate this object."""
        matrix = self.matrix.copy()
        optional_constructor_args = ""
        if self.x_scale != "lin" and self._freq is None:
            optional_constructor_args += f", x_scale={self.x_scale!r}"
        if self.display_name is not None:
            optional_constructor_args += f", display_name={self.display_name!r}"
        if self.color is not None:
            optional_constructor_args += f", color={self.color!r}"
        if self.s0 != 1.0:
            optional_constructor_args += f", s0={self.s0!r}"
        if self.srad != self.s0:
            optional_constructor_args += f", srad={self.srad!r}"
        if self._freq is None:
            preamble = []
            grid_args = f"fmin={self.fmin}, fmax={self.fmax}, df={self._log_number(self.df)}, "
        else:
            preamble = ["import numpy as np"]
            grid_args = f"freq={self._describe_frequency_vector()}, "

        logged_calls = [f"# {self.filename.capitalize()}",
                        *preamble,
                        f"{self.filename} = TMM("
                        f"{grid_args}"
                        f"project_folder=fm.folder_path, "
                        f"incidence='{self.incidence}', "
                        f"incidence_angle={self._incidence_angle}, "
                        f"diffuse_method='{self.diffuse_method}', "
                        f"filename={self.filename!r}"
                        f"{optional_constructor_args})"]

        for key, value in matrix.items():
            if key == "termination":
                if value["type"] == "constant_z":
                    logged_calls.append(f"{self.filename}.constant_z({repr(self.z[0].item())})")
            elif key != "material_model":
                if value["type"] == "porous_layer":
                    optional_args = ""
                    if value.get("fibre_type", 1) != 1:
                        optional_args += f", fibre_type={value['fibre_type']}"
                    if value.get("porosity", 0.95) != 0.95:
                        optional_args += f", porosity={value['porosity']}"
                    if value.get("tortuosity", 1.0) != 1.0:
                        optional_args += f", tortuosity={value['tortuosity']}"
                    for arg, record_key in POROUS_LOG_KEYS:
                        if value.get(record_key) is not None:
                            optional_args += f", {arg}={self._log_number(value[record_key])}"
                    if value.get("frame", "rigid") != "rigid":
                        optional_args += f", frame={value['frame']!r}"
                    logged_calls.append(
                        f"{self.filename}.porous_layer("
                        f"model='{value['model']}', "
                        f"t={float(value['thickness [mm]'])!r}, "
                        f"sigma={value['flow_resistivity [k*Pa*s/m²]']}"
                        f"{optional_args})"
                        )
                elif value["type"] == "air_layer":
                    logged_calls.append(f"{self.filename}.air_layer(t={float(value['thickness [mm]'])!r})")
                elif value["type"] == "perforated_panel_layer":
                    contact_arg = f", porous_contact={value['porous_contact']}" if value.get("porous_contact") else ""
                    end_correction_arg = ""
                    if value.get("method") in {"barrier", "eq_fluid", "bessel_ingard"} \
                            and value.get("end_correction") is not None:
                        end_correction_arg = f"end_correction='{value['end_correction']}', "
                    elif value.get("method") == "barrier_mpp" and value.get("end_correction") == "maa1987":
                        end_correction_arg = "end_correction='maa1987', "
                    logged_calls.append(
                        f"{self.filename}.perforated_panel_layer("
                        f"t={float(value['thickness [mm]'])!r}, "
                        f"d={float(value['hole_diameter [mm]'])!r}, "
                        f"s={float(value['hole_spacing [mm]'])!r}, "
                        f"{end_correction_arg}"
                        f"rho={value['rho [kg/m3]']}, "
                        f"method='{value['method']}'"
                        f"{contact_arg}"
                        f"{self._mechanical_resistance_arg(value)}"
                        f"{self._conical_args(value, 'd_exit', 'exit_hole_diameter [mm]')}"
                        f")"
                        )
                elif value["type"] == "slotted_panel_layer":
                    contact_arg = f", porous_contact={value['porous_contact']}" if value.get("porous_contact") else ""
                    logged_calls.append(
                        f"{self.filename}.slotted_panel_layer("
                        f"t={float(value['thickness [mm]'])!r}, "
                        f"w={float(value['slot_width [mm]'])!r}, "
                        f"s={float(value['slot_spacing [mm]'])!r}, "
                        f"rho={value['rho [kg/m3]']}, "
                        f"method='{value['method']}'"
                        f"{contact_arg}"
                        f"{self._mechanical_resistance_arg(value)}"
                        f"{self._conical_args(value, 'w_exit', 'exit_slot_width [mm]')}"
                        f")"
                        )
                elif value["type"] == "rectangular_slot_layer":
                    contact_arg = f", porous_contact={value['porous_contact']}" if value.get("porous_contact") else ""
                    logged_calls.append(
                        f"{self.filename}.rectangular_slot_layer("
                        f"t={self._log_number(value['thickness [mm]'])}, "
                        f"w={self._log_number(value['slot_width [mm]'])}, "
                        f"l={self._log_number(value['slot_length [mm]'])}, "
                        f"sw={self._log_number(value['slot_spacing_width [mm]'])}, "
                        f"sl={self._log_number(value['slot_spacing_length [mm]'])}, "
                        f"rho={self._log_number(value['rho [kg/m3]'])}"
                        f"{contact_arg}"
                        f"{self._mechanical_resistance_arg(value)}"
                        f")"
                        )
                elif value["type"] == "membrane_layer":
                    logged_calls.append(
                        f"{self.filename}.membrane_layer("
                        f"t={float(value['thickness [mm]'])!r}, "
                        f"rho={value['density [kg/m³]']}"
                        f"{self._mechanical_resistance_arg(value)}"
                        f")"
                        )
                elif value["type"] == "thin_plate_layer":
                    logged_calls.append(
                        f"{self.filename}.thin_plate_layer("
                        f"t={self._log_number(value['thickness [mm]'])}, "
                        f"rho={self._log_number(value['density [kg/m³]'])}, "
                        f"E={self._log_number(value['young_modulus [GPa]'])}, "
                        f"nu={self._log_number(value['poisson_ratio [-]'])}, "
                        f"loss_factor={self._log_number(value['loss_factor [-]'])}"
                        f"{self._mechanical_resistance_arg(value)}"
                        f")"
                        )
                elif value["type"] == "porous_facing_layer":
                    free_arg = f", free={value['free']}" if "free" in value else ""
                    logged_calls.append(
                        f"{self.filename}.porous_facing_layer("
                        f"t={float(value['thickness [mm]'])!r}, "
                        f"surface_mass={value['surface_mass [kg/m2]']}, "
                        f"resistance={value['resistance [Pa*s/m]']}, "
                        f"porosity={value['porosity']}, "
                        f"tortuosity={value['tortuosity']}, "
                        f"method='{value['method']}'"
                        f"{free_arg}"
                        f"{', in_contact=True' if value.get('in_contact') else ''}"
                        f")"
                        )
            else:
                params = ", ".join(f"{name!r}: {self._log_number(item, keep_int=True)}"
                                   for name, item in value["params"].items())
                logged_calls.append(f"{self.filename}.material_model('{value['type']}', params={{{params}}})")

        backing_data = self._backing_record(matrix)[1]
        if backing_data is not None:
            backing = backing_data.get("backing")
            if backing is None:
                logged_calls.append(
                    f"{self.filename}.compute("
                    f"rigid_backing={backing_data['rigid_backing']}, "
                    f"show_layers=True)"
                )
            else:
                logged_calls.append(
                    f"{self.filename}.compute("
                    f"backing='{backing}', "
                    f"conj={backing_data['impedance_conjugate']}, "
                    f"show_layers=True)"
                )
        else:
            logged_calls.append(f"{self.filename}.compute(rigid_backing=False, show_layers=True)")

        logged_calls += [f"{self.filename}.plot(figsize=(7, 4), plots=['alpha'], save_fig=True, timestamp=False)",
                         f"{self.filename}.save()"]

        return logged_calls

    @staticmethod
    def _log_number(value, keep_int=False):
        """
        Return ``value`` as Python source for ``log_rebuild()``.

        NumPy scalars and 0-d arrays become plain Python values, real numbers are written as floats (``float('nan')``
        and ``float('inf')`` for non-finite ones), and anything else uses ``repr``. With ``keep_int=True`` integers
        stay integers.
        """
        if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and value.ndim == 0):
            value = value.item()
        if keep_int and isinstance(value, int) and not isinstance(value, bool):
            return repr(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            return repr(value) if np.isfinite(value) else f"float('{value}')"
        return repr(value)

    @staticmethod
    def _conical_args(value, name, key):
        """Return the ``d_exit``/``w_exit`` and ``exit_radiation`` keywords for ``log_rebuild()``, or ``""``."""
        if value.get("method") != "conical":
            return ""
        radiation = value.get("exit_radiation") or "air"
        radiation_arg = f", exit_radiation='{radiation}'" if radiation != "air" else ""
        return f", {name}={TMM._log_number(value[key])}{radiation_arg}"

    @staticmethod
    def _mechanical_resistance_arg(value):
        """Return the ``mechanical_resistance`` keyword for ``log_rebuild()``, or an empty string when unset."""
        resistance = value.get("mechanical_resistance [-]")
        return f", mechanical_resistance={TMM._log_number(resistance)}" if resistance else ""

    def print_rebuild(self):
        """Print the calls needed to recreate this object."""
        logged_calls = self.log_rebuild()
        for call in logged_calls:
            print(call)

    @staticmethod
    def _is_report_number(value):
        """Return True for real scalar values that can be formatted numerically."""
        return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))

    @staticmethod
    def _report_number(value):
        """
        Return ``value`` as report text: two decimals at or above 1, more below 1.

        Values below 1 get four significant digits within two to six decimals, without trailing zeros past the second,
        so a loss factor of 0.015 prints as ``0.015`` and 0.3 as ``0.30``. Nonzero values below 1e-6 use three
        significant digits in exponent form, so they do not print as zero. Non-finite values print as they are.
        """
        value = float(value)
        if not np.isfinite(value):
            return str(value)
        if value == 0 or abs(value) >= 1:
            return f"{value:0.2f}"
        if abs(value) < 1e-6:
            return f"{value:.3g}"
        decimals = min(6, max(2, 3 - int(np.floor(np.log10(abs(value))))))
        text = f"{value:0.{decimals}f}".rstrip("0")
        return text if len(text.split(".")[1]) >= 2 else f"{value:0.2f}"

    def _layer_report_items(self, layer_data, layer_key=None):
        """
        Return ordered, method-aware layer fields for text and spreadsheet reports.

        The stored ``self.matrix`` dictionaries intentionally keep all reconstruction metadata. This helper filters only
        the presentation layer so reports show parameters that are relevant to the selected formulation. ``layer_key``
        identifies the ``material_model`` record, which reports ``type`` and ``params`` only.
        """
        layer_type = layer_data.get("type")
        method = str(layer_data.get("method", "")).lower()
        model = str(layer_data.get("model", "")).lower()

        if layer_key == "material_model":
            keys = ["type", "params"]
        elif layer_type == "porous_layer":
            keys = ["type", "model", "thickness [mm]", "flow_resistivity [k*Pa*s/m²]"]
            if model in {"mechel_1976", "wilson_db"}:
                keys.append("porosity")
            elif model == "wilson_2015":
                keys.extend(["porosity", "tortuosity"])
            elif model == "mechel_grundmann":
                keys.append("fibre_type")
            elif model in JCA_MODELS:
                keys.extend(["porosity", "tortuosity", "viscous_length [µm]", "thermal_length [µm]"])
                if model != "jca":
                    keys.append("thermal_permeability [m²]")
                if model == "jcapl":
                    keys.extend(["static_tortuosity", "static_thermal_tortuosity"])
            elif model == "attenborough":
                keys.extend(["porosity", "tortuosity", "pore_shape_factor"])
            elif model == "double_porosity":
                keys.extend(["porosity", "tortuosity", "viscous_length [µm]", "thermal_length [µm]",
                             "thermal_permeability [m²]", "perforation_diameter [mm]", "perforation_spacing [mm]",
                             "rigid_backed"])
            if layer_data.get("frame", "rigid") == "limp":
                if "porosity" not in keys:
                    keys.append("porosity")
                keys.extend(["frame", "bulk_density [kg/m³]"])
            if layer_data.get("preset") is not None:
                keys.insert(2, "preset")
                if layer_data.get("preset_overrides"):
                    keys.insert(3, "preset_overrides")
        elif layer_type == "air_layer":
            keys = ["type", "thickness [mm]"]
        elif layer_type == "membrane_layer":
            keys = ["type", "thickness [mm]", "density [kg/m³]", "mechanical_resistance [-]"]
        elif layer_type == "thin_plate_layer":
            keys = ["type", "thickness [mm]", "density [kg/m³]", "young_modulus [GPa]", "poisson_ratio [-]",
                    "loss_factor [-]", "critical_frequency [Hz]", "thin_plate_limit [Hz]", "mechanical_resistance [-]"]
        elif layer_type == "porous_facing_layer":
            keys = ["type", "method", "thickness [mm]", "surface_mass [kg/m2]", "resistance [Pa*s/m]"]
            if method == "rebillard":
                keys.extend(["porosity", "tortuosity"])
            elif method == "pieren":
                keys.append("free")
            keys.extend(["rho [kg/m3]", "pore_diameter [mm]", "pore_spacing [mm]", "open_area [%]", "in_contact"])
        elif layer_type == "perforated_panel_layer":
            keys = [
                "type",
                "method",
                "thickness [mm]",
                "hole_diameter [mm]",
                "exit_hole_diameter [mm]",
                "hole_spacing [mm]",
                "open_area [%]",
            ]
            keys.extend(["end_correction", "exit_radiation"])
            keys.extend(["rho [kg/m3]", "mechanical_resistance [-]", "porous_contact",
                         "contact_resistance [Pa*s/m]", "contact_facing"])
        elif layer_type == "slotted_panel_layer":
            keys = ["type", "method", "thickness [mm]", "slot_width [mm]", "exit_slot_width [mm]", "slot_spacing [mm]",
                    "open_area [%]", "exit_radiation", "rho [kg/m3]", "mechanical_resistance [-]", "porous_contact",
                    "contact_resistance [Pa*s/m]", "contact_facing"]
        elif layer_type == "rectangular_slot_layer":
            keys = ["type", "thickness [mm]", "slot_width [mm]", "slot_length [mm]", "slot_spacing_width [mm]",
                    "slot_spacing_length [mm]", "open_area [%]", "end_correction [mm]", "rho [kg/m3]",
                    "mechanical_resistance [-]", "porous_contact", "contact_resistance [Pa*s/m]",
                    "contact_facing"]
        elif layer_type == "backing":
            keys = ["type", "backing", "rigid_backing", "diffuse_method", "impedance_conjugate"]
        else:
            keys = [key for key in layer_data.keys() if key != "matrix"]

        return [(key, layer_data[key]) for key in keys if key in layer_data and layer_data[key] is not None]

    def _report_layer_keys(self):
        """Return matrix keys in the order used for layer reports."""
        numeric_keys = sorted(key for key in self.matrix.keys() if isinstance(key, int))
        other_keys = [key for key in self.matrix.keys() if not isinstance(key, int)]
        return numeric_keys + other_keys

    @staticmethod
    def _band_status(f_low, f_high, fmin, fmax):
        """Describe where the band ``[fmin, fmax]`` leaves the valid band ``[f_low, f_high]`` (``None`` = open)."""
        f_low = fmin if f_low is None else f_low
        f_high = fmax if f_high is None else f_high
        if f_high < fmin or f_low > fmax or f_low > f_high:
            return "outside", "outside over the whole band"
        below, above = f_low > fmin, f_high < fmax
        if below and above:
            return "partly outside", f"outside below {f_low:0.0f} Hz and above {f_high:0.0f} Hz"
        if below:
            return "partly outside", f"outside below {f_low:0.0f} Hz"
        if above:
            return "partly outside", f"outside above {f_high:0.0f} Hz"
        return "inside", "inside"

    def porous_validity(self, layer_data):
        """
        Check a porous layer record against the validity range of its model, as tabulated in ``POROUS_VALIDITY``.

        Up to three checks are made. ``range`` maps the source's bound on ``f/sigma`` or ``X = rho0*f/sigma`` to a
        frequency band for the layer's ``sigma`` and compares it with the computed band. ``sigma`` and ``frequency``
        compare the layer with the span of the data the model was fitted to. Each check has a ``basis``: ``"stated"``
        when the source states the range as a limit, ``"data"`` when it is only the span of the fitted data.

        Parameters
        ----------
        layer_data : dict
            A ``porous_layer`` record of ``self.matrix``.

        Returns
        -------
        dict
            ``model``, ``material``, ``source`` and ``checks``, a list of dictionaries with ``check``, ``basis``,
            ``limits``, ``status`` (``"inside"``, ``"partly outside"`` or ``"outside"``) and ``detail``. ``checks`` is
            empty when the source states no range.
        """
        model = layer_data.get("model")
        entry = POROUS_VALIDITY.get(model)
        if entry is None:
            return {"model": model, "material": None, "source": None, "checks": []}
        sigma_k = layer_data["flow_resistivity [k*Pa*s/m²]"]
        sigma = sigma_k * 1000
        fmin, fmax = float(np.min(self.freq)), float(np.max(self.freq))
        checks = []
        if entry["range"] is not None:
            scale = 1 / sigma if entry["variable"] == "f/sigma" else self.rho0 / sigma
            low, high = entry["range"]
            status, detail = self._band_status(None if low is None else low / scale,
                                               None if high is None else high / scale, fmin, fmax)
            limits = entry["variable"]
            if low is not None:
                limits = f"{low:.3g} <= {limits}"
            if high is not None:
                limits = f"{limits} <= {high:.3g}"
            checks.append({"check": "range", "basis": entry["basis"], "limits": limits, "status": status,
                           "detail": detail})
        if entry["sigma"] is not None:
            low, high = entry["sigma"]
            inside = low <= sigma_k <= high
            checks.append({"check": "sigma", "basis": "data", "limits": f"{low:g}-{high:g} k*Pa*s/m²",
                           "status": "inside" if inside else "outside",
                           "detail": "inside" if inside else f"sigma = {sigma_k:g} is outside"})
        if entry["frequency"] is not None:
            low, high = entry["frequency"]
            status, detail = self._band_status(low, high, fmin, fmax)
            checks.append({"check": "frequency", "basis": "data", "limits": f"{low:g}-{high:g} Hz", "status": status,
                           "detail": detail})
        return {"model": model, "material": entry["material"], "source": entry["source"], "checks": checks}

    def show_layers(self, conversion=None, validity=False):
        """
        Print each layer with its details.

        Parameters
        ----------
        conversion : list or float and string, optional
            List containing conversion ratio and string containing the name of the converted unit.
        validity : bool, optional
            Also print, for each porous layer, whether its flow resistivity and the computed frequency band fall
            within the validity range of its model. See ``porous_validity()``.
        """
        if conversion is None:
            conversion = [0.0393701, "[inches]"]
        print("Device properties:")
        print(f"\tFilename: {self.filename}")
        if self.display_name is not None:
            print(f"\tDisplay name: {self.display_name}")
        if self.color is not None:
            print(f"\tColor: {self.color}")
        print("\t(1 - Front face)")
        print(f"\t({len(self.matrix)} - Rear Face)")
        print(f"\tSound incidence: {self.incidence}")
        if self.incidence == "diffuse":
            print(f"\tDiffuse method: {self.diffuse_method}")
            print(f"\tAngle: {min(self.incidence_angle):0.0f}° - {max(self.incidence_angle):0.0f}°\n")
        else:
            print(f"\tAngle: {(self.incidence_angle[0]):0.0f}°\n")

        total_depth = 0
        for i, layer_key in enumerate(self._report_layer_keys(), start=1):
            print(f"Layer {i}:")
            for key, value in self._layer_report_items(self.matrix[layer_key], layer_key):
                if self._is_report_number(value):
                    if "[mm]" in key:
                        converted = key.replace("[mm]", conversion[1])
                        print(f"\t{key}: {self._report_number(value)} | {converted}: {value * conversion[0]:0.2f}")
                    else:
                        print(f"\t{key}: {self._report_number(value)}")
                    if "thickness" in key:
                        total_depth += value
                else:
                    print(f"\t{key}: {value}")
            if validity and self.matrix[layer_key].get("type") == "porous_layer":
                self._print_porous_validity(self.matrix[layer_key])
        print(f"\nTotal treatment depth [mm]: {total_depth:0.2f} | " +
              f"Total treatment depth {conversion[1]}: {total_depth * conversion[0]:0.2f}")

    def _print_porous_validity(self, layer_data):
        """Print the ``porous_validity()`` checks of one porous layer under its ``show_layers()`` entry."""
        result = self.porous_validity(layer_data)
        if not result["checks"]:
            print("\tvalidity: no range stated by the source")
            return
        print(f"\tvalidity ({result['material']}):")
        for check in result["checks"]:
            basis = "stated" if check["basis"] == "stated" else "fitted data"
            print(f"\t\t{check['check']}, {basis}, {check['limits']}: {check['detail']}")
        print(f"\t\tsource: {result['source']}")

    def _normal_incidence_alpha(self):
        """Return the normal-incidence absorption coefficient, whatever the treatment's incidence mode is."""
        if self.incidence == "normal":
            _, alpha = self.reflection_and_absorption_coefficient(self.z)
            return np.asarray(alpha, dtype=float).reshape(-1)

        # 'self.alpha' is Paris-averaged under diffuse incidence and 'self.z' is the field impedance, so neither
        # can be used here. The 0 deg column of 'z_angle' is the only normal-incidence quantity available.
        angles = self._stored_z_angle_angles()
        try:
            angle_idx = int(self._match_stored_angles([0.0], angles)[0])
        except ValueError as error:
            raise ValueError(
                f"The causality bound is derived at normal incidence, and this treatment has no 0 deg column to "
                f"read it from: {error} Recompute with incidence='normal', or with an angular range that starts "
                f"at 0 deg."
            ) from None

        return np.asarray(self.alpha_angle(angle_idx), dtype=float).reshape(-1)

    def _rigid_backing_entry(self):
        """Return the stored backing record, raising when it is missing or is not a rigid wall."""
        entry = next(
            (value for value in self.matrix.values() if isinstance(value, dict) and value.get("type") == "backing"),
            None,
        )
        if entry is None:
            raise RuntimeError("The causality bound needs a computed treatment. Call compute() first.")

        backing = entry.get("backing", "rigid" if entry.get("rigid_backing", True) else "air")
        if backing != "rigid":
            raise ValueError(
                f"The causality bound is derived for a rigid-backed layer with no transmission, but this treatment "
                f"was computed with backing={backing!r}. Recompute with backing='rigid'."
            )

        return entry

    def _static_bulk_modulus_ratio(self, bulk_modulus_ratio, thermal):
        """Return the static ``Beff / B0`` to use, its provenance, and the per-layer breakdown."""
        overridden = bulk_modulus_ratio is not None
        try:
            ratio, layers = causality.effective_bulk_modulus_ratio(
                self.matrix,
                thermal=thermal,
                gamma=self.air_prop["specific_heat_ratio"],
            )
        except ValueError:
            # An override is exactly how a stack that cannot be derived from layer metadata is meant to be
            # handled, so the per-layer breakdown is best effort in that case rather than fatal.
            if not overridden:
                raise
            ratio, layers = None, []

        if bulk_modulus_ratio is not None:
            if bulk_modulus_ratio <= 0:
                raise ValueError("bulk_modulus_ratio must be greater than zero.")
            return float(bulk_modulus_ratio), "user", layers

        return ratio, "auto", layers

    def causality_check(self, bulk_modulus_ratio=None, thermal="isothermal", tail="model"):
        """
        Compare the treatment depth against the minimum thickness allowed by the causality principle.

        Evaluates the Yang-Chen-Fu-Sheng bound [1] ``d_min <= d``, where
        ``d_min = (c0 / (4 * pi**2)) * (Beff / B0) * integral of -ln[1 - A(f)] / f**2 df`` over the normal-incidence,
        rigid-backed absorption spectrum. A treatment with ``ratio = d_min / depth`` close to unity is optimal in the
        sense of the source paper: it uses no more thickness than causality demands for the spectrum it delivers.

        The integral runs from zero to infinite frequency while the treatment is only computed over a finite band, so
        the out-of-band tails are estimated and reported separately. Read ``tail_fraction`` before trusting the number:
        a result whose tails carry a large share warns, whichever side of the bound it lands on and under either
        ``tail`` mode. The Notes give the recipe for a wider band.

        Parameters
        ----------
        bulk_modulus_ratio : float, optional
            Static ``Beff / B0`` overriding the value derived from the layer metadata. Required for stacks containing a
            membrane layer, a constant impedance or a material model, whose static compressibility is not described by
            Wood's law. A porosity is not accepted in its place because it cannot express a ratio below unity, which the
            isothermal limit and a membrane's compliance both need.
        thermal : string, optional
            Static compression assumed for the derived ratio. ``'isothermal'`` applies the specific heat ratio to porous
            layers, which is their true zero-frequency limit. ``'adiabatic'`` reproduces the assumption made in the
            source paper.
        tail : string, optional
            ``'model'`` adds the out-of-band tails. ``'none'`` returns the in-band integral only, which underestimates
            ``d_min``.

        Returns
        -------
        Dictionary with ``d_min``, ``d_min_in_band`` and ``depth`` in millimetres, the dimensionless ``ratio`` between
        the first and the last, the static bulk modulus ratio and its per-layer breakdown, the three integral pieces,
        and any warnings raised.

        Notes
        -----
        A ``ratio`` above unity is reported rather than raised. No passive rigid-backed structure can exceed the bound,
        so it means the static bulk modulus is too large, the stack holds compliance that Wood's law discards, or a
        layer model is not passive. The ``diagnosis`` entry names which, and the empirical Delany-Bazley family,
        non-causal at low ``f / sigma``, is the usual cause.

        A large ``tail_fraction`` means the band was too narrow to pin ``d_min`` down. The fix is to recompute over a
        wider band, on a copy so the treatment keeps its own::

            import copy
            import numpy as np

            wide = copy.deepcopy(treatment)
            wide.freq = np.geomspace(7, 20000, 1500)
            wide.rebuild()
            wide.compute(backing="rigid", show_layers=False)
            result = wide.causality_check()

        Widening collapses the extrapolated share while barely moving ``ratio``. Push ``fmin`` down only as far as the
        layer models stay passive: the empirical porous models return slightly negative absorption below a floor set by
        the model and the flow resistivity. Confirm with ``result['diagnosis']['passivity']`` rather than with
        ``check_passivity()``, which also screens the oblique columns and can fail on a widened diffuse copy even when
        the causality result is sound. Keep the upper edge at or below 20 kHz, and read nothing but the causality result
        off the widened copy.

        A ``ratio`` well below unity is design feedback rather than a fault: the same depth could carry more absorption,
        more bandwidth, or a lower band edge. A perforated facing over an undamped cavity is the usual example, and
        porous material behind the same facing lifts the ratio close to unity at unchanged depth.

        References
        ----------
        [1] M. Yang, S. Chen, C. Fu and P. Sheng, "Optimal sound-absorbing structures", Materials Horizons,
            vol. 4, pp. 673-680, 2017.
        """
        self._raise_if_stale_results("causality_check()")
        self._rigid_backing_entry()

        ratio_value, source, layers = self._static_bulk_modulus_ratio(bulk_modulus_ratio, thermal)
        alpha = self._normal_incidence_alpha()

        result = causality.causal_minimum_thickness(self.freq, alpha, self.c0, ratio_value, tail=tail)

        depth = self.depth
        diagnosis, message = causality.diagnose(
            result,
            depth,
            ratio_value,
            self.freq,
            alpha,
            available_ratios=self._available_bulk_modulus_ratios(),
        )
        if message is not None:
            result["warnings"].append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)

        result.update({"depth": depth,
                       "ratio": diagnosis["ratio"],
                       "bulk_modulus_ratio": ratio_value,
                       "bulk_modulus_ratio_source": source,
                       "thermal": thermal,
                       "layers": layers,
                       "diagnosis": diagnosis,
                       })

        return result

    def plot_causality(self, show_fig=True, result=None, n_oct=1, **kwargs):
        """
        Plot how the causality minimum thickness accumulates across this treatment's spectrum.

        Three stacked panels: the normal-incidence absorption the causality constraint was evaluated on, the minimum
        thickness accumulated up to each frequency against the treatment depth, and the same curve as fractional-octave
        bands. The middle panel draws the constraint itself, since the accumulated curve must finish below the depth
        line.

        Most of ``d_min`` is usually accumulated where the absorption curve looks worst: the ``1 / f**2`` weight of the
        sum rule makes a mediocre low-frequency roll-off contribute far more minimum thickness than a flat
        high-frequency plateau.

        Parameters
        ----------
        show_fig : bool, optional
            If True, display the Matplotlib figure after generating it.
        result : dict, optional
            A ``causality_check()`` result. Passing a result keeps the figure consistent with numbers already inspected.
            If omitted, one is computed with default options.
        n_oct : int, optional
            Fractional octave resolution of the band panel, following the ``filter_alpha()`` convention.
        **kwargs
            Keyword arguments passed to ``tmm._plot.causality_data``.

        See Also
        --------
        tmm._plot.causality_data
        TMM.causality_check
        """
        self._raise_if_stale_results("plot_causality()")
        if "filename" not in kwargs:
            kwargs["filename"] = self.filename
        else:
            kwargs["filename"] = self._validate_filename(kwargs["filename"]) or self.filename
        if "project_folder" not in kwargs:
            kwargs["project_folder"] = self.project_folder
        else:
            kwargs["project_folder"] = self._validate_project_folder(kwargs["project_folder"]) or os.getcwd()
        _, _, _ = plot.causality_data(self, result=result, n_oct=n_oct, **kwargs)
        if show_fig:
            plt.show()

    def plot_finite_sample_absorption(self, sample_size=3.162, variants=None, n_oct=None, show_fig=True, **kwargs):
        """
        Plot the diffuse-field absorption of a finite sample computed by ``finite_sample_absorption()``.

        The figure compares variants of the same angle-dependent surface impedance: normal incidence, the
        infinite-sample Paris average, and the finite-sample coefficient with its ``normalization`` and
        ``angle_weights`` options. The default sample
        is a 3.162 m square, about 10 m2. The treatment must be computed with ``incidence="diffuse"``. The angular range
        of ``incidence_angle`` sets the integration range and is printed in the title.

        Parameters
        ----------
        sample_size : float, tuple of float, or list, optional
            Sample side length or sides ``(a, b)`` [m]. A list of sizes draws one curve per size for a single
            finite-sample variant.
        variants : None, str, dict, or sequence, optional
            Curves to draw: ``"normal"``, ``"paris"``, ``"thomasson"``, ``"true"``, ``"available"``,
            ``"rectangular_floor"``, ``"irregular_floor"``, ``"irregular_wall"``, ``"all"``, or dicts of
            ``finite_sample_absorption()`` options with an optional ``label``. ``None`` draws ``"paris"`` and
            ``"thomasson"``.
        n_oct : int or None, optional
            Fractional-octave resolution of an added band panel, following the ``filter_alpha()`` convention.
        show_fig : bool, optional
            If True, display the Matplotlib figure after generating it.
        **kwargs
            Keyword arguments passed to ``tmm._plot.finite_sample_data``.

        See Also
        --------
        tmm._plot.finite_sample_data
        TMM.finite_sample_absorption
        """
        self._raise_if_stale_results("plot_finite_sample_absorption()")
        if "filename" not in kwargs:
            kwargs["filename"] = self.filename
        else:
            kwargs["filename"] = self._validate_filename(kwargs["filename"]) or self.filename
        if "project_folder" not in kwargs:
            kwargs["project_folder"] = self.project_folder
        else:
            kwargs["project_folder"] = self._validate_project_folder(kwargs["project_folder"]) or os.getcwd()
        _, _, _ = plot.finite_sample_data(self, sample_size=sample_size, variants=variants, n_oct=n_oct, **kwargs)
        if show_fig:
            plt.show()

    def _available_bulk_modulus_ratios(self):
        """Return the static ``Beff / B0`` this stack supports under each thermal assumption."""
        available = {}
        for mode in causality.VALID_THERMAL:
            try:
                available[mode], _ = causality.effective_bulk_modulus_ratio(
                    self.matrix,
                    thermal=mode,
                    gamma=self.air_prop["specific_heat_ratio"],
                )
            except ValueError:
                continue

        return available

    def check_passivity(self, tolerance=0.0, angles=True):
        """
        Report whether the computed absorption spectrum is physically realizable.

        A passive treatment absorbs between none and all of the incident energy, so ``0 <= alpha <= 1``. A spectrum
        leaving that range describes a treatment that creates energy or reflects more than it receives, which means a
        layer model has been evaluated outside its valid range. The empirical Delany-Bazley family is the usual cause,
        returning negative absorption at low ``f / sigma`` where its curve fits no longer hold.

        Run it on its own as well, not only through ``causality_check()``: the causality bound assumes passivity, so a
        non-passive spectrum falls outside its premises entirely.

        Parameters
        ----------
        tolerance : float, optional
            Slack allowed outside ``[0, 1]`` before a sample counts as non-passive, for absorbing float noise.
        angles : bool, optional
            Also screen every stored angle-dependent absorption column, not just the treatment's own ``alpha``.

        Returns
        -------
        Dictionary with ``passive``, the observed ``alpha_range``, the contiguous ``bands [Hz]`` that offend, the number
        of offending samples, and the spectra that were screened.
        """
        self._raise_if_stale_results("check_passivity()")

        spectra = {"alpha": np.asarray(self.alpha, dtype=float)}
        if angles and getattr(self, "_z_angle", None) is not None:
            for index, angle in enumerate(self._stored_z_angle_angles()):
                spectra[f"{angle:g} deg"] = np.asarray(self.alpha_angle(index), dtype=float)

        reports = {name: causality.passivity_report(self.freq, values, tolerance=tolerance)
                   for name, values in spectra.items()}
        offending = {name: report for name, report in reports.items() if not report["passive"]}

        return {"passive": not offending,
                "alpha_range": (min(report["alpha_range"][0] for report in reports.values()),
                                max(report["alpha_range"][1] for report in reports.values())),
                "bands [Hz]": sorted({band for report in offending.values() for band in report["bands [Hz]"]}),
                "n_offending": sum(report["n_offending"] for report in reports.values()),
                "tolerance": float(tolerance),
                "spectra": reports,
                }

    def causality_limit(self, solve_for="frequency", alpha_target=0.9, f_low=None, f_high=None, depth=None,
                        bulk_modulus_ratio=None, thermal="isothermal"):
        """
        Solve the causality bound for an idealised flat absorption target.

        The bound ties three quantities together: sample thickness, frequency bandwidth and the absorption magnitude
        within the band. Specifying two of them fixes the third. For a target holding ``alpha_target`` over
        ``[f_low, f_high]``, with ``L0 = abs(ln(1 - alpha_target))``, the closed forms are
        ``d = (c0 * L0 / (4 * pi**2)) * (Beff / B0) * (1 / f_low - 1 / f_high)`` and its two rearrangements.

        Parameters
        ----------
        solve_for : string, optional
            ``'frequency'`` returns the lowest attainable band edge, ``'depth'`` the required thickness, and
            ``'absorption'`` the highest flat absorption coefficient the band can hold.
        alpha_target : float, optional
            Flat absorption coefficient held across the band. Ignored when solving for it.
        f_low : float, optional
            Lower band edge in Hz. Ignored when solving for it.
        f_high : float or None, optional
            Upper band edge in Hz. ``None`` means the target extends to infinite frequency.
        depth : float, optional
            Treatment thickness in millimetres. Defaults to this treatment's depth. Ignored when solving for it.
        bulk_modulus_ratio : float, optional
            Static ``Beff / B0`` overriding the value derived from the layer metadata. See ``causality_check()`` for why
            a porosity is not accepted in its place.
        thermal : string, optional
            Static compression assumed for the derived ratio, ``'isothermal'`` or ``'adiabatic'``.

        Returns
        -------
        Dictionary with the solved quantity and every input used, so a printed result states its own assumptions.

        Notes
        -----
        This describes an idealised step spectrum, not the absorption of this stack. It reads as "no causal, passive,
        rigid-backed structure of this thickness and static bulk modulus can hold a flat ``alpha_target`` below
        ``f_low``", never as "this treatment works down to ``f_low``".

        Omitting ``f_high`` asks for a target that holds ``alpha_target`` at every frequency above ``f_low``, which
        demands more thickness than any finite band. Real absorbers, and resonators in particular, trade bandwidth for a
        lower band edge, and the bound fixes the trade exactly: for a target whose upper edge is ``r`` times its lower
        one, the attainable ``f_low`` scales by ``1 - 1 / r``. One octave halves it against the infinite-bandwidth
        answer and a third octave takes it to a fifth. Quoting the ``f_high=None`` figure for a narrowband design
        therefore understates what is reachable by a factor of two or more, so pass the bandwidth actually wanted.

        Read the other way, a narrowband target needs little thickness and a broadband one does not. Holding
        ``alpha_target`` across a narrow notch in the low bass takes a fraction of the thickness that holding it from
        the notch's lower edge upward would. Causality does not forbid deep bass in a shallow cavity, it forbids deep
        bass together with bandwidth.
        """
        ratio_value, _, _ = self._static_bulk_modulus_ratio(bulk_modulus_ratio, thermal)
        if depth is None and solve_for != "depth":
            depth = self.depth

        return causality.causality_limit(
            solve_for,
            self.c0,
            ratio_value,
            alpha_target=None if solve_for == "absorption" else alpha_target,
            f_low=None if solve_for == "frequency" else f_low,
            f_high=f_high,
            depth=None if solve_for == "depth" else depth,
        )

    def filter_alpha(self, n_oct=1, view=True, show_table=False, **kwargs):
        """
        Average the absorption coefficient into fractional-octave bands. See ``tmm._plot.oct_filter`` for kwargs.

        Parameters
        ----------
        n_oct : int, optional
            Fractional-octave resolution: ``1`` for octave bands, ``3`` for third-octave bands.
        view : bool, optional
            If True, plot the narrowband and band-averaged absorption together.
        show_table : bool, optional
            If True, print the band values as a table.

        Returns
        -------
        Bands' center frequency array and filtered absorption array.
        """
        self._raise_if_stale_results("filter_alpha()")
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

            try:
                from IPython.display import display
                display(df)
            except ImportError:
                print(df.to_string())

        return bands, result

    def save2sheet(self, timestamp=False, conversion=None, ext=".xlsx", chart_styles=None, n_oct=3, metadata=True,
                   export_all=False):
        """
        Save current treatment results to XLSX or CSV.

        XLSX exports contain ``Data``, ``Bands`` and ``Setup`` sheets. The ``Setup`` sheet stores the TMM setup and the
        method-aware layer report. CSV exports keep numeric data and metadata separate: selected-method data are written
        to ``.csv`` and, when ``metadata=True``, the setup and layer report are written to ``*_metadata.csv``.

        The selected-method export follows the currently computed incidence settings. When ``incidence='diffuse'`` and
        ``diffuse_method='paris'``, impedance columns are omitted because Paris averaging returns diffuse absorption but
        does not define a unique diffuse complex impedance. ``export_all=True`` is a CSV-only diagnostic export that
        includes angle-wise impedance and absorption, field-incidence diffuse impedance/absorption, and Paris diffuse
        absorption.

        Parameters
        ----------
        timestamp : bool, optional
            Boolean to add timestamping to the filename.
        conversion : list or float and string, optional
            List containing conversion ratio and string containing the name of the converted unit.
        ext : string, optional
            Desired file extension. Available extensions are ``'.xlsx'`` and ``'.csv'``.
        chart_styles : list of ints, optional
            List containing indexes of styles to use in the plots inside the exported spreadsheet.
        n_oct : int, optional
            Fractional octave bands that the absorption will be filtered to.
        metadata : bool, optional
            If True, write a sidecar metadata CSV. XLSX workbooks always include metadata in the ``Setup`` sheet.
        export_all : bool, optional
            If True, write a CSV-only diagnostic export containing angle-wise and diffuse quantities instead of only the
            selected-method data.
        """
        self._raise_if_stale_results("save2sheet()")
        from tmm import _sheet_export

        return _sheet_export.save2sheet(
            self,
            timestamp=timestamp,
            conversion=conversion,
            ext=ext,
            chart_styles=chart_styles,
            n_oct=n_oct,
            metadata=metadata,
            export_all=export_all,
        )

    def save(self):
        """
        Save this TMM object as an HDF5 checkpoint.

        The file is written to ``{project_folder}/Treatments/{filename}.h5``. The ``Treatments`` folder is created when
        needed. The checkpoint holds the reduced form produced by ``reduced_copy(keep_angles=None)``: cached layer
        transfer matrices and ``z_angle`` are left out, while the metadata ``rebuild()`` needs is retained. ``load()``
        calls ``rebuild()``, which recomputes both.

        The HDF5 file is intended as a package-internal checkpoint for reopening and rebuilding a TMM object with a
        compatible version of this package. It is not a stable interchange format for external software.

        Returns
        -------
        None
            The saved path is printed.
        """
        self._raise_if_stale_results("save()")
        folder_check = os.path.exists(self.project_folder + os.sep + "Treatments")
        if folder_check is False:
            os.mkdir(self.project_folder + os.sep + "Treatments")

        h5utils.save_class_to_hdf5(self.reduced_copy(keep_angles=None), filename=self.filename,
                                   folder=self.project_folder + os.sep + "Treatments" + os.sep)
        print("HDF5 file saved at " + self.project_folder + os.sep + "Treatments" + os.sep + self.filename + ".h5")

    def load(self, filename):
        """
        Load a TMM object from an HDF5 checkpoint.

        Parameters
        ----------
        filename : str
            Input filename without the ``.h5`` extension. If ``{project_folder}/Treatments`` exists, the file is loaded
            from that folder. Otherwise, it is loaded directly from ``project_folder``. If no project folder was set,
            the current working directory is used.

        Notes
        -----
        Loading replaces the current object's saved attributes and then calls ``rebuild()`` to reconstruct reduced layer
        matrices and computed results. The receiver object's current ``project_folder`` is kept after loading so copied
        HDF5 files remain attached to the project from which they were loaded. The checkpoint must use layer methods and
        metadata supported by the current package version.
        """
        project_folder = self._project_folder
        folder_check = os.path.exists(self.project_folder + os.sep + "Treatments")
        if self.project_folder:
            if folder_check is False:
                h5utils.load_class_from_hdf5(self, filename, folder=self.project_folder + os.sep)
            else:
                h5utils.load_class_from_hdf5(self, filename,
                                             folder=self.project_folder + os.sep + "Treatments" + os.sep)
        else:
            h5utils.load_class_from_hdf5(self, filename)
        self._project_folder = project_folder
        self.rebuild()
        print(filename + ".h5 loaded successfully.")

    def plot(self, show_fig=True, **kwargs):
        """
        Plot this treatment's acoustic response.

        This is a convenience wrapper around ``tmm._plot.acoustic_data``. Keyword arguments are forwarded to that
        function. If ``filename`` or ``project_folder`` are not supplied, this method uses the treatment's own
        ``filename`` and ``project_folder``.

        The cosmetic ``display_name`` and ``color`` attributes are used by the plotting backend. ``display_name`` is
        used as a human-facing title or legend label when available, while ``color`` sets the primary treatment curve
        color.

        Parameters
        ----------
        show_fig : bool, optional
            If True, display the Matplotlib figure after generating it.
        **kwargs
            Keyword arguments passed to ``tmm._plot.acoustic_data``.

        See Also
        --------
        tmm._plot.acoustic_data
        """
        self._raise_if_stale_results("plot()")
        if "filename" not in kwargs:
            kwargs["filename"] = self.filename
        else:
            kwargs["filename"] = self._validate_filename(kwargs["filename"]) or self.filename
        if "project_folder" not in kwargs:
            kwargs["project_folder"] = self.project_folder
        else:
            kwargs["project_folder"] = self._validate_project_folder(kwargs["project_folder"]) or os.getcwd()
        _, _, _ = plot.acoustic_data([self], **kwargs)
        if show_fig:
            plt.show()
