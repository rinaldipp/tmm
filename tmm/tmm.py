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
from scipy import integrate
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from scipy.signal import butter, freqz, savgol_filter
from scipy.special import jv

from tmm import _causality as causality
from tmm import _h5utils as h5utils
from tmm import _plot as plot
from tmm import _utils as utils
from tmm.database.path import path as database_path

plot.set_style()

# Cap on the rounding precision _grid_decimals derives for a frequency grid. Twelve decimals is far finer
# than any real grid needs, so it only matters for a degenerate grid, which is then left effectively unrounded.
MAX_GRID_DECIMALS = 12

# Truncation of the series in rectangular_slot_layer: the Stinson-Champoux density sum over the slot-length index
# (the width index is summed in closed form), and the Ingard end-correction sum over indices 0 to this value.
RECT_SLOT_DENSITY_TERMS = 1000
RECT_SLOT_END_CORRECTION_TERMS = 180

# Angular intensity weightings w(theta) of Jeong (2013), Table II, tabulated from the phased beam-tracing results of
# Jeong (2010). Rows are the 250 Hz to 4 kHz octave bands, columns 0 to 90 degrees in 15-degree steps. Keys follow the
# figure captions of Jeong (2010): Table II labels its w1 block "floor in a rectangular room", but its values are the
# irregular-room floor of Fig. 6b; w2 ("floor in an irregular room") is the rectangular-room floor of Fig. 7; w3
# ("sidewall in a rectangular room") is the irregular-room wall of Fig. 4b. Table II prints 0.11 for w1 at 4 kHz and
# 45 degrees; Fig. 6b shows about 1.1, and 1.11 restores the unit mean over angle of the other rows.
JEONG_WEIGHT_BANDS = (250.0, 500.0, 1000.0, 2000.0, 4000.0)
JEONG_WEIGHT_ANGLES = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
JEONG_ANGLE_WEIGHTS = {
    "irregular_floor": ((1.63, 1.46, 1.56, 1.46, 0.41, 0.49, 0.0),
                        (1.18, 1.25, 1.14, 1.07, 0.99, 0.82, 0.54),
                        (1.60, 1.58, 1.27, 0.96, 0.84, 0.75, 0.0),
                        (1.76, 1.75, 1.39, 1.00, 0.76, 0.33, 0.0),
                        (1.60, 1.57, 1.40, 1.11, 0.75, 0.40, 0.17)),
    "rectangular_floor": ((1.20, 1.14, 1.04, 1.23, 1.22, 0.80, 0.36),
                          (0.85, 0.69, 1.31, 1.32, 1.25, 0.87, 0.71),
                          (1.38, 1.43, 1.28, 1.23, 0.87, 0.53, 0.26),
                          (1.62, 1.56, 1.43, 1.16, 0.77, 0.42, 0.05),
                          (1.62, 1.57, 1.40, 1.14, 0.84, 0.43, 0.0)),
    "irregular_wall": ((1.25, 1.43, 1.03, 0.99, 1.06, 0.81, 0.43),
                       (1.61, 1.59, 1.43, 1.15, 0.79, 0.42, 0.0),
                       (1.63, 1.57, 1.40, 1.15, 0.82, 0.42, 0.0),
                       (1.64, 1.57, 1.41, 1.15, 0.82, 0.42, 0.0),
                       (1.64, 1.58, 1.41, 1.14, 0.81, 0.41, 0.0)),
}


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

    def equivalent_fluid_model(self, sigma, model="mac", fibre_type=1, porosity=0.95, tortuosity=1.0):
        """
        Return the complex propagation constant ``kc`` and characteristic impedance ``zc`` for an equivalent-fluid
        porous material.

        The coefficient-set models use the common power-law form ``kc = k0*(1 + C1*X**(-C2) - 1j*C3*X**(-C4))`` and
        ``Zc = Z0*(1 + C5*X**(-C6) - 1j*C7*X**(-C8))``, where ``X = rho0*f/sigma`` and ``sigma`` is in SI units
        [Pa*s/m2] inside this method. This family includes ``db``, ``miki``, ``qunli``, ``mac``, ``mechel_gf_lowX``,
        ``mechel_gf_highX``, ``mechel_rf_lowX``, and ``mechel_rf_highX``. The implementation follows the generic
        empirical equivalent-fluid form presented by Cox and D'Antonio [1], section 6.5.1. The coefficient table is
        consistent with the Delany-Bazley-Miki model options listed in the COMSOL Acoustics Module porous-model
        documentation: https://doc.comsol.com/6.4/doc/com.comsol.help.aco/aco_ug_pressure.05.005.html#1226466.

        The remaining branches use separate equations. ``komatsu`` follows Komatsu [2], equations 15-18, using powers of
        ``2 - log10(f/sigma)``. ``mechel_1976`` follows Mechel's [3] low-frequency extension of the Delany-Bazley
        absorber formula and uses ``porosity`` as the open porosity ``h``. ``mechel_grundmann`` keeps the
        Mechel-Grundmann polynomial formulation from the reference implementation accompanying [1]. ``wilson_2015``
        implements Wilson [4], equations 24-25, with unit shape factors, ``porosity`` mapped to ``phi`` and
        ``tortuosity`` mapped as described under Parameters. ``wilson_db`` keeps the compact Wilson relaxation model
        from the MATLAB reference script accompanying [1].

        Use ``mechel_1976`` when open porosity is part of the material definition and ``wilson_2015`` when porosity and
        tortuosity are known. Use ``qunli`` for porous plastics and open foams within the COMSOL-listed range of
        approximately 200-2000 Hz and ``3e3 <= sigma <= 24e3`` Pa*s/m2, and ``mac`` for the Modified Champoux-Allard
        coefficient set within its listed range of approximately 45-11000 Hz.

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
            Open pore volume fraction for branches that expose porosity. Used by ``mechel_1976``, ``wilson_2015``, and
            ``wilson_db``.
        tortuosity : float, optional
            High-frequency tortuosity-like factor for ``model="wilson_2015"``. Wilson's internal ``q`` parameter is
            computed as ``sqrt(tortuosity)``.

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
        """
        coefficients = {  # List of coefficients for each available model
            "db": [0.0978, 0.7, 0.189, 0.595, 0.0571, 0.754, 0.087, 0.732],  # Delaney-Bazley
            "miki": [0.122, 0.618, 0.18, 0.618, 0.079, 0.632, 0.12, 0.632],  # Miki
            "qunli": [0.188, 0.554, 0.163, 0.592, 0.209, 0.548, 0.105, 0.607],  # Qunli
            "mechel_gf_lowX": [0.135, 0.646, 0.396, 0.458, 0.0688, 0.707, 0.196, 0.549],  # Mechel, glass fiber, low X
            "mechel_gf_highX": [0.102, 0.705, 0.179, 0.674, 0.0235, 0.887, 0.0875, 0.77],  # Mechel, glass fiber, high X
            "mechel_rf_lowX": [0.136, 0.641, 0.322, 0.502, 0.081, 0.699, 0.191, 0.556],  # Mechel, rock fiber, low X
            "mechel_rf_highX": [0.103, 0.716, 0.179, 0.663, 0.0563, 0.725, 0.127, 0.655],  # Mechel, rock fiber, high X
            "mac": [0.0982, 0.685, 0.288, 0.526, 0.0729, 0.66228, 0.187, 0.5379],  # Modified Allard and Champoux
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
        else:
            available_models = [key for key in coefficients.keys()]
            available_models.append("komatsu")
            available_models.append("mechel_1976")
            available_models.append("wilson_2015")
            available_models.append("wilson_db")
            available_models.append("mechel_grundmann")
            raise NameError("Unidentified model. Choose between the available models: ", available_models)

        return kc, zc

    def porous_layer(self, sigma=27, t=5, model="mac", fibre_type=1, porosity=0.95, tortuosity=1.0, layer=None):
        """
        Add a porous layer modelled as an equivalent fluid.

        The porous material is treated as an equivalent fluid with complex characteristic impedance ``Zc`` and complex
        wavenumber ``kc`` from the selected empirical model. For oblique incidence, the tangential wavenumber is
        conserved at the air/material interface: ``kt = k0*sin(theta)`` and ``kz = sqrt(kc**2 - kt**2)``. The layer
        transfer matrix is then built with the normal wavenumber ``kz`` and the normal characteristic impedance
        ``Zc,n = Zc*kc/kz``: ``[[cos(kz*t), j*Zc,n*sin(kz*t)], [j*sin(kz*t)/Zc,n, cos(kz*t)]]``. This is the standard
        equivalent-fluid extended-reaction form; for notation, see Allard and Atalla [1].

        Parameters
        ----------
        sigma : float or int, optional
            Flow resistivity of the porous material [k*Pa*s/m²]
        t : float or int, optional
            Thickness of the porous material [mm]
        model : string, optional
            Equivalent-fluid model name. See ``equivalent_fluid_model()`` for the options.
        fibre_type : int, optional
            Fibre type for the Mechel-Grundmann model: ``1`` for basalt or rock wool, ``2`` for glass fibre.
        porosity : float, optional
            Open pore volume fraction for models that expose it. Used by ``model="wilson_2015"``, ``model="wilson_db"``,
            and ``model="mechel_1976"``.
        tortuosity : float, optional
            High-frequency tortuosity-like factor for ``model="wilson_2015"``. Wilson's internal ``q`` is computed as
            ``sqrt(tortuosity)``.
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.

        References
        ----------
        [1] J. F. Allard and N. Atalla, "Propagation of Sound in Porous Media: Modelling Sound Absorbing
            Materials", 2nd ed., Wiley, 2009.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        sigma_k = sigma * 1000  # Convert kPa*s/m2 to Pa*s/m2

        kc, zc = self.equivalent_fluid_model(
            sigma_k,
            model=model,
            fibre_type=fibre_type,
            porosity=porosity,
            tortuosity=tortuosity,
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

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "porous_layer",
                              "flow_resistivity [k*Pa*s/m²]": sigma,
                              "thickness [mm]": t,
                              "model": model,
                              "fibre_type": fibre_type,
                              "porosity": porosity,
                              "tortuosity": tortuosity,
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

    def membrane_layer(self, t=1, rho=8050, layer=None):
        """
        Add a limp membrane or mass-sheet layer.

        The membrane is modeled as a local pressure-jump impedance with surface mass ``m_s = rho*t`` and specific
        impedance ``Z_m = 1j*omega*m_s``, where ``t`` is converted from millimeters to meters. The transfer matrix is
        ``[[1, Z_m/S0], [0, 1]]`` because this class uses pressure and volume velocity as the matrix state; ``S0`` is
        the front surface area. The resulting surface impedance recovered by ``compute()`` is scaled back to specific
        impedance.

        This is the mass term of the classical limp-membrane absorber model. Brandão, *Acustica de Salas: Projeto e
        Modelagem*, section 2.3.2, writes ``Z_m = 1j*omega*m_s`` after neglecting membrane resistance and combines it in
        series with the surface impedance of the backing cavity, which here comes from the layers behind the membrane.

        The model neglects bending stiffness, finite-panel modal behavior, edge/support losses, and intrinsic membrane
        resistance. It is appropriate for a limp local-reaction mass sheet, not a full structural panel model.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the membrane [mm]
        rho : float or int, optional
            Material density [kg/m³]
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.
        """
        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters

        zc = 1j * self.w0 * rho * t_meters / self.s0
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
                              "matrix": Tm,
                              }

    def porous_facing_layer(self, t=1, surface_mass=None, resistance=None, porosity=0.7, tortuosity=1.0,
                            method="rebillard", free=True, rho=None, d=None, s=None, open_area=None,
                            layer=None):
        """
        Add a thin porous-facing or fabric sheet as a local pressure-jump layer.

        The layer is represented by a zero-thickness transfer matrix ``[[1, Zs/S0], [0, 1]]``, where ``Zs`` is the
        specific sheet impedance and ``S0`` is the front surface area used by the volume-velocity matrix state.

        ``method="rebillard"`` implements the unbonded thin porous-facing expression from Rebillard et al. [1]. The
        implementation uses the negligible-flexural-stiffness limit. With open porosity ``h``, tortuosity ``tau``,
        facing thickness ``e``, surface mass ``m``, and flow resistance ``R = sigma*e``, the coupling term is
        ``I = 1j*omega*(tau - 1)*h*rho0*e + R*h**2``.

        ``method="pieren"`` implements the thin woven-fabric sheet impedance described by Pieren [2]. With
        ``free=True``, the sheet resistance and limp surface mass are combined in parallel,
        ``Zs = 1j*omega*m*R / (1j*omega*m + R)``. With ``free=False``, the motionless fabric limit ``Zs = R`` is used.
        The aliases ``method="pieren_free"`` and ``method="pieren_fixed"`` set this option explicitly.

        Measured ``surface_mass`` and ``resistance`` are preferred for real fabrics. If ``surface_mass`` is ``None`` and
        ``rho`` is supplied, it is estimated as ``rho*t``; if ``resistance`` is ``None``, it is estimated from the ideal
        circular-pore expression ``32*eta*t/(open_area*d**2)``. For ``method="rebillard"``, omitting both uses a nominal
        thin facing with ``surface_mass=0.25`` kg/m2 and ``resistance=200`` Pa*s/m.

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

        References
        ----------
        [1] P. Rebillard, J.-F. Allard, C. Depollier, P. Guignouard, W. Lauriks, C. Verhaegen and A. Cops, "The
            effect of a porous facing on the impedance and the absorption coefficient of a layer of porous material",
            Journal of Sound and Vibration, vol. 156, pp. 541-555, 1992.
        [2] R. Pieren, "Sound absorption modeling of thin woven fabrics backed by an air cavity", Textile Research
            Journal, vol. 82, pp. 864-874, 2012.
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
            if free:
                if surface_mass is None:
                    raise ValueError("surface_mass or rho must be supplied for a free Pieren fabric.")
                jw_mass = 1j * self.w0 * surface_mass
                z_sheet = jw_mass * resistance / (jw_mass + resistance)
            else:
                z_sheet = np.full_like(self.freq, resistance, dtype=complex)
        else:
            if surface_mass is None:
                raise ValueError("surface_mass or rho must be supplied for the Rebillard porous-facing model.")
            if not 0 < porosity <= 1:
                raise ValueError("porosity must be greater than zero and less than or equal to one.")
            if tortuosity <= 0:
                raise ValueError("tortuosity must be greater than zero.")

            rho2 = porosity * self.rho0
            rho_a = (tortuosity - 1.0) * rho2
            coupling = 1j * self.w0 * rho_a * t_meters + resistance * porosity**2
            c_term = 1j * self.w0 * rho2 * t_meters + coupling
            b_term = 1j * self.w0 * surface_mass + coupling
            denominator = porosity * (coupling * (1.0 - porosity) + b_term * porosity) + (
                1.0 - porosity
            ) * (coupling * porosity + c_term * (1.0 - porosity))
            z_sheet = (c_term * b_term - coupling**2) / denominator

        z_sheet = z_sheet / self.s0
        z_sheet = self._broadcast_to_incidence_angles(z_sheet)
        ones = np.ones_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
        zeros = np.zeros_like(self.freq, shape=(len(self.freq), len(self.incidence_angle)))
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
                       "matrix": Tf,
                       }
        if method == "pieren":
            matrix_data["free"] = free
        self.matrix[layer] = matrix_data

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
                                                resistance=0.0):
        """
        Return the local sheet impedance for ``perforated_panel_layer(method="bessel_ingard")``.

        The Bessel short-tube impedance is evaluated over the hole length extended by ``end_length``, the series
        ``resistance`` [Pa*s/m] is added per hole, and the optional full-surface plate mass is coupled in parallel.
        """
        radius = d_meters / 2
        viscosity = self.air_prop["air_viscosity"]
        kd = radius * np.sqrt(self.rho0 * self.w0 / viscosity)
        bessel_argument = kd * np.sqrt(-1j)
        aperture_factor = 1 - 2 * jv(1, bessel_argument) / (bessel_argument * jv(0, bessel_argument))
        z_sheet = 1j * self.w0 * self.rho0 * (t_meters + end_length) / (open_area * aperture_factor)
        z_sheet = z_sheet + resistance / open_area

        if rho is not None:
            plate_mass = 1j * self.w0 * rho * t_meters
            z_sheet = 1 / (1 / z_sheet + 1 / plate_mass)

        return z_sheet

    def perforated_panel_layer(self, t=19, d=8, s=16, open_area=None, rho=None, end_correction=None, method="barrier",
                               porous_contact=0, layer=None):
        """
        Add a perforated plate with circular holes as a transfer-matrix layer.

        The available formulations fall into two matrix topologies. ``"barrier"``, ``"barrier_mpp"``, and
        ``"bessel_ingard"`` model the perforated plate as a local surface impedance using a pressure-jump transfer
        matrix. Their aperture and optional plate-mass terms are first combined as specific impedances; the matrix entry
        is then ``Zp/S0`` because the package matrices use volume velocity, where ``S0`` is ``self.s0``. ``"eq_fluid"``
        treats the holes as finite viscothermal ducts and uses a finite-length propagation matrix. When ``rho`` is
        supplied, a plate-mass path is coupled in parallel with the perforation impedance, with the mass convention
        given under ``rho`` below.

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
        ``km = 1 + 1/sqrt(1 + k**2/2) + 0.85*d/t``. If ``rho`` is supplied, the solid-fraction plate mass
        ``1j*omega*rho*t*(1 - phi)`` is added in parallel as a package extension; it is not part of Maa's aperture
        impedance.

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
            ``method="barrier_mpp"`` use the solid-fraction mass ``1j*omega*rho*t*(1 - phi)``.
        end_correction : string or None, optional
            End-correction model. Options are ``"ingard"``, ``"jb"`` / ``"jaouen_becot"``, ``"nesterov"``, and
            ``"beranek"``. If ``None``, ``method="bessel_ingard"`` uses ``"ingard"`` and ``method="barrier"`` and
            ``method="eq_fluid"`` use ``"jb"``. This argument is ignored by ``method="barrier_mpp"``.
        method : string, optional
            Chooses the perforated-plate formulation. Available options are ``"barrier"``, ``"barrier_mpp"``,
            ``"eq_fluid"``, and ``"bessel_ingard"``.
        porous_contact : float, optional
            Porous-contact factor ``K`` from 0 to 1. ``0`` disables the term. Only ``method="barrier"`` and
            ``method="bessel_ingard"`` accept a nonzero value.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be added to the existing ones.

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
        """
        valid_methods = {"barrier", "barrier_mpp", "eq_fluid", "bessel_ingard"}
        if method not in valid_methods:
            raise ValueError(
                "perforated_panel_layer method must be 'barrier', 'barrier_mpp', 'eq_fluid', or 'bessel_ingard'."
            )
        self._validate_porous_contact(porous_contact)
        if porous_contact and method not in {"barrier", "bessel_ingard"}:
            raise ValueError("porous_contact is only supported by method='barrier' and method='bessel_ingard'.")
        if layer is None:
            layer = len(self.matrix)

        uses_end_correction = method in {"barrier", "eq_fluid", "bessel_ingard"}
        effective_end_correction = None
        if uses_end_correction:
            default_end_correction = "ingard" if method == "bessel_ingard" else "jb"
            effective_end_correction = default_end_correction if end_correction is None else end_correction
        elif end_correction is not None:
            warnings.warn(
                f"end_correction is ignored for perforated_panel_layer(method='{method}'); "
                "the selected method defines its end correction internally.",
                stacklevel=2,
            )

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

        end_length = None
        t_corr = None
        if uses_end_correction:
            end_length = self._circular_hole_end_length(d_meters, open_area, effective_end_correction)
            t_corr = t_meters + end_length
        contact_resistance = self._porous_contact_resistance(layer, porous_contact, (end_length or 0.0) / 2)
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
            zpp = zpp + contact_resistance / open_area
            if rho:
                mip = 1j * self.w0 * rho * t_meters * (1 - open_area)  # Specific mass impedance of the plate
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
            kr = np.sqrt(1 + cis ** 2 / 32) + np.sqrt(2) / 32 * cis * d_meters / t_meters
            rm = 32 * vis / open_area * t_meters / d_meters ** 2 * kr
            km = 1 + 1 / np.sqrt(1 + cis ** 2 / 2) + 0.85 * d_meters / t_meters
            m = self.rho0 * t_meters / open_area * km

            zpp = rm + 1j * self.w0 * m
            if rho:
                mip = 1j * self.w0 * rho * t_meters * (1 - open_area)  # Specific mass impedance of the plate
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
                                                               resistance=contact_resistance)
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

        self.matrix[layer] = {"type": "perforated_panel_layer",
                              "thickness [mm]": t,
                              "hole_diameter [mm]": d,
                              "hole_spacing [mm]": s,
                              "open_area [%]": open_area * 100,
                              "end_correction": effective_end_correction,
                              "rho [kg/m3]": rho,
                              "method": method,
                              "porous_contact": porous_contact or None,
                              "contact_resistance [Pa*s/m]": contact_resistance if porous_contact else None,
                              "matrix": Tp,
                              }

    def slotted_panel_layer(self, t=19, w=8, s=16, open_area=None, rho=None, method="barrier", porous_contact=0,
                            layer=None):
        """
        Add a regular long-slot panel as a lumped transfer-matrix layer.

        This method models a plate with infinitely long, periodically spaced slits as a local surface impedance with
        transfer matrix ``[[1, Zs], [0, 1]]``. The current public implementation supports only ``method="barrier"``.

        The open-area ratio is ``phi = w/s`` unless ``open_area`` is supplied directly. The slot end correction follows
        the long-slot expression of Kristiansen and Vigran [1], ``t_eff = t + 2*w*(-1/pi)*log(sin(pi*phi/2))``, and the
        slot impedance is assembled as ``Zs = (Rp + 1j*omega*rho0*t_eff) / phi``, where
        ``Rp = 0.5*sqrt(2*eta*rho0*omega)*(4 + 2*t/w)`` represents viscous losses in the slot. If ``rho`` is supplied,
        the solid-fraction plate mass ``1j*omega*rho*t*(1 - phi)`` is coupled in parallel with the slot impedance, so
        the mass path scales with the closed area fraction of the sheet.

        With ``porous_contact`` set to ``K > 0``, the resistance ``K*sigma*delta`` is added to ``Rp``, where
        ``delta = -(w/pi)*log(sin(pi*phi/2))`` is the one-side end correction and ``sigma`` sums the flow resistivity
        of the ``porous_layer`` entries directly before and after the plate. Ingard [2], equation 3, derived the term
        (``K = 1``) for circular holes in contact with a porous layer; applying it to slots and scaling it by
        ``0 <= K <= 1`` follows [3], section 5.2.4.1. ``compute()`` re-evaluates the term from the neighbours present
        at that time.

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
            solid-fraction mass ``1j*omega*rho*t*(1 - phi)`` is coupled in parallel with the slot impedance.
        method : string, optional
            Slotted-panel calculation method. Only ``"barrier"`` is supported.
        porous_contact : float, optional
            Porous-contact factor ``K`` from 0 to 1. ``0`` disables the term.
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.

        References
        ----------
        [1] U. R. Kristiansen and T. E. Vigran, "On the design of resonant absorbers using a slotted plate", Applied
            Acoustics, vol. 43, pp. 39-48, 1994.
        [2] U. Ingard, "Perforated facing and sound absorption", Journal of the Acoustical Society of America, vol. 26,
            pp. 151-154, 1954.
        [3] T. E. Vigran, "Manual for NorFLAG, version 4.0", Trondheim, 2018.
        """
        if method != "barrier":
            raise ValueError("slotted_panel_layer method must be 'barrier'.")
        self._validate_porous_contact(porous_contact)
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

        end_length = w_meters * (-1 / np.pi) * np.log(np.sin(0.5 * np.pi * open_area))  # One side
        t_corr = t_meters + 2 * end_length
        contact_resistance = self._porous_contact_resistance(layer, porous_contact, end_length)
        vis = self.air_prop["air_viscosity"]
        Rp = 0.5 * np.sqrt(2 * vis * self.rho0 * self.w0) * (4 + (2 * t) / w) + contact_resistance
        Xp = self.rho0 * t_corr
        zs = (Rp + 1j * self.w0 * Xp) / open_area
        if rho:
            mip = 1j * self.w0 * rho * t_meters * (1 - open_area)  # Specific mass impedance of the plate
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
                              "slot_spacing [mm]": s,
                              "open_area [%]": open_area * 100,
                              "rho [kg/m3]": rho,
                              "method": method,
                              "porous_contact": porous_contact or None,
                              "contact_resistance [Pa*s/m]": contact_resistance if porous_contact else None,
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

    def rectangular_slot_layer(self, t=10, w=5, l=20, sw=10, sl=30, rho=None, resistance=0, porous_contact=0,
                               layer=None):
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

        where ``R`` is a series flow resistance such as a fabric. The term ``1j*omega*rho_r*t`` treats the slot air as a
        lumped mass, which holds while the plate is much thinner than the wavelength. If ``rho`` is supplied, the
        full-surface plate mass ``1j*omega*rho*t`` is coupled in parallel with ``Z_ap``, as in the membrane and aperture
        circuit of Kang and Fuchs [5], equations 1 and 9.

        With ``porous_contact`` set to ``K > 0``, the resistance ``K*sigma*delta`` is added to ``R``, where ``sigma``
        sums the flow resistivity of the ``porous_layer`` entries directly before and after the plate. Ingard [6],
        equation 3, derived the term (``K = 1``) for circular holes in contact with a porous layer; applying it to
        rectangular slots and scaling it by ``0 <= K <= 1`` follows [7], section 5.2.4.1. ``compute()`` re-evaluates
        the term from the neighbours present at that time.

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
        resistance : float, optional
            Flow resistance in series with the slots, such as a fabric on the plate [Pa*s/m].
        porous_contact : float, optional
            Porous-contact factor ``K`` from 0 to 1. ``0`` disables the term.
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.

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
        if resistance < 0:
            raise ValueError("resistance must not be negative.")
        self._validate_porous_contact(porous_contact)
        if layer is None:
            layer = len(self.matrix)

        t_meters, w_meters, l_meters, sw_meters, sl_meters = (value / 1000 for value in (t, w, l, sw, sl))
        open_area = (w_meters * l_meters) / (sw_meters * sl_meters)
        end_correction = self._ingard_rectangular_end_correction(w_meters, l_meters, sw_meters, sl_meters)
        rho_r = self._rectangular_pore_density(w_meters, l_meters)
        contact_resistance = self._porous_contact_resistance(layer, porous_contact, end_correction)

        zs = (resistance + contact_resistance + 1j * self.w0 * rho_r * t_meters
              + 1j * self.w0 * self.rho0 * 2 * end_correction) / open_area
        if rho:
            plate_mass = 1j * self.w0 * rho * t_meters
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
                              "resistance [Pa*s/m]": resistance,
                              "rho [kg/m3]": rho,
                              "porous_contact": porous_contact or None,
                              "contact_resistance [Pa*s/m]": contact_resistance if porous_contact else None,
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

        previous_backing = None
        if self.matrix:
            last_layer = self.matrix[list(self.matrix.keys())[-1]]
            if isinstance(last_layer, dict) and last_layer.get("type") == "backing":
                previous_backing = last_layer

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

    def _compute_global_matrix(self, backing, backing_was_omitted, conj):
        """Multiply the layer matrices, apply the termination, and store the resulting impedances."""
        self.matrix = dict(collections.OrderedDict(sorted(self.matrix.items())))

        if "rigid_backing" in self.matrix[list(self.matrix.keys())[-1]]:
            self.matrix.pop(list(self.matrix.keys())[-1])
            if backing_was_omitted and backing is None:
                backing = "rigid"

        # Sheets with porous contact read their neighbours when built, and layers may be added in any order, so
        # they are rebuilt here against the final stack.
        for key, record in list(self.matrix.items()):
            if isinstance(record, dict) and record.get("porous_contact"):
                self._add_layer_from_record(key, record)

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

    def _add_layer_from_record(self, key, value):
        """Re-create one stored layer at ``key`` from its ``self.matrix`` metadata."""
        if value["type"] == "porous_layer":
            self.porous_layer(sigma=value["flow_resistivity [k*Pa*s/m²]"],
                              t=value["thickness [mm]"],
                              model=value["model"],
                              fibre_type=value.get("fibre_type", 1),
                              porosity=value.get("porosity", 0.95),
                              tortuosity=value.get("tortuosity", 1.0),
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
                                        layer=key)
        elif value["type"] == "slotted_panel_layer":
            self.slotted_panel_layer(t=value["thickness [mm]"],
                                     w=value["slot_width [mm]"],
                                     s=value["slot_spacing [mm]"],
                                     rho=value["rho [kg/m3]"],
                                     method=value["method"],
                                     porous_contact=value.get("porous_contact") or 0,
                                     layer=key)
        elif value["type"] == "rectangular_slot_layer":
            self.rectangular_slot_layer(t=value["thickness [mm]"],
                                        w=value["slot_width [mm]"],
                                        l=value["slot_length [mm]"],
                                        sw=value["slot_spacing_width [mm]"],
                                        sl=value["slot_spacing_length [mm]"],
                                        rho=value["rho [kg/m3]"],
                                        resistance=value["resistance [Pa*s/m]"],
                                        porous_contact=value.get("porous_contact") or 0,
                                        layer=key)
        elif value["type"] == "membrane_layer":
            self.membrane_layer(t=value["thickness [mm]"],
                                rho=value["density [kg/m³]"],
                                layer=key)
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
                                     layer=key)

    def rebuild(self):
        """Rebuild the layer matrices from the stored metadata and recompute the results."""
        matrix = self.matrix.copy()
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

        if matrix[list(matrix.keys())[-1]]["type"] == "backing":
            backing_data = matrix[list(matrix.keys())[-1]]
            if "diffuse_method" in backing_data:
                self._diffuse_method = (current_diffuse_method if preserve_diffuse_method
                                        else backing_data["diffuse_method"])
            self.compute(rigid_backing=backing_data["rigid_backing"],
                         backing=backing_data.get("backing"),
                         conj=backing_data["impedance_conjugate"],
                         show_layers=False)
        else:
            self.compute(rigid_backing=False, show_layers=False)
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
            grid_args = f"fmin={self.fmin}, fmax={self.fmax}, df={self.df:0.2f}, "
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
                    logged_calls.append(
                        f"{self.filename}.porous_layer("
                        f"model='{value['model']}', "
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"sigma={value['flow_resistivity [k*Pa*s/m²]']}"
                        f"{optional_args})"
                        )
                elif value["type"] == "air_layer":
                    logged_calls.append(f"{self.filename}.air_layer(t={value['thickness [mm]']:0.1f})")
                elif value["type"] == "perforated_panel_layer":
                    contact_arg = f", porous_contact={value['porous_contact']}" if value.get("porous_contact") else ""
                    end_correction_arg = ""
                    if value.get("method") in {"barrier", "eq_fluid", "bessel_ingard"} \
                            and value.get("end_correction") is not None:
                        end_correction_arg = f"end_correction='{value['end_correction']}', "
                    logged_calls.append(
                        f"{self.filename}.perforated_panel_layer("
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"d={value['hole_diameter [mm]']:0.1f}, "
                        f"s={value['hole_spacing [mm]']:0.1f}, "
                        f"{end_correction_arg}"
                        f"rho={value['rho [kg/m3]']}, "
                        f"method='{value['method']}'"
                        f"{contact_arg}"
                        f")"
                        )
                elif value["type"] == "slotted_panel_layer":
                    contact_arg = f", porous_contact={value['porous_contact']}" if value.get("porous_contact") else ""
                    logged_calls.append(
                        f"{self.filename}.slotted_panel_layer("
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"w={value['slot_width [mm]']:0.1f}, "
                        f"s={value['slot_spacing [mm]']:0.1f}, "
                        f"rho={value['rho [kg/m3]']}, "
                        f"method='{value['method']}'"
                        f"{contact_arg}"
                        f")"
                        )
                elif value["type"] == "rectangular_slot_layer":
                    contact_arg = f", porous_contact={value['porous_contact']}" if value.get("porous_contact") else ""
                    logged_calls.append(
                        f"{self.filename}.rectangular_slot_layer("
                        f"t={value['thickness [mm]']!r}, "
                        f"w={value['slot_width [mm]']!r}, "
                        f"l={value['slot_length [mm]']!r}, "
                        f"sw={value['slot_spacing_width [mm]']!r}, "
                        f"sl={value['slot_spacing_length [mm]']!r}, "
                        f"rho={value['rho [kg/m3]']!r}, "
                        f"resistance={value['resistance [Pa*s/m]']!r}"
                        f"{contact_arg}"
                        f")"
                        )
                elif value["type"] == "membrane_layer":
                    logged_calls.append(
                        f"{self.filename}.membrane_layer("
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"rho={value['density [kg/m³]']}"
                        f")"
                        )
                elif value["type"] == "porous_facing_layer":
                    free_arg = f", free={value['free']}" if "free" in value else ""
                    logged_calls.append(
                        f"{self.filename}.porous_facing_layer("
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"surface_mass={value['surface_mass [kg/m2]']}, "
                        f"resistance={value['resistance [Pa*s/m]']}, "
                        f"porosity={value['porosity']}, "
                        f"tortuosity={value['tortuosity']}, "
                        f"method='{value['method']}'"
                        f"{free_arg}"
                        f")"
                        )
            else:
                logged_calls.append(f"{self.filename}.material_model('{value['type']}', params={value['params']})")

        if matrix[list(matrix.keys())[-1]]["type"] == "backing":
            backing_data = matrix[list(matrix.keys())[-1]]
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

    def print_rebuild(self):
        """Print the calls needed to recreate this object."""
        logged_calls = self.log_rebuild()
        for call in logged_calls:
            print(call)

    @staticmethod
    def _is_report_number(value):
        """Return True for real scalar values that can be formatted numerically."""
        return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))

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
        elif layer_type == "air_layer":
            keys = ["type", "thickness [mm]"]
        elif layer_type == "membrane_layer":
            keys = ["type", "thickness [mm]", "density [kg/m³]"]
        elif layer_type == "porous_facing_layer":
            keys = ["type", "method", "thickness [mm]", "surface_mass [kg/m2]", "resistance [Pa*s/m]"]
            if method == "rebillard":
                keys.extend(["porosity", "tortuosity"])
            elif method == "pieren":
                keys.append("free")
            keys.extend(["rho [kg/m3]", "pore_diameter [mm]", "pore_spacing [mm]", "open_area [%]"])
        elif layer_type == "perforated_panel_layer":
            keys = [
                "type",
                "method",
                "thickness [mm]",
                "hole_diameter [mm]",
                "hole_spacing [mm]",
                "open_area [%]",
            ]
            if method in {"barrier", "eq_fluid", "bessel_ingard"}:
                keys.append("end_correction")
            keys.extend(["rho [kg/m3]", "porous_contact", "contact_resistance [Pa*s/m]"])
        elif layer_type == "slotted_panel_layer":
            keys = ["type", "method", "thickness [mm]", "slot_width [mm]", "slot_spacing [mm]", "open_area [%]",
                    "rho [kg/m3]", "porous_contact", "contact_resistance [Pa*s/m]"]
        elif layer_type == "rectangular_slot_layer":
            keys = ["type", "thickness [mm]", "slot_width [mm]", "slot_length [mm]", "slot_spacing_width [mm]",
                    "slot_spacing_length [mm]", "open_area [%]", "end_correction [mm]", "resistance [Pa*s/m]",
                    "rho [kg/m3]", "porous_contact", "contact_resistance [Pa*s/m]"]
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

    def show_layers(self, conversion=None):
        """
        Print each layer with its details.

        Parameters
        ----------
        conversion : list or float and string, optional
            List containing conversion ratio and string containing the name of the converted unit.
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
                        print(f"\t{key}: {value:0.2f} | {converted}: {value * conversion[0]:0.2f}")
                    else:
                        print(f"\t{key}: {value:0.2f}")
                    if "thickness" in key:
                        total_depth += value
                else:
                    print(f"\t{key}: {value}")
        print(f"\nTotal treatment depth [mm]: {total_depth:0.2f} | " +
              f"Total treatment depth {conversion[1]}: {total_depth * conversion[0]:0.2f}")

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
