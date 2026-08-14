"""
Toolbox for design and prediction of multilayered acoustic treatments. 
Also contains a material model based on the GRAS database.

Developed by Rinaldi Petrolli. 
For questions about usage, bugs, licensing and/or contributions contact me at rinaldipp@gmail.com.

References
----------
[1] R. Petrolli, A. Zorzo and P. D'Antonio, " Comparison of measurement and prediction for acoustical treatments 
    designed with Transfer Matrix Models ", in Euronoise, October 2021.

For further information check the function specific documentation.
"""
import collections
import copy
import os
import warnings

import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.signal import butter, freqz, savgol_filter
from scipy.special import jv

from tmm import _h5utils as h5utils
from tmm import _plot as plot
from tmm import _utils as utils
from tmm.database.path import path as database_path

plot.set_style()


class TMM:
    """
    Transfer Matrix Method model for multilayer acoustic treatments.

    A ``TMM`` object defines a frequency grid, incidence condition, reference
    areas, and a front-to-rear layer stack. Layers are added in physical order
    from the incident face toward the rear termination, then ``compute()``
    evaluates surface impedance, admittance, and absorption for the selected
    backing condition.

    The class also provides plotting, HDF5 persistence, spreadsheet/CSV export,
    and fractional-octave post-processing helpers. ``display_name`` and
    ``color`` are cosmetic metadata used by plots and reports; they do not
    change the transfer-matrix calculation.
    """
    def __init__(self, fmin=20, fmax=5000, df=1, incidence="diffuse", incidence_angle=None, project_folder=None,
                 filename=None, display_name=None, color=None, x_scale="lin", diffuse_method="field", s0=1.0,
                 srad=None, freq=None):
        """
        Parameters
        ----------
        fmin : int, optional
            Minimum frequency of interest.
        fmax : int, optional
            Maximum frequency of interest.
        df : int, optional
            Frequency resolution.
        incidence : string, optional
            String containing the desired type of incidence. ``'normal'`` is normal incidence, ``'angle'`` is a
            single oblique angle, and ``'diffuse'`` is a field-incidence approximation. Diffuse-incidence
            assumptions may not be appropriate for all treatment types.
        incidence_angle : list of ints, optional
            List containing the minimum and maximum incidence angles and the step size.
        project_folder : string, optional
            Path to which files will be saved. If None is passed the current directory will be used.
        filename : string, optional
            Filename that will be used to save data and image files.
        display_name : string, optional
            Human-readable treatment name used in reports and plots. This is independent from ``filename``.
        color : string, optional
            Matplotlib color specification used for cosmetic plotting and reporting only.
        x_scale : string, optional
            X axis scale - 'lin' for linear or 'log' for logarithmic.
        diffuse_method : string, optional
            Method used when ``incidence='diffuse'``. ``'field'`` computes a field-incidence impedance from an
            angular admittance average over the configured incidence angles. ``'paris'`` computes the
            statistical diffuse absorption coefficient as a ratio of angular integrals: the numerator integrates
            ``alpha(theta)*sin(theta)*cos(theta)`` and the denominator integrates ``sin(theta)*cos(theta)`` over
            the configured angular range. See ``field_impedance()`` for the field-incidence impedance formula.
        s0 : float, optional
            Front/reference area in square meters used by the volume-velocity transfer matrices. The default
            value of 1.0 keeps impedances in specific-acoustic-impedance form.
        srad : float, optional
            Rear termination area in square meters. For ``backing='radiation'`` this is interpreted as the area
            of the circular radiating aperture. If omitted, ``srad`` defaults to ``s0``.
        freq : array_like, optional
            Explicit frequency vector in Hz. When provided it takes precedence over ``fmin``/``fmax``/``df`` and
            ``fmin``/``fmax`` are derived from its endpoints. The spacing may be non-uniform because the TMM
            evaluates every frequency independently. When omitted, the uniform grid built from
            ``fmin``/``fmax``/``df`` is used.
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
        if freq is None:
            self._fmin = fmin
            self._fmax = fmax
            self._freq = None
        else:
            self._freq = self._as_frequency_vector(freq)
            self._fmin = float(self._freq[0])
            self._fmax = float(self._freq[-1])
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

        The tolerance absorbs the float error of the ``linspace`` angular grid, including the value just
        below 90 degrees that the diffuse range substitutes for grazing incidence.
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
        Spacing may be non-uniform — TMM evaluates each frequency independently."""
        freq = np.asarray(freq, dtype=float).ravel()
        if freq.size == 0:
            raise ValueError("freq must contain at least one frequency in Hz.")
        if freq.size > 1 and np.any(np.diff(freq) <= 0):
            raise ValueError("freq must be strictly ascending.")
        return freq

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

    @property
    def fmin(self):
        """Return minimum frequency of analysis."""
        return self._fmin

    @fmin.setter
    def fmin(self, new_fmin):
        """Set new minimum frequency value."""
        self._fmin = new_fmin
        self._mark_layers_stale("fmin was changed after model data existed.")

    @property
    def fmax(self):
        """Return maximum frequency of analysis."""
        return self._fmax

    @fmax.setter
    def fmax(self, new_fmax):
        """Set new maximum frequency value."""
        self._fmax = new_fmax
        self._mark_layers_stale("fmax was changed after model data existed.")

    @property
    def df(self):
        """Return frequency resolution."""
        return self._df

    @df.setter
    def df(self, new_df):
        """Set new frequency resolution value."""
        self._df = new_df
        self._mark_layers_stale("df was changed after model data existed.")

    @property
    def freq(self):
        """Return frequency values."""
        if self._freq is None:
            if self._x_scale == "lin":
                return np.linspace(self.fmin,
                                   self.fmax,
                                   int((self.fmax - self.fmin) / self.df) + 1).round(1)
            elif self._x_scale == "log":
                return np.logspace(np.log10(self.fmin),
                                   np.log10(self.fmax),
                                   int((self.fmax - self.fmin) / self.df) + 1).round(1)
        else:
            return self._freq

    @freq.setter
    def freq(self, new_freq):
        """Set an explicit frequency vector (Hz). Spacing may be non-uniform."""
        self._freq = self._as_frequency_vector(new_freq)
        self._fmin = float(self._freq[0])
        self._fmax = float(self._freq[-1])
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

        This matches ``incidence_angle`` for a normally computed treatment. On an object returned by
        ``reduced_copy()`` it reports only the retained angles, and it is empty when ``z_angle`` was
        discarded. Column ``i`` of ``z_angle`` corresponds to ``stored_angles[i]``, so this is the honest
        answer to "which angles does this object still have?" for a trimmed treatment.
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
        """Return frequency-grid spacing: ``'lin'`` for linear or ``'log'`` for logarithmic."""
        return self._x_scale

    @x_scale.setter
    def x_scale(self, new_x_scale):
        """Set frequency-grid spacing to ``'lin'`` or ``'log'``."""
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
        """Return scattering coefficient (given by material_model only)."""
        if self._scat is not None:
            return self._scat
        else:
            return np.zeros_like(self.freq)

    @scat.setter
    def scat(self, new_scat):
        """Set scattering coefficient."""
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
        """Returns the treatment depth in millimeters."""
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

        The normal-incidence relation is ``r = (Zs - Z0) / (Zs + Z0)``. For oblique incidence, ``Zs`` is treated
        as the surface-normal impedance at the boundary and the incident plane wave has normal characteristic
        impedance ``Z0 / cos(theta)``. The implemented reflection coefficient is therefore
        ``r(theta) = (Zs*cos(theta) - Z0) / (Zs*cos(theta) + Z0)`` and
        ``alpha(theta) = 1 - abs(r(theta))**2``.

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
            Positional index of the desired angle in 'self.incidence_angle'. On an object returned by
            'reduced_copy()' this indexes the retained angles instead, and the reflection is evaluated at
            the angle those columns were actually computed for.

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
        configured angular range. The ``sin(theta)`` term represents the solid-angle density for an isotropic
        field over a hemisphere, while ``cos(theta)`` projects incident intensity onto the surface normal.
        This averaging produces a scalar absorption coefficient; it is not a definition of a diffuse complex
        impedance. Use ``field_impedance()`` when a scalar field-incidence impedance is needed.
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

    def equivalent_fluid_model(self, sigma, model="mac", fibre_type=1, porosity=0.95, tortuosity=1.0):
        """
        Return the complex propagation constant ``kc`` and characteristic impedance
        ``zc`` for an equivalent-fluid porous material.

        These models treat the porous layer as an equivalent fluid with a complex
        propagation constant and characteristic impedance. Empirical branches are
        useful when flow resistivity is the only measured material parameter, which
        is common in practical absorber design, but the selected model should remain
        compatible with the material class and applicability range of its source
        formulation.

        The coefficient-set models use the common power-law form
        ``kc = k0*(1 + C1*X**(-C2) - 1j*C3*X**(-C4))`` and
        ``Zc = Z0*(1 + C5*X**(-C6) - 1j*C7*X**(-C8))``, where
        ``X = rho0*f/sigma`` and ``sigma`` is in SI units [Pa*s/m2] inside this
        method. This family includes ``db``, ``miki``, ``qunli``, ``mac``,
        ``mechel_gf_lowX``, ``mechel_gf_highX``, ``mechel_rf_lowX``, and
        ``mechel_rf_highX``. The implementation follows the generic empirical
        equivalent-fluid form presented in Cox and D'Antonio, *Acoustic Absorbers
        and Diffusers*, 3rd ed., section 6.5.1. The coefficient table is
        consistent with the Delany-Bazley-Miki model options listed in the COMSOL
        Acoustics Module porous-model documentation:
        https://doc.comsol.com/6.4/doc/com.comsol.help.aco/aco_ug_pressure.05.005.html#1226466.

        The remaining branches use separate equations rather than the shared
        coefficient-set form. ``komatsu`` follows Komatsu (2008), equations 15-18,
        using powers of ``2 - log10(f/sigma)``. ``mechel_1976`` follows Mechel's
        low-frequency extension of the Delany-Bazley absorber formula and uses
        ``porosity`` as the open porosity ``h``. ``mechel_grundmann`` keeps the
        Mechel-Grundmann polynomial formulation from the Cox and D'Antonio reference
        implementation. ``wilson_2015`` implements Wilson (2015), equations 24-25,
        with unit shape factors; ``porosity`` maps to ``phi`` and ``tortuosity``
        is interpreted as the high-frequency tortuosity-like factor
        ``alpha_inf = q**2``, so Wilson's internal ``q`` is computed as
        ``sqrt(tortuosity)``. ``wilson_db`` keeps the compact Wilson relaxation
        model from the Cox and D'Antonio MATLAB reference script.

        Model-selection notes: use the empirical coefficient-set models when only
        flow resistivity is known and the material falls within the source model's
        intended range. Use ``mechel_1976`` when open porosity is part of the
        material definition. Use ``wilson_2015`` when porosity and tortuosity are
        known or can be specified. Use ``qunli`` for porous plastics and open foams
        within the COMSOL-listed range of approximately 200-2000 Hz and
        ``3e3 <= sigma <= 24e3`` Pa*s/m2. Use ``mac`` for the Modified
        Champoux-Allard coefficient set within the listed frequency range of
        approximately 45-11000 Hz.

        Parameters
        ----------
        sigma : float
            Flow resistivity in SI units [Pa*s/m2]. ``porous_layer()`` accepts
            [kPa*s/m2] and converts before calling this method.
        model : str, optional
            Equivalent-fluid model name.
        fibre_type : int, optional
            Fibre type for ``mechel_grundmann``. ``1`` is basalt/rock wool and ``2``
            is glass fibre.
        porosity : float, optional
            Open pore volume fraction for branches that expose porosity. Used by
            ``mechel_1976``, ``wilson_2015``, and ``wilson_db``.
        tortuosity : float, optional
            High-frequency tortuosity-like factor for ``model="wilson_2015"``.
            Wilson's internal ``q`` parameter is computed as
            ``sqrt(tortuosity)``.

        Returns
        -------
        kc, zc : ndarray
            Complex propagation constant and characteristic acoustic impedance.
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
        Adds a layer of porous material to the existing device.

        The porous material is treated as an equivalent fluid with complex characteristic impedance ``Zc`` and
        complex wavenumber ``kc`` from the selected empirical model. For oblique incidence, the tangential
        wavenumber is conserved at the air/material interface:
        ``kt = k0*sin(theta)`` and ``kz = sqrt(kc**2 - kt**2)``. The layer transfer matrix is then built with
        the normal wavenumber ``kz`` and the normal characteristic impedance ``Zc,n = Zc*kc/kz``:
        ``[[cos(kz*t), j*Zc,n*sin(kz*t)], [j*sin(kz*t)/Zc,n, cos(kz*t)]]``.
        This is the standard equivalent-fluid extended-reaction form; for notation, see Allard and Atalla,
        *Propagation of Sound in Porous Media*, 2nd ed., 2009.

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
        porosity : float, optional
            Open pore volume fraction for models that expose it. Currently this is used by
            ``model="wilson_2015"``, ``model="wilson_db"``, and ``model="mechel_1976"``.
        tortuosity : float, optional
            High-frequency tortuosity-like factor for ``model="wilson_2015"``.
            Wilson's internal ``q`` is computed as ``sqrt(tortuosity)``.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.
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
        Adds an air layer to the existing device.

        For an inviscid air layer under oblique plane-wave incidence, the tangential wavenumber is conserved
        and the normal propagation constant is ``k0*cos(theta)``. The pressure/volume-velocity transfer matrix
        therefore uses the normal air impedance ``Z0/cos(theta)``: ``A = D = cos(k0*t*cos(theta))``,
        ``B = j*Z0/cos(theta)*sin(k0*t*cos(theta))`` and
        ``C = j*cos(theta)/Z0*sin(k0*t*cos(theta))`` before area scaling.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the air layer [mm]
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.
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

        This is an auxiliary direct-assignment utility rather than a physical transfer-matrix layer. The supplied
        scalar is interpreted as a specific surface impedance in the same units as ``self.z``. It is copied to
        ``self.z`` with length ``len(self.freq)`` and to ``self.z_angle`` with shape
        ``(len(self.freq), len(self.incidence_angle))``. A perfectly matched normal-incidence surface is obtained
        with ``z=self.z0``; a pressure-release surface can be represented with ``z=0``. Very large values approximate
        a rigid wall. For angle incidence and ``diffuse_method="paris"``, the cached absorption coefficient is
        calculated from the configured incidence angles.

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

        The membrane is modeled as a local pressure-jump impedance with surface mass
        ``m_s = rho*t`` and specific impedance ``Z_m = 1j*omega*m_s``, where ``t`` is
        converted from millimeters to meters. The transfer matrix is
        ``[[1, Z_m/S0], [0, 1]]`` because this class uses pressure and volume velocity
        as the matrix state; ``S0`` is the front surface area. The resulting surface
        impedance recovered by ``compute()`` is scaled back to specific impedance.

        This is the mass term used in the classical limp-membrane absorber model.
        Brandão, *Acustica de Salas: Projeto e Modelagem*, section 2.3.2, writes the
        membrane impedance as ``Z_m = 1j*omega*m_s`` after neglecting membrane
        resistance, then combines it in series with the surface impedance of the
        backing air or porous cavity. This method only adds the membrane term; air
        cavities, porous layers, and rigid/radiation/backing conditions should be
        modeled with subsequent layers and ``compute()``.

        The model neglects bending stiffness, finite-panel modal behavior,
        edge/support losses, and intrinsic membrane resistance. It is appropriate for
        a limp local-reaction mass sheet, not a full structural panel model.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the membrane [mm]
        rho : float or int, optional
            Material density [kg/m³]
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.
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

        The layer is represented by a zero-thickness transfer matrix
        ``[[1, Zs/S0], [0, 1]]``, where ``Zs`` is the specific sheet impedance
        and ``S0`` is the front surface area used by the volume-velocity matrix
        state.

        ``method="rebillard"`` implements the unbonded thin porous-facing
        expression from Rebillard et al. (1992), "The effect of a porous facing
        on the impedance and the absorption coefficient of a layer of porous
        material". The implementation uses the negligible-flexural-stiffness
        limit. With open porosity ``h``, tortuosity ``tau``, facing thickness
        ``e``, surface mass ``m``, and flow resistance ``R = sigma*e``, the
        coupling term is ``I = 1j*omega*(tau - 1)*h*rho0*e + R*h**2``.

        ``method="pieren"`` implements the thin woven-fabric sheet impedance
        described by Pieren (2012), "Sound absorption modeling of thin woven
        fabrics backed by an air cavity". With ``free=True``, the sheet
        resistance and limp surface mass are combined in parallel,
        ``Zs = 1j*omega*m*R / (1j*omega*m + R)``. With ``free=False``, the
        motionless fabric limit ``Zs = R`` is used. The aliases
        ``method="pieren_free"`` and ``method="pieren_fixed"`` set this option
        explicitly.

        For ``method="rebillard"``, omitting ``surface_mass`` and
        ``resistance`` uses a nominal thin porous-facing sheet with
        ``surface_mass=0.25 kg/m2`` and ``resistance=200 Pa*s/m``; the other
        defaults are ``t=1 mm``, ``porosity=0.7``, and ``tortuosity=1``. Direct
        measured ``surface_mass`` and ``resistance`` values are preferred for
        real fabrics. If ``surface_mass`` is set to ``None`` and ``rho`` is
        supplied, it can be estimated as ``rho*t``; if ``resistance`` is set to
        ``None``, it can be estimated from the ideal circular-pore expression
        ``32*eta*t/(open_area*d**2)``. These estimates are convenience
        fallbacks; real fabrics often require measured airflow resistance.

        Parameters
        ----------
        t : float or int, optional
            Facing thickness [mm].
        surface_mass : float or None, optional
            Mass per unit area [kg/m2]. If ``None`` and ``rho`` is supplied, it
            is estimated as ``rho*t``.
        resistance : float or None, optional
            Specific airflow resistance [Pa*s/m]. If ``None``, it is estimated
            from ``d`` and ``open_area``.
        porosity : float, optional
            Open porosity used by the Rebillard model.
        tortuosity : float, optional
            Tortuosity used by the Rebillard model.
        method : string, optional
            ``"rebillard"``, ``"pieren"``, ``"pieren_free"``, or
            ``"pieren_fixed"``.
        free : bool, optional
            For ``method="pieren"``, choose the free-moving fabric expression
            when ``True`` or the motionless resistance-only limit when ``False``.
        rho : float or None, optional
            Apparent fabric density [kg/m3], used only to estimate
            ``surface_mass`` when needed.
        d : float or None, optional
            Equivalent pore diameter [mm], used only to estimate ``resistance``.
        s : float or None, optional
            Equivalent pore spacing [mm], used to estimate ``open_area`` when
            needed.
        open_area : float or None, optional
            Open-area ratio used for the geometric resistance estimate.
        layer : None or int, optional
            Optional layer index. If ``None``, the layer is appended.
        """
        method = method.lower()
        if method == "pieren_free":
            method = "pieren"
            free = True
        elif method == "pieren_fixed":
            method = "pieren"
            free = False
        elif method not in {"rebillard", "pieren"}:
            raise ValueError("porous_facing_layer method must be 'rebillard', 'pieren', 'pieren_free', or 'pieren_fixed'.")

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

    def _bessel_ingard_circular_sheet_impedance(self, t_meters, d_meters, open_area, rho=None):
        """
        Return the local sheet impedance for ``perforated_panel_layer(method="bessel_ingard")``.

        This combines a Bessel short-tube aperture impedance, an Ingard square-cell end correction for a regular
        circular-hole array, a real aperture end-resistance term, and optional full-surface plate-mass coupling in
        parallel with the aperture path. The branch is source-assembled rather than a single closed-form equation
        from one reference.
        """
        radius = d_meters / 2
        viscosity = self.air_prop["air_viscosity"]
        kd = radius * np.sqrt(self.rho0 * self.w0 / viscosity)
        bessel_argument = kd * np.sqrt(-1j)
        aperture_factor = 1 - 2 * jv(1, bessel_argument) / (bessel_argument * jv(0, bessel_argument))
        z_aperture = 1j * self.w0 * self.rho0 * t_meters / (open_area * aperture_factor)

        hole_area = np.pi * radius ** 2
        xi = np.sqrt(open_area)
        end_length = 0.96 * np.sqrt(hole_area) * (1 - 1.24 * xi + 0.27 * xi ** 2)
        z_end = 1j * self.w0 * self.rho0 * end_length / open_area
        z_end_resistance = np.sqrt(2) * kd * viscosity / (2 * radius * open_area)
        z_sheet = z_aperture + z_end + z_end_resistance

        if rho is not None:
            plate_mass = 1j * self.w0 * rho * t_meters
            z_sheet = 1 / (1 / z_sheet + 1 / plate_mass)

        return z_sheet

    def perforated_panel_layer(self, t=19, d=8, s=16, open_area=None, rho=None, end_correction=None, method="barrier",
                               layer=None):
        """
        Add a perforated plate with circular holes as a transfer-matrix layer.

        The available formulations fall into two matrix topologies. ``"barrier"``, ``"barrier_mpp"``, and
        ``"bessel_ingard"`` model the perforated plate as a local surface impedance using a pressure-jump transfer
        matrix. Their aperture and optional plate-mass terms are first combined as specific impedances; the matrix
        entry is then ``Zp/S0`` because the package matrices use volume velocity, where ``S0`` is ``self.s0``.
        ``"eq_fluid"`` treats the holes as finite viscothermal ducts and uses a finite-length propagation matrix.
        When ``rho`` is supplied, the optional plate-mass path is coupled in parallel with the perforation
        impedance. The ``"bessel_ingard"`` branch uses a full-surface limp mass ``1j*omega*rho*t``. The
        ``"barrier"`` and ``"barrier_mpp"`` branches use the solid-fraction convention
        ``1j*omega*rho*t*(1 - phi)``.

        ``method="barrier"`` implements the Helmholtz-resonator perforated-sheet impedance of Cox and D'Antonio,
        *Acoustic Absorbers and Diffusers*, 3rd ed., equation 7.21 (section 7.3.3.3), which combines the acoustic
        mass of equation 7.6 with the viscous resistance of equation 7.12. The sheet impedance is
        ``Zp = Rm + 1j*omega*rho0*t_eff/phi``, where ``phi`` is the open-area ratio,
        ``t_eff = 2*delta*a + t`` for hole radius ``a`` and end-correction factor ``delta``, and
        ``Rm = sqrt(8*eta*rho0*omega)*(1 + t/(2*a))/phi``. The book writes ``Rm`` with the kinematic viscosity as
        ``(rho0/phi)*sqrt(8*nu*omega)*(1 + t/(2*a))``; the two forms are equal because ``nu = eta/rho0``, and
        ``air_prop`` supplies the dynamic viscosity ``eta``. The remaining ``z_s2`` term of equation 7.21 is the
        impedance below the sheet, which the pressure-jump matrix supplies, and the small viscous mass term of
        equation 7.6 is omitted, as it is in equation 7.21 itself.

        If ``rho`` is supplied, a solid-fraction plate-mass approximation ``1j*omega*rho*t*(1 - phi)`` is coupled
        in parallel with the perforation impedance, following the topology of equation 7.16. The book's ``z_mem``
        there is the full panel mass per unit area; the ``(1 - phi)`` weighting is a package convention.

        ``method="barrier_mpp"`` implements Maa's approximate microperforated-panel sheet impedance for circular
        submillimetre perforations. For the pure aperture branch (``rho=None``), the implementation follows Maa
        (1998), *Potential of microperforated panel absorber*, equations 5a, 5b, and 6. Maa writes the impedance
        normalized by ``rho0*c0``; this method converts it to dimensional surface impedance as
        ``Zp = 32*eta*t*kr/(phi*d**2) + 1j*omega*rho0*t*km/phi``, with
        ``k = d*sqrt(omega*rho0/(4*eta))``, ``kr = sqrt(1 + k**2/32) + sqrt(2)*k*d/(32*t)``, and
        ``km = 1 + 1/sqrt(1 + k**2/2) + 0.85*d/t``. If ``rho`` is supplied, the package adds a separate
        solid-fraction plate-mass impedance ``1j*omega*rho*t*(1 - phi)`` in parallel with the aperture path; that
        plate-mass coupling is a package modelling extension and is not part of Maa's pure aperture impedance.

        ``method="eq_fluid"`` treats each circular hole as a finite viscothermal duct. The circular-pore propagation
        constant and characteristic impedance are returned by ``viscothermal_circular()``; the characteristic
        impedance is then scaled by ``1/phi`` to homogenize the pore over the full panel area. The corrected length
        ``t_eff`` is used in the finite transfer matrix
        ``[[cos(kc*t_eff), 1j*Zc/phi*sin(kc*t_eff)], [1j*phi/Zc*sin(kc*t_eff), cos(kc*t_eff)]]``, with the package
        area scaling applied in the same way as other layer matrices.

        ``method="bessel_ingard"`` computes a local circular-aperture sheet impedance using the exact Bessel
        short-tube aperture expression used in the Maa / Cox-D'Antonio microperforation source family. Cox and
        D'Antonio, *Acoustic Absorbers and Diffusers*, 3rd ed., section 7.5.3, recommend evaluating this Bessel
        expression directly rather than using Maa's asymptotic approximation. For hole radius ``a``, angular
        frequency ``omega``, air density ``rho0``, dynamic viscosity ``eta``, and open-area ratio ``phi``,

        ``s = a*sqrt(-1j*omega*rho0/eta)``

        ``D = 1 - 2*J1(s)/(s*J0(s))``

        ``Z_ap = 1j*omega*rho0*t/(phi*D)``.

        The inertial end term uses Ingard's square-cell aperture end-correction family for circular holes in a
        regular array,

        ``Delta = 0.96*sqrt(A0)*(1 - 1.24*sqrt(phi) + 0.27*phi)``,

        where ``A0 = pi*a**2``. It is added as ``Z_end = 1j*omega*rho0*Delta/phi``. The real aperture end-resistance
        term is added as ``sqrt(2)*kd*eta/(2*a*phi)``, with ``kd = a*sqrt(omega*rho0/eta)``. If ``rho`` is supplied,
        the full-surface plate mass ``1j*omega*rho*t`` is coupled in parallel with the aperture path. This branch is
        intended for circular perforations, including microperforated geometries, where a local Bessel sheet
        impedance is preferred over the finite-duct ``eq_fluid`` matrix or the Maa-style ``barrier_mpp``
        approximation.

        If ``open_area`` is not supplied, the method assumes a square pitch and computes
        ``phi = pi*d**2/(4*s**2)``. If ``open_area`` is supplied, ``s`` is replaced by the equivalent square pitch
        for metadata.

        The ``end_correction`` options set the corrected hole length ``t_eff`` only for ``method="barrier"`` and
        ``method="eq_fluid"``. If omitted, these two methods use ``"jb"``. Available options are:
        ``"nesterov"``, the circular-pattern correction listed by Cox and D'Antonio; ``"jb"`` /
        ``"jaouen_becot"``, their square-pattern Jaouen-Becot correction; and ``"beranek"``, the single-hole
        infinite-baffle approximation ``t_eff = t + 0.85*d``. All three use the 0.85 prefactor given in the book's
        text and table 7.1. ``method="barrier_mpp"`` uses Maa's internal
        acoustic-mass term, and ``method="bessel_ingard"`` uses the square-cell aperture end-correction term
        described in its formulation above; neither branch uses the ``end_correction`` argument.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the perforated plate [mm].
        d : float or int, optional
            Hole diameter [mm].
        s : float or int, optional
            Hole spacing from the center of one hole to the next [mm].
        open_area : float, optional
            Ratio of open area. If set to ``None`` it is calculated from the hole spacing ``s``. If a value is
            supplied, the equivalent spacing is calculated and stored.
        rho : float, int or None, optional
            Plate density [kg/m3]. If ``None`` is passed, a fully rigid plate is assumed. If supplied,
            ``method="bessel_ingard"`` uses the full-surface mass ``1j*omega*rho*t`` while ``method="barrier"``
            and ``method="barrier_mpp"`` use the solid-fraction mass ``1j*omega*rho*t*(1 - phi)``.
        end_correction : string or None, optional
            End-correction model for ``method="barrier"`` and ``method="eq_fluid"``. Options are ``"jb"`` /
            ``"jaouen_becot"``, ``"nesterov"``, and ``"beranek"``. If ``None``, ``"jb"`` is used for those two
            methods. This argument is ignored by ``method="barrier_mpp"`` and ``method="bessel_ingard"``.
        method : string, optional
            Chooses the perforated-plate formulation. Available options are ``"barrier"``, ``"barrier_mpp"``,
            ``"eq_fluid"``, and ``"bessel_ingard"``.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be added to the existing ones.
        """
        valid_methods = {"barrier", "barrier_mpp", "eq_fluid", "bessel_ingard"}
        if method not in valid_methods:
            raise ValueError(
                "perforated_panel_layer method must be 'barrier', 'barrier_mpp', 'eq_fluid', or 'bessel_ingard'."
            )

        uses_generic_end_correction = method in {"barrier", "eq_fluid"}
        effective_end_correction = None
        if uses_generic_end_correction:
            effective_end_correction = "jb" if end_correction is None else end_correction
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

        t_corr = None
        if uses_generic_end_correction:
            if effective_end_correction == "nesterov":
                # Cox and D'Antonio, Acoustic Absorbers and Diffusers, Table 7.1:
                # circular holes in a circular pattern.
                delta = 0.85 * (1 - 1.47 * open_area ** (1/2) + 0.47 * open_area ** (3/2))
                t_corr = 2 * delta * d_meters / 2 + t_meters
            elif effective_end_correction == "jaouen_becot" or effective_end_correction == "jb":
                # Cox and D'Antonio, Acoustic Absorbers and Diffusers, Table 7.1:
                # circular holes in a square pattern.
                delta = 0.85 * (1 - 1.13 * open_area ** (1/2) - 0.09 * open_area + 0.27 * open_area ** (3/2))
                t_corr = 2 * delta * d_meters / 2 + t_meters
            elif effective_end_correction == "beranek":
                # Single circular hole in an infinite baffle.
                t_corr = t_meters + 0.85 * d_meters
            else:
                raise ValueError(
                    "perforated_panel_layer end_correction must be 'jb', 'jaouen_becot', 'nesterov', "
                    "or 'beranek'."
                )
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
            Local circular-aperture Bessel sheet with square-cell end correction, aperture end resistance, and
            optional full-surface plate-mass coupling.
            """
            zpp = self._bessel_ingard_circular_sheet_impedance(t_meters, d_meters, open_area, rho=rho)
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

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "perforated_panel_layer",
                              "thickness [mm]": t,
                              "hole_diameter [mm]": d,
                              "hole_spacing [mm]": s,
                              "open_area [%]": open_area * 100,
                              "end_correction": effective_end_correction,
                              "rho [kg/m3]": rho,
                              "method": method,
                              "matrix": Tp,
                              }

    def slotted_panel_layer(self, t=19, w=8, s=16, open_area=None, rho=None, method="barrier", layer=None):
        """
        Add a regular long-slot panel as a lumped transfer-matrix layer.

        This method models a plate with infinitely long, periodically spaced slits as a local surface impedance
        with transfer matrix ``[[1, Zs], [0, 1]]``. The current public implementation supports only
        ``method="barrier"``.

        The open-area ratio is ``phi = w/s`` unless ``open_area`` is supplied directly. The slot end correction
        follows the Kristiansen and Vigran long-slot expression,
        ``t_eff = t + 2*w*(-1/pi)*log(sin(pi*phi/2))``, and the slot impedance is assembled as
        ``Zs = (Rp + 1j*omega*rho0*t_eff) / phi``, where
        ``Rp = 0.5*sqrt(2*eta*rho0*omega)*(4 + 2*t/w)`` represents viscous losses in the slot. If ``rho`` is
        supplied, the solid-fraction plate mass ``1j*omega*rho*t*(1 - phi)`` is coupled in parallel with the slot
        impedance, where ``phi`` is the open-area ratio. This keeps the mass path proportional to the closed area
        fraction of the slotted sheet.

        Reference: U. R. Kristiansen and T. E. Vigran, "On the Design Of Resonant Absorbers Using a Slotted Plate"
        (1994).

        Parameters
        ----------
        t : float or int, optional
            Thickness of the slotted plate [mm]
        w: float or int, optional
            Slit width [mm]
        s : float or int, optional
            Slit spacing from the center of one slit to the next [mm]
        open_area : float, optional
            Ratio of open area. If set to ``None`` it is calculated with the slit spacing ``s``. If a value is
            supplied, the equivalent spacing is calculated and stored.
        rho : float, int or None, optional
            Plate density [kg/m3]. If ``None`` is passed, a fully rigid plate is assumed. If supplied, the
            solid-fraction mass ``1j*omega*rho*t*(1 - phi)`` is coupled in parallel with the slot impedance.
        method : string, optional
            Slotted-panel calculation method. Only ``"barrier"`` is supported.
        layer : None or int, optional
            Optional value to choose the layer level. If None is passed the layer will be adding to the existing ones.
        """
        if method != "barrier":
            raise ValueError("slotted_panel_layer method must be 'barrier'.")

        if open_area is None:
            open_area = w / s
        else:
            s = w / open_area

        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        w_meters = w / 1000

        if open_area > 1:
            raise ValueError("Slit spacing must be larger than slit width.")

        t_corr = t_meters + 2 * w_meters * (-1 / np.pi) * np.log(np.sin(0.5 * np.pi * open_area))
        vis = self.air_prop["air_viscosity"]
        Rp = 0.5 * np.sqrt(2 * vis * self.rho0 * self.w0) * (4 + (2 * t) / w)
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

        if layer is None:
            layer = len(self.matrix)

        self.matrix[layer] = {"type": "slotted_panel_layer",
                              "thickness [mm]": t,
                              "slot_width [mm]": w,
                              "slot_spacing [mm]": s,
                              "open_area [%]": open_area * 100,
                              "rho [kg/m3]": rho,
                              "method": method,
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

        The formulation follows the Zwikker-Kosten/Tijdeman circular tube solution, with notation consistent with
        later equivalent-fluid treatments such as Stinson. It describes propagation inside the pore only.
        Perforated-panel open-area scaling and end corrections are applied by
        ``perforated_panel_layer(method="eq_fluid")``.

        Parameters
        ----------
        d : int or float
            Hole diameter [m]
        open_area : int or float, optional
            Deprecated compatibility argument. If supplied, ``zc`` is scaled by ``1/open_area`` to reproduce the
            previous panel-scaled return value. New code should omit this argument and apply panel scaling in the
            layer builder.

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

    def material_model(self, type="door", params=None):
        """
        Set an empirical boundary impedance model adapted from the GRAS material helpers distributed with
        Hargreaves, Rendell, and Lam (2019), "A framework for auralization of boundary element method simulations
        including source and receiver directivity".

        This method is separate from the transfer-matrix layer builders. It directly assigns ``self.z`` and, where
        available, ``self.scat`` from empirical room-acoustics boundary models used in the GRAS auralization
        examples.

        The ``floor``, ``ceiling``, ``concrete``, ``plaster``, ``door``, and ``window`` branches are Python
        adaptations of the Scene 9 MATLAB helpers ``MaterialModel_Scene9*.m``. The ``mdf`` branch follows
        ``MaterialModel_Scene3MDF.m``. The resistive-data branches load GRAS surface-description absorption data,
        convert random-incidence absorption to a purely real normalized admittance with the 55 degree rule,
        ``Y = cos(55 deg)*(1 - sqrt(1 - alpha))/(1 + sqrt(1 - alpha))``, and interpolate the admittance onto
        ``self.freq``. TMM intentionally uses natural cubic spline interpolation for these empirical fits. This
        differs from the MATLAB helper endpoint-slope spline convention and avoids unstable endpoint artifacts in
        sparse empirical data. Where scattering data are available in the bundled CSV files, ``self.scat`` is
        interpolated in the same way. The scattering output is a TMM package addition from the CSV data; the source
        MATLAB helpers return admittance only.

        The ``door`` and ``window`` branches retain the hybrid construction from the original MATLAB helpers. They
        combine a resistive absorption-data fit with a reactive mass-spring-damper panel admittance and a
        Linkwitz-Riley-style crossover. The source helpers describe this blend as the non-linear crossover method
        of Aretz et al. These branches are practical boundary-condition models with explicit assumptions, not
        general material laws. TMM uses the current object's air properties rather than the helper files' hard-coded
        ``rho0=1.21 kg/m3`` and ``c0=343 m/s`` values. The optional ``smooth`` parameter for ``door`` and ``window``
        is a TMM extension.

        GRAS database: https://depositonce.tu-berlin.de//handle/11303/7506
        Supplemental data: https://asa.scitation.org/doi/suppl/10.1121/1.5096171

        Available materials:
        -------------------
         - Floor
         - Ceiling
         - Door
         - Concrete
         - Plaster
         - MDF
         - Window

        Parameters
        ----------
        type : str, optional
            String descriptor of the desired material available in the database.
        params : dict, optional
            Dictionary containing calculation parameters for 'door' and 'window' materials. See the docstrings below.
        """
        params = None if params is None else dict(params)
        self._scat = None

        def material_spline(x, y):
            return CubicSpline(x, y, bc_type="natural")

        if type == "floor":
            """
            This is a model of the floor material defined in Scene 9 of the GRAS database. 
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_floor.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = material_spline(fMeas, YsMeas)
            Sf = material_spline(fMeas, sMeas)
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "ceiling":
            """
            This is a model of the ceiling material defined in Scene 9 of the GRAS database. 
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_ceiling.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = material_spline(fMeas, YsMeas)
            Sf = material_spline(fMeas, sMeas)
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "door":
            """
            Python adaptation of ``MaterialModel_Scene9Door.m`` from the Hargreaves, Rendell, and Lam GRAS Scene 9
            helper code. No direct GRAS door material data were provided, so this branch preserves the source
            helper's constructed boundary model and caveat: it is intended as a practical room-boundary condition
            where insufficient data exist, not as a general door material law.

            It comprises two approaches:

            1) area purely resistive fit to octave-band summed absorption and transmission coefficient data. 
               Both absorption and transmission coefficients were used since the former did not rise at low frequencies, 
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

        elif type == "concrete":
            """
            This is a model of the concrete material defined in Scene 9 of the GRAS database. 
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_concrete.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = material_spline(fMeas, YsMeas)
            Sf = material_spline(fMeas, sMeas)
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "plaster":
            """
            This is a model of the plaster material defined in Scene 9 of the GRAS database. 
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_plaster.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = material_spline(fMeas, YsMeas)
            Sf = material_spline(fMeas, sMeas)
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "mdf":
            """
            This is a Python adaptation of ``MaterialModel_Scene3MDF.m`` for the MDF material defined in Scene 3 of
            the GRAS database.
            It is a purely real (resistive) admittance found from the measured absorption coefficient data using a 
            spline fit.
            """
            # Load the random incidence absorption coefficient data included in the GRAS database:
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_MDF25mmA_plane_00deg.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

            # Convert to purely real admittance assuming material follows '55 degree rule':
            YsMeas = np.cos(np.deg2rad(55)) * (1 - np.sqrt(1 - aMeas)) / (1 + np.sqrt(1 - aMeas))

            # Interpolate to specific frequency list using a spline fit:
            Yf = material_spline(fMeas, YsMeas)
            Sf = material_spline(fMeas, sMeas)
            YsInterp = Yf(self.freq)
            SsInterp = Sf(self.freq)

            self.z = 1 / YsInterp
            self.scat = SsInterp

        elif type == "window":
            """
            Python adaptation of ``MaterialModel_Scene9Windows.m`` from the Hargreaves, Rendell, and Lam GRAS
            Scene 9 helper code. It combines two approaches:

            1) area purely resistive fit to the third-octave band absorption coefficient data provided with the 
               GRAS dataset.

            2) area reactive Mass-Spring-Damper model of the assumed fundamental resonance of the window panels. 
               This was included since such effects are well known be reactive, and this affects room modal frequencies. 
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
            csvData = pandas.read_csv(database_path() + "_csv" + os.sep + "mat_scene09_windows.csv", header=None).T
            fMeas = csvData[0]  # Third-octave band center frequencies
            aMeas = csvData[1]  # Third-octave band center absorption coefficients
            sMeas = csvData[2]  # Third-octave band center scattering coefficients

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
        self.matrix = {"material_model": {"type": type,
                                          "params": params}}

    def field_impedance(self, z):
        """
        Calculates field-incidence impedance for a set of angle-dependent impedances.

        This method averages surface admittance rather than absorption. Given angle-dependent impedance
        ``Zs(theta)``, it evaluates
        ``Y_f = integral((1 / Zs(theta)) sin(theta) dtheta) / integral(sin(theta) dtheta)``
        over ``self.incidence_angle`` and returns ``Z_f = 1 / Y_f``. This produces a scalar field-incidence
        impedance that can be passed through the usual reflection relation. It is mathematically distinct from
        statistical diffuse absorption, which averages ``alpha(theta)`` with ``sin(theta)*cos(theta)`` weighting.
        See Aretz, *Combined Wave And Ray Based Room Acoustic Simulations Of Small Rooms*, 2012, p. 86.

        Because this is a complex admittance average, the result is sensitive to the impedance reference plane.
        A lossless top air layer can alter the averaged field-incidence impedance and the derived absorption even
        though the angle-wise absorption coefficients are unchanged. Use ``diffuse_method='paris'`` when the desired
        output is statistical diffuse absorption.

        Parameters
        ----------
        z : array
            Multidimensional array with angle dependent set of impedances.

        Returns
        -------
        Field impedance array.
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

        ``srad`` is interpreted as the circular radiating area, so ``a = sqrt(srad/pi)`` and ``ka = k0*a``. The
        specific radiation impedance is ``Zrad/Z0 ~= 0.25*(ka)**2 + 1j*0.6133*ka``. This follows from the
        low-frequency unflanged-pipe relation summarized by Silva et al. (2009), Eqs. 2-3 and 10-11, with
        Levine and Schwinger's static end correction ``L/a = 0.6133`` under this package's positive-reactance
        convention.

        The returned load is divided by ``srad`` because the transfer matrices use volume velocity.
        """
        radius = np.sqrt(self.srad / np.pi)
        ka = self.k0 * radius
        zrad = self.z0 * (0.25 * ka ** 2 + 1j * 0.6133 * ka)
        return (zrad / self.srad).reshape((len(zrad), 1))

    def compute(self, rigid_backing=True, backing=None, conj=False, show_layers=True):
        """
        Calculates the global transfer matrix for the existing layers.

        The layer matrices are multiplied from incident side to termination side. Termination is selected with
        ``backing``: ``'rigid'`` for a hard wall, ``'air'`` for an oblique plane-wave air load, or
        ``'radiation'`` for a low-frequency unflanged circular-pipe radiation load.

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
        Finite backing loads use the loaded transfer-matrix relation
        ``Zs = S0*(A*ZL + B)/(C*ZL + D)``. ``backing='air'`` is the default no-rigid-wall termination when
        ``rigid_backing=False`` and is distinct from the analytical ``backing='radiation'`` option.

        If ``incidence='diffuse'`` and ``diffuse_method='field'`` are used with a top air layer, this method emits
        a ``RuntimeWarning`` because field-incidence impedance averaging is reference-plane sensitive. This warning
        does not apply to ``diffuse_method='paris'``, which averages angle-wise absorption coefficients.
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
        """Removes matrix data from self.matrix to reduce file size."""
        for matrix in self.matrix.keys():
            if "matrix" in list(self.matrix[matrix].keys()):
                self.matrix[matrix]["matrix"] = None

    def reduce_size(self):
        """Removes the value of some attributes to reduce file size."""
        self.clear_matrix()
        self._z_angle = None
        self._z_angle_angles = None

    def reduced_copy(self, keep_angles=(0.0,)):
        """
        Return a lightweight copy of this treatment for storage or transport.

        The copy carries the layer metadata needed by ``rebuild()`` but not the cached layer transfer
        matrices, and retains only the selected columns of ``z_angle``. Scalar results are untouched, so
        ``z``, ``alpha``, and ``z_norm`` stay exact for both diffuse methods; ``alpha_angle()`` stays exact
        for every retained angle. This object is not modified.

        Parameters
        ----------
        keep_angles : sequence of float or None, optional
            Incidence angles in degrees whose ``z_angle`` columns are retained, in the same units as the
            ``incidence_angle`` constructor argument. Each value must be one of ``self.incidence_angle``;
            retained columns are returned in ascending angle order. The default ``(0.0,)`` keeps normal
            incidence, which is the first angle of both default diffuse ranges. ``None`` discards
            ``z_angle`` entirely, matching ``reduce_size()``.

        Returns
        -------
        TMM
            A new treatment holding the reduced data.

        Raises
        ------
        ValueError
            If ``keep_angles`` is empty, or names an angle this treatment did not compute. Diffuse ranges
            that start above zero have no normal-incidence column, so they need an explicit ``keep_angles``.

        Notes
        -----
        For ``incidence='normal'`` and ``incidence='angle'`` there is a single angle, so that column always
        spans ``incidence_angle``, ``keep_angles`` is not consulted, and the copy is a fully functional
        treatment; the size reduction there comes from dropping the layer matrices alone.

        For ``incidence='diffuse'`` with a proper subset of angles the copy cannot reproduce quantities
        that integrate over the hemisphere, so ``field_impedance()``, ``diffuse_absorption_coefficient()``,
        and ``save2sheet(export_all=True)`` raise a descriptive error rather than returning a wrong result.
        Call ``rebuild()`` to restore the full angular data; it recomputes ``z_angle`` for every angle and
        clears the reduced marker.

        ``compute()`` cannot run directly on the copy because the layer matrices were dropped. Use
        ``rebuild()``, which reconstructs them from the layer metadata first.
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

    def rebuild(self):
        """Rebuild treatment layers to update frequency range."""
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
                                                layer=key)
                elif value["type"] == "slotted_panel_layer":
                    self.slotted_panel_layer(t=value["thickness [mm]"],
                                             w=value["slot_width [mm]"],
                                             s=value["slot_spacing [mm]"],
                                             rho=value["rho [kg/m3]"],
                                             method=value["method"],
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
            else:
                self.material_model(value["type"], params=value["params"])

        if matrix[list(matrix.keys())[-1]]["type"] == "backing":
            backing_data = matrix[list(matrix.keys())[-1]]
            if "diffuse_method" in backing_data:
                self._diffuse_method = current_diffuse_method if preserve_diffuse_method else backing_data["diffuse_method"]
            self.compute(rigid_backing=backing_data["rigid_backing"],
                         backing=backing_data.get("backing"),
                         conj=backing_data["impedance_conjugate"],
                         show_layers=False)
        else:
            self.compute(rigid_backing=False, show_layers=False)
        self._clear_stale_state()

    def log_rebuild(self):
        """Logs a list of commands calls needed to recreate the TMM object."""
        matrix = self.matrix.copy()
        optional_constructor_args = ""
        if self.x_scale != "lin":
            optional_constructor_args += f", x_scale={self.x_scale!r}"
        if self.display_name is not None:
            optional_constructor_args += f", display_name={self.display_name!r}"
        if self.color is not None:
            optional_constructor_args += f", color={self.color!r}"
        if self.s0 != 1.0:
            optional_constructor_args += f", s0={self.s0!r}"
        if self.srad != self.s0:
            optional_constructor_args += f", srad={self.srad!r}"
        logged_calls = [f"# {self.filename.capitalize()}",
                        f"{self.filename} = TMM("
                        f"fmin={self.fmin}, "
                        f"fmax={self.fmax}, "
                        f"df={self.df:0.2f}, "
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
                    end_correction_arg = ""
                    if value.get("method") in {"barrier", "eq_fluid"} and value.get("end_correction") is not None:
                        end_correction_arg = f"end_correction='{value['end_correction']}', "
                    logged_calls.append(
                        f"{self.filename}.perforated_panel_layer("
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"d={value['hole_diameter [mm]']:0.1f}, "
                        f"s={value['hole_spacing [mm]']:0.1f}, "
                        f"{end_correction_arg}"
                        f"rho={value['rho [kg/m3]']}, "
                        f"method='{value['method']}'"
                        f")"
                        )
                elif value["type"] == "slotted_panel_layer":
                    logged_calls.append(
                        f"{self.filename}.slotted_panel_layer("
                        f"t={value['thickness [mm]']:0.1f}, "
                        f"w={value['slot_width [mm]']:0.1f}, "
                        f"s={value['slot_spacing [mm]']:0.1f}, "
                        f"rho={value['rho [kg/m3]']}, "
                        f"method='{value['method']}'"
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
        """Prints the list of commands calls needed to recreate the TMM object."""
        logged_calls = self.log_rebuild()
        for call in logged_calls:
            print(call)

    @staticmethod
    def _is_report_number(value):
        """Return True for real scalar values that can be formatted numerically."""
        return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))

    def _layer_report_items(self, layer_data):
        """
        Return ordered, method-aware layer fields for text and spreadsheet reports.

        The stored ``self.matrix`` dictionaries intentionally keep all reconstruction metadata. This helper filters
        only the presentation layer so reports show parameters that are relevant to the selected formulation.
        """
        layer_type = layer_data.get("type")
        method = str(layer_data.get("method", "")).lower()
        model = str(layer_data.get("model", "")).lower()

        if layer_type == "porous_layer":
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
            if method in {"barrier", "eq_fluid"}:
                keys.append("end_correction")
            keys.append("rho [kg/m3]")
        elif layer_type == "slotted_panel_layer":
            keys = ["type", "method", "thickness [mm]", "slot_width [mm]", "slot_spacing [mm]", "open_area [%]",
                    "rho [kg/m3]"]
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
        Method to print each layer with its details.

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
            for key, value in self._layer_report_items(self.matrix[layer_key]):
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

    def filter_alpha(self, n_oct=1, view=True, show_table=False, **kwargs):
        """
        Filters the absorption coefficient into fractional octave bands. See tmm._plot.acoustic data for kwargs.

        Parameters
        ----------
        n_oct : int, optional
            Fractional octave bands that the absorption will be filtered to.
        view : bool, optional
            Boolean to display plot with filtered absorption.
        show_table : bool, optional
            Boolean to display the filtered values in a table.

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

        XLSX exports contain ``Data``, ``Bands`` and ``Setup`` sheets. The
        ``Setup`` sheet stores the TMM setup and the method-aware layer report.
        CSV exports keep numeric data and metadata separate: selected-method
        data are written to ``.csv`` and, when ``metadata=True``, the setup and
        layer report are written to ``*_metadata.csv``.

        The selected-method export follows the currently computed incidence
        settings. When ``incidence='diffuse'`` and ``diffuse_method='paris'``,
        impedance columns are omitted because Paris averaging returns diffuse
        absorption but does not define a unique diffuse complex impedance.
        ``export_all=True`` is a CSV-only diagnostic export that includes
        angle-wise impedance and absorption, field-incidence diffuse
        impedance/absorption, and Paris diffuse absorption.

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
            If True, write a sidecar metadata CSV. XLSX workbooks always include
            metadata in the ``Setup`` sheet.
        export_all : bool, optional
            If True, write a CSV-only diagnostic export containing angle-wise
            and diffuse quantities instead of only the selected-method data.
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

        The file is written to ``{project_folder}/Treatments/{filename}.h5``.
        The ``Treatments`` folder is created when needed. The checkpoint holds
        the reduced form produced by ``reduced_copy(keep_angles=None)``: cached
        layer transfer matrices and ``z_angle`` are left out, while the metadata
        ``rebuild()`` needs is retained. ``load()`` calls ``rebuild()``, which
        recomputes both.

        The HDF5 file is intended as a package-internal checkpoint for
        reopening and rebuilding a TMM object with a compatible version of this
        package. It is not a stable interchange format for external software.

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
            Input filename without the ``.h5`` extension. If
            ``{project_folder}/Treatments`` exists, the file is loaded from
            that folder. Otherwise, it is loaded directly from
            ``project_folder``. If no project folder was set, the current
            working directory is used.

        Notes
        -----
        Loading replaces the current object's saved attributes and then calls
        ``rebuild()`` to reconstruct reduced layer matrices and computed
        results. The receiver object's current ``project_folder`` is kept after
        loading so copied HDF5 files remain attached to the project from which
        they were loaded. The checkpoint must use layer methods and metadata
        supported by the current package version.
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

        This is a convenience wrapper around ``tmm._plot.acoustic_data``.
        Keyword arguments are forwarded to that function. If ``filename`` or
        ``project_folder`` are not supplied, this method uses the treatment's
        own ``filename`` and ``project_folder``.

        The cosmetic ``display_name`` and ``color`` attributes are used by the
        plotting backend. ``display_name`` is used as a human-facing title or
        legend label when available, while ``color`` sets the primary treatment
        curve color.

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
