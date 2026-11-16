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
import ast
import collections
import copy
import fractions
import functools
import keyword
import os
import warnings

import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy import special
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator
from scipy.signal import butter, freqz, savgol_filter
from scipy.special import jv

from tmm import _causality as causality
from tmm import _h5utils as h5utils
from tmm import _impedance_retrieval as retrieval
from tmm import _plot as plot
from tmm import _utils as utils
from tmm._reference_data import (  # noqa: F401  (re-exported for tmm.tmm imports)
    ALPHA_W_CLASSES, ALPHA_W_REFERENCE, ALPHA_W_THIRD_OCTAVE_INDICES, FABRIC_PARAMETERS, FABRIC_PRESETS,
    JEONG_ANGLE_WEIGHTS, JEONG_WEIGHT_ANGLES, JEONG_WEIGHT_BANDS, PLATE_PARAMETERS, PLATE_PRESETS, POROUS_PRESET_KINDS,
    POROUS_PRESETS, POROUS_VALIDITY, PRESET_PARAMETERS, PRESET_STATUSES, _FullTextFrame,
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

# Truncation of the end_correction="allard_atalla" series of perforated_panel_layer. The lattice indices are summed over
# the quarter disc m**2 + n**2 <= q_max**2, with q_max the larger of ALLARD_ATALLA_MIN_INDEX and the index at which the
# Bessel argument 2*pi*(R/D)*q reaches ALLARD_ATALLA_ARGUMENT, so the number of terms grows as (D/R)**2. The terms
# outside the disc are added as the plane integral of their large-argument form,
# (R/D)/(pi*X**2)*(1 - cos(2*X)/X) with X the argument at q_max. The number of terms grows as 1/open_area at small open
# areas, so the option is refused below ALLARD_ATALLA_MIN_OPEN_AREA, where "jb" approaches the same single-hole limit.
ALLARD_ATALLA_ARGUMENT = 20.0
ALLARD_ATALLA_MIN_INDEX = 40
ALLARD_ATALLA_MIN_OPEN_AREA = 1e-5

# Largest amount by which, at any frequency, the absorption from the field-incidence impedance may exceed the
# statistical (Paris) absorption over the same angles before compute() warns for diffuse_method="field". Both lie
# between 0 and 1, and the suggested value is 0.2 in absorption coefficient. A field result below the statistical one
# never warns. That level separates two kinds of stack. Cavities filled with porous material stay below it, while
# strongly resonant empty air cavities, whose admittance average is dominated by oblique-angle poles, exceed it. A
# lower value makes the check stricter, and a higher one makes it more lenient.
FIELD_EXCESS_LIMIT = 0.2

# Angle step [deg] above which the field-excess warning also names the step as a possible cause. Between two angles
# that far apart even a damped stack's impedance changes by more than a tenth, so the comparison can differ without
# an oblique-angle resonance.
COARSE_ANGLE_STEP = 5.0

# Largest root-mean-square difference between the input band absorption and the band means of the impedance that
# impedance_from_absorption() retrieves before it warns, for every method. It equals the root-mean-square form of eq. 36
# of Mondet et al. (2020), their rule for six octave bands of Sabine data with no value above 1 and fewer than three
# below 0.1. The validity verdict of a method="model" fit applies the rule its data meet (fit_validity_limit).
ABSORPTION_FIT_LIMIT = 0.05

# Largest median difference between the normal-incidence reflection factors of two equally good RMK+1 fits before
# impedance_from_absorption() reports that the band data do not fix the impedance.
ABSORPTION_FIT_SPREAD = 0.2

# Sample area [m²] of impedance_from_absorption(incidence="sabine") when sample_size is omitted: a square of the area
# Mondet et al. (2020), p. 2, assume for samples of unknown size, having stated that ISO 354 recommends 10 to 12 m².
SABINE_SAMPLE_AREA = 11.0

# Gauss-Legendre azimuth points on [0, pi/2] of the finite-sample forward model of incidence="sabine".
SABINE_AZIMUTH_POINTS = 8

# Points per octave at which method="real" inverts the finite-sample model for incidence="sabine".
SABINE_NODES_PER_OCTAVE = 24

# Largest absorption value impedance_from_absorption() accepts; larger values are taken for percentages.
ABSORPTION_PERCENT_LIMIT = 2.0

# Largest side [m] of a finite sample that finite_sample_absorption(), alpha_w(), impedance_from_absorption() and
# material_from_table() accept; larger sides are taken for millimetres, and the run time grows with the side squared.
SAMPLE_SIZE_LIMIT = 50.0

# Largest number of points of a frequency grid, derived or explicit; a larger grid is refused before it is built.
GRID_POINT_LIMIT = 1_000_000

# Kinds of data and methods of impedance_from_absorption(), and the fields of its termination record that show_layers()
# and save2sheet() report, in this order.
ABSORPTION_INCIDENCES = ("normal", "random", "sabine")
ABSORPTION_METHODS = ("real", "55deg", "model")
ABSORPTION_REPORT_KEYS = ("type", "incidence", "method", "absorber", "root", "sample_size [m]", "angle_limit [deg]",
                          "extrapolation", "bands [Hz]", "alpha", "parameters", "band_rms")

# Folder of the literature tables absorption.csv, scattering.csv and sources.csv.
LITERATURE_FOLDER = database_path() + "_literature" + os.sep

# Kind of data that material_from_table() passes to impedance_from_absorption() for each kind of absorption_table() row:
# Sabine data where the source states a reverberation-room measurement, at full or model scale, and random-incidence
# values for rows compiled without a stated method and for rows derived from hall reverberation.
LITERATURE_INCIDENCE = {"reverberation_room": "sabine", "reverberation_room_scale_model": "sabine",
                        "compiled": "random", "hall_derived": "random"}

# Band centres [Hz] of the absorption_table() and scattering_table() columns, and the a/lambda values of the scattering
# rows given against normalized frequency.
LITERATURE_ABSORPTION_BANDS = (125, 250, 500, 1000, 2000, 4000, 6000, 8000)
LITERATURE_SCATTERING_BANDS = (100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                               4000, 5000, 8000)
LITERATURE_SCATTERING_A_OVER_LAMBDA = ("0.125", "0.25", "0.5", "1", "2", "4", "8")

# Equivalent-fluid models built from the Johnson et al. dynamic tortuosity, and the frame options of porous_layer.
JCA_MODELS = ("jca", "jcal", "jcapl")
POROUS_FRAMES = ("rigid", "limp")

# Equivalent-fluid models of equivalent_fluid_model() and porous_layer(), and the branches of porous_facing_layer().
POROUS_MODELS = ("db", "miki", "qunli", "mechel_gf_lowX", "mechel_gf_highX", "mechel_rf_lowX", "mechel_rf_highX", "mac",
                 "garai_pompoli", "komatsu", "mechel_1976", "wilson_db", "wilson_2015", "mechel_grundmann",
                 *JCA_MODELS, "attenborough", "double_porosity")
FACING_METHODS = ("rebillard", "pieren", "pieren_free", "pieren_fixed")

# Types of material_model().
MATERIAL_MODEL_TYPES = ("floor", "ceiling", "door", "concrete", "plaster", "mdf", "window")

# Input that the impedance of each termination comes from, which the message of a NaN absorption names.
TERMINATION_ARGUMENTS = {"constant_z": "z, the impedance given", "material_model": "params, the panel parameters given",
                         "impedance_from_absorption": "bands and alpha, the absorption data given",
                         "material_from_table": "row, the table row given"}

# Window length [points] of the Savitzky-Golay smoothing of material_model(); the grid needs at least this many points.
MATERIAL_SMOOTHING_WINDOW = 31

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

# Preset fields each porous model reads besides sigma; porous_layer(preset=...) raises when the preset leaves one of
# them empty and the call does not pass it. Models not listed read sigma only. frame="limp" adds porosity and
# bulk_density.
POROUS_MODEL_NEEDS = {
    "mechel_1976": ("porosity",),
    "wilson_db": ("porosity",),
    "wilson_2015": ("porosity", "tortuosity"),
    "attenborough": ("porosity", "tortuosity"),
    "jca": ("porosity", "tortuosity", "viscous_length", "thermal_length"),
    "jcal": ("porosity", "tortuosity", "viscous_length", "thermal_length", "thermal_permeability"),
    "jcapl": ("porosity", "tortuosity", "viscous_length", "thermal_length", "thermal_permeability"),
    "double_porosity": ("porosity", "tortuosity", "viscous_length", "thermal_length", "thermal_permeability"),
}


def _preset_text(value, exact=False):
    """
    Return ``value`` as it appears in a preset override message: ``:g`` for real numbers, or with ``exact`` the shortest
    text that reads back as the same float, and ``repr`` otherwise. NumPy scalars and 0-d arrays count as numbers.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        return repr(float(value)) if exact else f"{value:g}"
    return repr(value)


def _preset_change(name, preset_value, value):
    """Return ``"name old -> new"`` for an override message, with exact numbers where ``:g`` would print both alike."""
    old, new = _preset_text(preset_value), _preset_text(value)
    if old == new:
        old, new = _preset_text(preset_value, exact=True), _preset_text(value, exact=True)
    return f"{name} {old} -> {new}"


def _literature_csv(name):
    """Read one literature table, indexed by ``id``, with empty text as ``""`` and empty numbers as ``NaN``."""
    frame = pandas.read_csv(LITERATURE_FOLDER + name, dtype=str, keep_default_na=False, encoding="utf-8")
    numeric = {"thickness_mm", "mass_kg_m2", "density_kg_m3", "airspace_mm", "depth_mm", "period_mm", "width_mm",
               "h_over_a"}
    for column in frame.columns:
        if column.startswith(("a_", "s_")) or column in numeric:
            # where() rather than replace("", NaN), which relies on a downcast that pandas 2.2 deprecates.
            frame[column] = pandas.to_numeric(frame[column].where(frame[column] != ""))
    return frame.set_index("id")


def _with_citations(frame):
    """
    Return ``frame`` with ``citation``, the citation of ``source_id``, and ``primary_citation``, those of the
    ``primary_ref`` entries joined by ``" | "``, read from ``sources.csv`` and placed after ``primary_ref``.
    """
    sources = pandas.read_csv(LITERATURE_FOLDER + "sources.csv", dtype=str, keep_default_na=False, encoding="utf-8")
    citations = sources.set_index("source_id")["citation"]
    frame = frame.copy()
    position = frame.columns.get_loc("primary_ref") + 1
    frame.insert(position, "citation", frame["source_id"].map(citations).fillna(""))
    frame.insert(position + 1, "primary_citation", [" | ".join(citations.get(ref, "") for ref in refs.split(";") if ref)
                                                    for refs in frame["primary_ref"]])
    return frame


def _filter_rows(frame, filters, search, search_columns, valid=None):
    """
    Return the rows of ``frame`` whose value in each column named in ``filters`` is one of the wanted values given
    there, one text or a list of texts, and whose index or one of whose ``search_columns`` contains ``search`` as plain
    text, ignoring case. ``None`` skips a filter, and a missing text value matches nothing. A column named in ``valid``
    accepts the names given there, also those that no row holds, and any other column accepts the values of its rows.
    """
    keep = None
    valid = {} if valid is None else valid
    for column, wanted in filters.items():
        if wanted is None:
            continue
        available = sorted(set(frame[column]))
        names = valid.get(column, available)
        listed = (f"Valid names: {list(names)}. Rows exist for: {available}." if column in valid
                  else f"Available: {available}.")
        if isinstance(wanted, str):
            wanted = [wanted]
        elif isinstance(wanted, (bytes, bytearray)) or not np.iterable(wanted):
            raise ValueError(f"Unknown {column} {utils.describe_value(wanted)}, which is not a text or a list of texts."
                             f" {listed}")
        else:
            wanted = list(wanted)
        unknown = sorted({str(item) for item in wanted if isinstance(item, str) and item not in names})
        unknown += [item for item in wanted if not isinstance(item, str)]
        if unknown:
            raise ValueError(f"Unknown {column} {unknown}. {listed}")
        found = frame[column].isin(wanted).to_numpy()
        keep = found if keep is None else keep & found
    if keep is not None:
        frame = frame[keep]
    if search is not None:
        needle = str(search)
        found = frame.index.to_series().astype(str).str.contains(needle, case=False, regex=False)
        for name in search_columns:
            found |= frame[name].fillna("").astype(str).str.contains(needle, case=False, regex=False).to_numpy()
        frame = frame[found.to_numpy()]
    return _FullTextFrame(frame)


def _sort_rows(frame, sort_by):
    """
    Return ``frame`` sorted ascending by the columns named in ``sort_by``, one text or a list of texts, where the index
    name stands for the index. Rows without a value come last, and rows that tie keep their order. ``None`` returns
    ``frame`` unchanged.
    """
    if sort_by is None:
        return frame
    available = [frame.index.name, *frame.columns]
    if isinstance(sort_by, (set, frozenset, dict)):
        raise ValueError(f"Unknown sort_by {utils.describe_value(sort_by)}, which is not a text or a list of texts; "
                         f"pass the column names as a list, in the order to sort by. Available: {available}.")
    names = ([sort_by] if isinstance(sort_by, str)
             else list(sort_by) if np.iterable(sort_by) and not isinstance(sort_by, (bytes, bytearray)) else [])
    if not names:
        raise ValueError(f"Unknown sort_by {utils.describe_value(sort_by)}, which is not a text or a non-empty list of "
                         f"texts. Available: {available}.")
    unknown = [name for name in names if not isinstance(name, str) or name not in available]
    if unknown:
        raise ValueError(f"Unknown sort_by {unknown}. Available: {available}.")
    # reset_index() numbers the rows from 0, so the sorted labels are row positions for iloc.
    order = frame.reset_index().sort_values(names, kind="stable", na_position="last").index
    return _FullTextFrame(frame.iloc[order])


def _literature_row(frame, row, lister, argument, other_csv):
    """
    Return one row of a literature table by id, or raise a ``ValueError`` that names the lister. An id of the other
    table, ``other_csv``, passed as ``argument`` is named as such.
    """
    if not isinstance(row, str):
        raise ValueError(f"{argument} takes the id of a {lister}() row as text, not {utils.describe_value(row)}.")
    if row not in frame.index:
        if row in _literature_csv(other_csv).index:
            other, wanted = (("a scattering_table", "an absorption_table") if lister == "absorption_table"
                             else ("an absorption_table", "a scattering_table"))
            raise ValueError(f"{row!r} is {other}() row; {argument} takes the id of {wanted}() row.")
        raise ValueError(f"No literature row {row!r}. Use TMM.{lister}(search=...) to find an id.")
    return frame.loc[row]


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
            Minimum frequency of interest [Hz]. Must be positive and finite, since the models are not defined at 0 Hz
            or below.
        fmax : float, optional
            Maximum frequency of interest [Hz]. Must be positive, finite and not below ``fmin``. An explicit ``freq``
            takes precedence, and the order is checked again when ``freq = None`` returns to the derived grid.
        df : float, optional
            Grid parameter above 0 that sets how many points the grid has, as ``int((fmax - fmin) / df) + 1``. On a
            linear ``x_scale`` that is the frequency resolution in Hz. On a logarithmic one it only sets the point
            count: the spacing is geometric, running from 0.02 Hz to 5.5 Hz across the default 20 Hz to 5 kHz band. It
            does not apply while an explicit ``freq`` vector is set, so ``None`` is accepted with one and stands for 1.
            Another value, or a grid of more than ``GRID_POINT_LIMIT`` points, raises a ``ValueError``, here and in the
            setter.
        incidence : string, optional
            ``'normal'`` for normal incidence, ``'angle'`` for a single oblique angle, or ``'diffuse'`` for a
            diffuse-field approximation.
        incidence_angle : list of float, optional
            Minimum angle, maximum angle and step, in degrees, as real numbers. ``incidence='angle'`` uses the first
            value.
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
            field-incidence impedance and takes ``alpha`` from it at normal incidence, see ``field_impedance()``.
            ``'paris'`` averages angle-wise absorption into a statistical diffuse absorption coefficient, see
            ``diffuse_absorption_coefficient()``.
        s0 : float, optional
            Front/reference area in square meters used by the volume-velocity transfer matrices, above 0. The default
            value of 1.0 keeps impedances in specific-acoustic-impedance form.
        srad : float, optional
            Rear termination area in square meters, above 0. For ``backing='radiation'`` this is interpreted as the area
            of the circular radiating aperture. If omitted, ``srad`` defaults to ``s0``.
        freq : array_like, optional
            Explicit frequency vector [Hz], a one-dimensional vector of real numbers, positive, finite and strictly
            ascending, with at most ``GRID_POINT_LIMIT`` points. A single number gives a grid of one frequency. When
            provided it owns the grid: ``fmin``/``fmax`` report its endpoints while ``df`` and ``x_scale`` report
            ``None``. The spacing may be non-uniform because the TMM evaluates every frequency independently. When
            omitted, the grid built from ``fmin``/``fmax``/``df`` is used. The four derived-grid parameters are retained
            either way, so assigning ``freq = None`` later restores the grid this constructor described.
        """
        diffuse_method = self._validate_diffuse_method(diffuse_method, "TMM")
        incidence = self._validate_incidence(incidence)
        if incidence_angle is None:
            incidence_angle = self._default_diffuse_incidence_angle(diffuse_method)
        incidence_angle = self._validate_incidence_angle(incidence_angle, incidence)
        if srad is None:
            srad = s0

        # df does not apply while an explicit vector owns the grid, so None is taken there as the default 1.
        self._df_from_default = df is None and freq is not None
        df = 1 if self._df_from_default else df
        self._df = utils.check_positive("TMM", "df", "the grid density parameter [Hz]", df)
        self._fmin = self._validate_frequency_limit(fmin, "fmin")
        self._fmax = self._validate_frequency_limit(fmax, "fmax")
        if freq is None and float(fmax) < float(fmin):
            raise ValueError(f"fmax ({fmax!r} Hz) must not be below fmin ({fmin!r} Hz).")
        if freq is None:
            self._check_grid_size("TMM", self._fmin, self._fmax, self._df)
        self._freq = None if freq is None else self._as_frequency_vector(freq)
        self._s0 = utils.check_positive("TMM", "s0", "the front reference area [m²]", s0)
        self._srad = utils.check_positive("TMM", "srad", "the rear termination area [m²]", srad)
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
            f"{reason} Stored transfer matrices are now stale. Call {self._stale_remedy()} before using computed "
            f"results.",
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

    def _stale_remedy(self):
        """Return the calls that bring stale layer matrices back: ``compute()`` also does it on a termination."""
        return "rebuild()" if self._termination_name() is None else "compute() or rebuild()"

    def _stale_details(self):
        """Return the stored stale reasons as a sentence for an error message, or ``""`` when there are none."""
        reasons = "; ".join(reason.rstrip(".") for reason in getattr(self, "_stale_reasons", []))
        return f" Reasons: {reasons}." if reasons else ""

    def _raise_if_stale_results(self, operation):
        """Raise when an operation would use stale layer matrices or computed results."""
        if getattr(self, "_layers_stale", False):
            raise RuntimeError(f"{operation} cannot use stale layer matrices. Call {self._stale_remedy()} first."
                               f"{self._stale_details()}")
        if getattr(self, "_results_stale", False):
            raise RuntimeError(f"{operation} cannot use stale computed results. Call compute() or rebuild() first."
                               f"{self._stale_details()}")

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
    def _validate_frequency_limit(value, name, method="TMM"):
        """Return ``value`` unchanged if it is one positive, finite number of hertz; the models are not defined at 0 Hz
        or below. Keeping the given type keeps ``log_rebuild()`` output as the user wrote it and the grid as before. A
        ``Fraction`` or float16 value is returned as a float, since NumPy cannot round a grid of fractions and float16
        holds too few digits for the grid points."""
        what = "the lowest frequency of the grid [Hz]" if name == "fmin" else "the highest frequency of the grid [Hz]"
        number = utils.check_positive(method, name, what, value, " The models are not defined at 0 Hz or below.")
        if isinstance(value, fractions.Fraction) or getattr(value, "dtype", None) == np.float16:
            return number
        return value

    @staticmethod
    def _check_grid_size(method, fmin, fmax, df, df_from_default=False):
        """Raise a ``ValueError`` when the grid of ``fmin``, ``fmax`` and ``df`` has more than ``GRID_POINT_LIMIT``
        points, without building it. The point count is that of the ``freq`` property, ``int(steps) + 1``. With
        ``df_from_default`` the message says that ``df`` is the default that ``df=None`` stood for."""
        with np.errstate(over="ignore"):
            steps = float((fmax - fmin) / df)
        points = int(steps) + 1 if steps < 1e12 else None
        if points is None or points > GRID_POINT_LIMIT:
            if not np.isfinite(steps):  # The arithmetic of the inputs overflowed; float64 may still give the count.
                with np.errstate(over="ignore"):
                    steps = float((np.float64(fmax) - np.float64(fmin)) / np.float64(df))
            count = (f"{points:,}" if points is not None else f"about {steps:.3g}" if np.isfinite(steps)
                     else "an unbounded number of")
            default = ", the default that df took when None was passed with freq," if df_from_default else ""
            raise ValueError(f"{method}: the frequency grid from fmin = {fmin!r} Hz to fmax = {fmax!r} Hz with "
                             f"df = {df!r}{default} has {count} points, more than GRID_POINT_LIMIT = "
                             f"{GRID_POINT_LIMIT:,}. Pass a larger df or a narrower range.")

    @staticmethod
    def _as_frequency_vector(freq, method="TMM"):
        """Validate an explicit frequency vector (Hz): a non-empty one-dimensional vector of real numbers, positive,
        finite and strictly ascending, with at most ``GRID_POINT_LIMIT`` points. Spacing may be non-uniform. TMM
        evaluates each frequency independently. A single number counts as a vector of one frequency."""
        try:
            array = utils.real_array(freq)
        except (TypeError, ValueError):  # A nested sequence of unequal lengths.
            array = np.asarray([], dtype=object)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim != 1 or array.dtype.kind not in "iuf" or utils.holds_boolean(freq):
            raise ValueError(f"{method}: freq, the explicit frequency vector [Hz], must be a one-dimensional vector of "
                             f"real numbers, not {utils.describe_value(freq)}.")
        if array.size > GRID_POINT_LIMIT:
            raise ValueError(f"{method}: freq, the explicit frequency vector [Hz], has {array.size:,} points, more "
                             f"than "
                             f"GRID_POINT_LIMIT = {GRID_POINT_LIMIT:,}.")
        freq = array.astype(float)
        if freq.size == 0:
            raise ValueError("freq must contain at least one frequency in Hz.")
        if not np.all(np.isfinite(freq)) or np.any(freq <= 0):
            raise ValueError("freq must contain only positive, finite frequencies in Hz: the models are not defined at "
                             "0 Hz or below.")
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
    def _check_angles(method, angles):
        """
        Return ``angles`` as a float array and the order that sorts it, and raise a ``ValueError`` that names the first
        angle out of range unless they are a one-dimensional sequence of distinct finite numbers of degrees from 0 to
        90.
        """
        what = "the incidence angles [deg]"
        try:
            array = np.asarray(angles)
        except (TypeError, ValueError):  # A nested sequence of unequal lengths.
            array = np.asarray([], dtype=object)
        if array.dtype.kind not in "iuf" or array.ndim != 1 or array.size == 0 or utils.holds_boolean(angles):
            raise ValueError(f"{method}: angles, {what}, must be a one-dimensional sequence of finite numbers from 0 "
                             f"to "
                             f"90, not {utils.describe_value(angles)}.")
        array = array.astype(float)
        outside = ~(np.isfinite(array) & (array >= 0) & (array <= 90))
        if np.any(outside):
            position = int(np.argmax(outside))
            raise ValueError(f"{method}: angles, {what}, must be finite numbers from 0 to 90, not "
                             f"{utils.describe_value(angles)} with {utils.describe_value(array[position])} at position "
                             f"{position}.")
        order = np.argsort(array, kind="stable")
        repeated = np.flatnonzero(np.diff(array[order]) == 0)
        if repeated.size:
            raise ValueError(f"{method}: angles, {what}, must be distinct, not {utils.describe_value(angles)} with "
                             f"{utils.describe_value(array[order][repeated[0]])} more than once.")
        return array, order

    @staticmethod
    def _validate_incidence(value):
        """Return a supported incidence mode."""
        if not isinstance(value, str) or value not in {"normal", "angle", "diffuse"}:
            raise ValueError(f"incidence must be 'normal', 'angle', or 'diffuse', not {utils.describe_value(value)}.")
        return value

    @staticmethod
    def _validate_diffuse_method(value, method):
        """Return a supported diffuse method, ``'field'`` or ``'paris'``."""
        if not isinstance(value, str) or value not in {"paris", "field"}:
            raise ValueError(f"{method}: diffuse_method must be 'paris' or 'field', not {utils.describe_value(value)}.")
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
        if not isinstance(value, str) or value not in {"lin", "log"}:
            raise ValueError(f"x_scale must be 'lin' or 'log', not {utils.describe_value(value)}.")
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
        shape = utils._shape(value)
        shown = (repr([utils.plain_scalar(item) for item in value]) if len(shape) == 1 and shape[0] <= 6
                 else utils.describe_value(value))
        try:
            values = utils.real_array(value)
        except (TypeError, ValueError):  # A nested sequence of unequal lengths.
            values = np.asarray([], dtype=object)
        if (values.dtype.kind not in "iuf" or values.size == 0 or utils.holds_boolean(value)
                or not np.all(np.isfinite(values))):
            raise ValueError(f"TMM: incidence_angle, the incidence angles [deg], must hold finite real numbers, not "
                             f"{shown}.")
        values = values.astype(float).reshape(-1)

        def refuse(rule):
            raise ValueError(f"TMM: incidence_angle, the incidence angles [deg], must be {rule}, not {shown}.")

        if incidence == "diffuse":
            if values.size != 3:
                refuse("[start, stop, step] for diffuse incidence")
            start, stop, step = values
            if step <= 0:
                refuse("[start, stop, step] with a step above 0 for diffuse incidence")
            if stop <= start:
                refuse("[start, stop, step] with stop above start for diffuse incidence")
            if start < 0 or stop > 90:
                refuse("[start, stop, step] within 0 to 90 degrees for diffuse incidence")
            if int(np.round((stop - start) / step)) < 1:
                refuse("[start, stop, step] with a step that leaves at least two angles between start and stop")
        elif incidence == "angle":
            angle = values[0]
            if angle < 0 or angle >= 90:
                refuse("an angle of at least 0 and below 90 degrees for angle incidence")

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
        new_fmin = self._validate_frequency_limit(new_fmin, "fmin", "TMM.fmin")
        self._check_grid_size("TMM.fmin", new_fmin, self._fmax, self._df,
                              bool(getattr(self, "_df_from_default", False)))
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
        new_fmax = self._validate_frequency_limit(new_fmax, "fmax", "TMM.fmax")
        self._check_grid_size("TMM.fmax", self._fmin, new_fmax, self._df,
                              bool(getattr(self, "_df_from_default", False)))
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
        new_df = utils.check_positive("TMM.df", "df", "the grid density parameter [Hz]", new_df)
        self._check_grid_size("TMM.df", self._fmin, self._fmax, new_df)
        self._df = new_df
        self._df_from_default = False
        self._mark_layers_stale("df was changed after model data existed.")

    @property
    def freq(self):
        """Return frequency values."""
        if self._freq is not None:
            return self._freq

        if self._fmin > self._fmax:
            raise ValueError(f"fmin ({self._fmin!r} Hz) is above fmax ({self._fmax!r} Hz). Set fmax first when moving "
                             f"the range up, and fmin first when moving it down.")
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
        if new_freq is None:
            self._check_grid_size("TMM.freq", self._fmin, self._fmax, self._df,
                                  bool(getattr(self, "_df_from_default", False)))
        self._freq = None if new_freq is None else self._as_frequency_vector(new_freq, "TMM.freq")
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
        self._s0 = utils.check_positive("TMM.s0", "s0", "the front reference area [m²]", new_s0)
        self._mark_layers_stale("s0 was changed after model data existed.")

    @property
    def srad(self):
        """Return device rear termination area."""
        return self._srad

    @srad.setter
    def srad(self, new_srad):
        """Set device rear termination area."""
        self._srad = utils.check_positive("TMM.srad", "srad", "the rear termination area [m²]", new_srad)
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
        self._diffuse_method = self._validate_diffuse_method(method, "TMM.diffuse_method")
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
        spline-interpolated onto ``self.freq``, and ``material_from_table()`` from a row of ``scattering_table()``.
        It can also be assigned directly, for example from a value measured or computed elsewhere.
        """
        return self._scat

    @scat.setter
    def scat(self, new_scat):
        """Set the scattering coefficient on ``self.freq``."""
        self._scat = new_scat

    @property
    def z_norm(self):
        """Return normalized surface impedance."""
        return self._normalized_impedance(self.z, self.z0)

    @staticmethod
    def _normalized_impedance(z, z0):
        """Return ``z / z0``, keeping an infinite impedance as ``inf + 0j`` instead of the ``inf + nan*j`` of complex
        division."""
        z = np.asarray(z)
        with np.errstate(invalid="ignore"):
            return np.where(np.isinf(z), np.inf, z / z0)

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
        ``r(theta) = (Zs*cos(theta) - Z0) / (Zs*cos(theta) + Z0)`` and ``alpha(theta) = 1 - abs(r(theta))**2``. An
        infinite ``Zs`` gives ``r = 1`` and ``alpha = 0``.

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

        zs = np.asarray(zs)
        with np.errstate(invalid="ignore"):
            r = (zs * cos_theta - self.z0) / (zs * cos_theta + self.z0)
        r = np.where(np.isinf(zs), 1, r)
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
        ``field_impedance()`` when a scalar field-incidence impedance is needed. ``angles`` [deg] defaults to
        ``incidence_angle``; given angles must be a one-dimensional sequence of distinct finite numbers from 0 to 90, or
        a ``ValueError`` is raised. They may come in any order, with the ``z_angle`` columns in the same order.
        """
        order = None
        if angles is not None:
            angles, order = self._check_angles("diffuse_absorption_coefficient", angles)
        if z_angle is None:
            self._raise_if_partial_z_angle("diffuse_absorption_coefficient()")
            z_angle = self.z_angle
        if angles is None:
            angles = self.incidence_angle

        angles = np.asarray(angles, dtype=float)
        z_angle = np.asarray(z_angle)
        if z_angle.ndim != 2 or z_angle.shape[1] != len(angles):
            raise ValueError(
                f"diffuse_absorption_coefficient: z_angle must have one column per angle of angles: got shape "
                f"{z_angle.shape} for {len(angles)} angles."
            )
        if order is not None:
            angles, z_angle = angles[order], z_angle[:, order]
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
                raise ValueError(f"angle_weights preset must be one of {sorted(JEONG_ANGLE_WEIGHTS)}, not "
                                 f"{utils.describe_value(angle_weights)}.")
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

    @staticmethod
    def _ordered_angle_weights(angle_weights, order):
        """
        Return ``angle_weights`` with its angle axis put in the ``order`` that sorts the angles it was given for. A
        preset name, a callable and ``None`` follow the sorted angles by themselves, and an array of another shape is
        returned as it is, for ``_angle_weights`` to refuse.
        """
        if angle_weights is None or isinstance(angle_weights, str) or callable(angle_weights):
            return angle_weights
        try:
            weights = np.asarray(angle_weights, dtype=float)
        except (TypeError, ValueError):
            return angle_weights
        if weights.ndim in (1, 2) and weights.shape[-1] == len(order):
            return weights[..., order]
        return angle_weights

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
            Sample side lengths ``(a, b)`` [m]. A single value gives a square. A side above ``SAMPLE_SIZE_LIMIT`` raises
            a ``ValueError``, since it is taken for a side in millimetres.
        z_angle : array or None, optional
            Angle-dependent surface impedance with one column per angle [Pa*s/m]. If ``None``, ``self.z_angle`` is used.
        angles : array or None, optional
            Incidence angles of the ``z_angle`` columns [deg], distinct, finite and from 0 to 90, in any order. The
            method sorts them together with the ``z_angle`` columns and an ``angle_weights`` array. If ``None``,
            ``self.incidence_angle`` is used. The upper limit sets the integration range, for example 90 or 78 degrees.
            For the termination of ``impedance_from_absorption()`` or ``material_from_table()`` on a diffuse object
            without ``z_angle``, the method takes ``self.z`` at every angle, and without ``angles`` it uses the angles
            of the Sabine forward model of that termination, 0 to the ``angle_limit`` of its record at steps of at most
            1 degree, or 0 to 90 degrees for data other than Sabine data. The result does not depend on
            ``diffuse_method``, and a Sabine retrieval reproduces its own band data.
        phi_points : int, optional
            Number of Gauss-Legendre points in azimuth on ``[0, pi/2]``, a whole number of at least 1. ``angles`` or
            ``phi_points`` outside these ranges raise a ``ValueError``.
        normalization : str, optional
            Reference power for the absorption coefficient: ``"incident"``, ``"true"``, or ``"available"``.
        angle_weights : None, str, array, or callable, optional
            Angular weighting of the incident intensity. ``None`` gives uniform weights. A string selects a preset:
            ``"rectangular_floor"``, ``"irregular_floor"``, or ``"irregular_wall"``. An array has one weight per angle,
            shape ``(len(angles),)``, or per frequency and angle, shape ``(len(freq), len(angles))``. A callable
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
        sides = utils.check_sample_size("finite_sample_absorption", sample_size, SAMPLE_SIZE_LIMIT)
        utils.check_number("finite_sample_absorption", "phi_points", "the number of azimuth points", phi_points,
                           "a whole number of at least 1", lambda x: x >= 1 and float(x).is_integer())
        order = None
        if angles is not None:
            angles, order = self._check_angles("finite_sample_absorption", angles)
        if not isinstance(normalization, str) or normalization not in {"incident", "true", "available"}:
            raise ValueError(f"finite_sample_absorption: normalization must be 'incident', 'true', or 'available', not "
                             f"{utils.describe_value(normalization)}.")
        if z_angle is None:
            self._raise_if_partial_z_angle("finite_sample_absorption()")
            if self._has_paris_termination_alpha():
                # A locally reacting termination has the same impedance at every angle, and by default the angles of the
                # forward model of impedance_from_absorption(), whatever incidence_angle says.
                if angles is None:
                    angles = self._termination_angles()
                z_angle = np.repeat(self.z[:, None], np.size(angles), axis=1)
            else:
                z_angle = self.z_angle
        if angles is None:
            angles = self.incidence_angle
        angles = np.asarray(angles, dtype=float)
        z_angle = np.asarray(z_angle)
        if z_angle.ndim != 2 or z_angle.shape != (len(self.freq), len(angles)) or len(angles) < 2:
            raise ValueError(
                f"finite_sample_absorption: z_angle must have one row per frequency and one column per angle of "
                f"angles, "
                f"with at least two angles: got shape {z_angle.shape} for {len(self.freq)} frequencies and "
                f"{len(angles)} angles."
            )
        if order is not None:
            # Angles in another order give the result of the same angles in ascending order.
            angles, z_angle = angles[order], z_angle[:, order]
            angle_weights = self._ordered_angle_weights(angle_weights, order)

        a, b = sides
        theta = np.deg2rad(angles)
        nodes, weights = np.polynomial.legendre.leggauss(int(phi_points))
        phi = np.pi / 4 * (nodes + 1)
        phi_weights = weights / 2  # Mean over [0, pi/2].
        z_f = self._field_impedance_on_frequency_grid(float(a), float(b), tuple(float(k) for k in self.w0 / self.c0),
                                                      tuple(float(t) for t in theta), tuple(float(p) for p in phi))
        zeta = self._normalized_impedance(z_angle, self.z0)[:, :, None]
        weighted_sin = self._angle_weights(angle_weights, angles) * np.sin(theta)[None, :]

        def angular_integral(values):
            return self._trapezoidal_integral((values @ phi_weights) * weighted_sin, theta, axis=1)

        with np.errstate(invalid="ignore"):
            absorbed = angular_integral(np.where(np.isinf(zeta), 0, 4 * zeta.real / np.abs(zeta + z_f) ** 2))
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
            treatment computed with ``incidence="diffuse"``. A side above ``SAMPLE_SIZE_LIMIT`` raises a ``ValueError``,
            since it is taken for a side in millimetres.

        Returns
        -------
        dict
            ``alpha_w`` (float), ``shape`` (the indicator letters, empty when there are none), ``label`` (for example
            ``"0.70(H)"``), ``class`` (``"A"`` to ``"E"`` or ``"not classified"``), ``practical`` (``alpha_p`` at 250 Hz
            to 4 kHz), ``reference`` (the shifted reference curve at the same bands), ``third_octave`` (the band means
            at 200 Hz to 5 kHz), and ``incidence`` (the absorption rated: ``"finite sample a x b m"``,
            ``"finite sample a x b m (0 to <limit> deg)"``, ``"diffuse (field)"``, ``"diffuse (paris)"``,
            ``"diffuse (paris, 0 to 90 deg)"``, ``"normal"``, or ``"angle <theta> deg"``).

        Raises
        ------
        RuntimeError
            If the treatment has not been computed.
        ValueError
            If the frequency grid does not cover the 200 Hz to 5 kHz third-octave bands or leaves one of them empty, if
            the absorption is not finite in a band, if ``sample_size`` is given without diffuse incidence, or if it has
            a side above ``SAMPLE_SIZE_LIMIT`` [m].

        Notes
        -----
        ISO 11654 rates reverberation-room measurements made to ISO 354 on a finite sample. A finite sample absorbs more
        than the infinite one through its edges, and ``sample_size`` rates the absorption of a finite sample of that
        size. Without it, the method rates ``self.alpha`` as the last ``compute()`` left it: with
        ``incidence="diffuse"`` the absorption of the field-incidence impedance (``diffuse_method="field"``, the
        default) or the Paris average (``diffuse_method="paris"``), with ``incidence="normal"`` the normal-incidence
        absorption, and with ``incidence="angle"`` the absorption at that angle. For the termination of
        ``impedance_from_absorption()`` or ``material_from_table()`` on a diffuse object it is the Paris absorption over
        0 to 90 degrees, whatever ``diffuse_method`` says, and ``sample_size`` rates the finite-sample absorption over 0
        to the ``angle_limit`` of a Sabine retrieval and over 0 to 90 degrees otherwise. The ``incidence`` entry of the
        result names the case, and for such a termination with an ``angle_limit`` below 90 degrees it adds that range to
        the finite-sample name. The standard considers the rating not appropriate below 250 Hz ([1], clause 1.2).

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
            if self._has_paris_termination_alpha():
                rated = "diffuse (paris, 0 to 90 deg)"
            elif self.incidence == "diffuse":
                rated = f"diffuse ({self.diffuse_method})"
            elif self.incidence == "angle":
                rated = f"angle {float(self.incidence_angle[0]):g} deg"
            else:
                rated = "normal"
        else:
            sides = utils.check_sample_size("alpha_w", sample_size, SAMPLE_SIZE_LIMIT)
            if self.incidence != "diffuse":
                raise ValueError("alpha_w(sample_size=...) needs a treatment computed with incidence='diffuse'.")
            alpha = self.finite_sample_absorption(sides)
            rated = "finite sample " + " x ".join(f"{side:g}" for side in sides) + " m"
            angles = self._termination_angles()
            if angles is not None and angles[-1] < np.nextafter(90.0, 0.0):
                # Six significant digits unless they round the limit, as they round 89.9999999 to 90.
                limit = float(angles[-1])
                rated += f" (0 to {limit:g} deg)" if float(f"{limit:g}") == limit else f" (0 to {limit!r} deg)"
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
    def porous_presets(search=None, status=None, model=None, kind=None, sort_by=None):
        """
        Return the ``porous_layer(preset=...)`` parameter sets as a table.

        Each row holds the parameters in ``porous_layer()`` units, then ``model``, the model ``porous_layer()`` uses for
        the row when the call gives none, and ``kind``, which is ``"jca"`` or ``"jcal"`` for complete sets and
        ``"sigma"`` for sets that give the flow resistivity and at most the bulk density, porosity and tortuosity.
        ``status``, ``product`` and ``manufacturer`` follow, and the last two columns are the ``source`` by author and
        year with the table, figure or section and the printed page, and a ``description`` of the material and of how
        the source obtained or used the values. Where the source names no product or manufacturer or gives no value of a
        parameter (``None`` in the preset), the cell is empty in the display and NaN in the data. Rows are ordered by
        flow resistivity.

        ``status`` says how the acoustic values of a row were obtained. The acoustic values are the flow resistivity or
        resistance, porosity, tortuosity, characteristic lengths and thermal permeability, and for plates the density
        and the elastic constants. A secondary value whose origin the source does not state, such as a thickness,
        surface mass or bulk density, is named in the description and does not change the status. A row is
        ``"measured"`` when its acoustic values were measured on the sample, by direct or ultrasonic methods. It is
        ``"derived"`` when none of its acoustic values was fitted and at least one was computed by arithmetic from
        values the source prints, and ``"fitted"`` when at least one comes from indirect characterization, that is, was
        extracted by inverting an absorber model against measured surface impedance, absorption, or dynamic density and
        bulk modulus. ``"manufacturer"`` marks values that come from the producer of the material, as the source states.
        ``"nominal"`` marks representative values for a kind of material rather than for one sample, such as handbook
        tables of typical material constants and values a source calls nominal. ``"unstated"`` marks a row whose source
        does not say how the acoustic values of the material or product were obtained, also when the package computes
        one of them by arithmetic from values of that unstated origin, which does not make the row ``"derived"``.
        ``"range"`` marks ``plasterboard_800`` of ``plate_presets()``, the one row that takes the lower ends of printed
        ranges.

        Parameters
        ----------
        search : str, optional
            Keep only rows whose preset name, product, manufacturer, source or description contains this text, ignoring
            case, for example ``"glass_wool"`` or ``"Kino"``.
        status : str or list of str, optional
            Keep only these statuses, for example ``"measured"`` or ``["measured", "fitted"]``. A status defined above
            that no row holds, or an empty list, gives an empty table, and a value that is not one of the seven raises
            ``ValueError``.
        model : str or list of str, optional
            Keep only rows with this default model, for example ``"jca"`` or ``["miki", "komatsu"]``. A model of
            ``porous_layer()`` that no row holds, such as ``"db"``, or an empty list, gives an empty table, and any
            other value raises ``ValueError``.
        kind : str or list of str, optional
            Keep only these kinds, out of ``"jca"``, ``"jcal"`` and ``"sigma"``, for example ``["jca", "jcal"]``. A kind
            that no row holds, or an empty list, gives an empty table, and any other value raises ``ValueError``.
        sort_by : str or list of str, optional
            Sort the rows ascending by this column or these columns, where ``"preset"`` is the preset name, for example
            ``"model"`` or ``["status", "sigma"]``. Text sorts by character code, so ``"foam_10.9k"`` comes before
            ``"foam_2.3k"`` and capitals come before lower case. Rows without a value come last, and rows that tie keep
            the order by flow resistivity.

        Returns
        -------
        pandas.DataFrame
            One row per preset, indexed by preset name. The table displays every row and column and its text columns in
            full; the global pandas display options are left unchanged.
        """
        columns = [*PRESET_PARAMETERS, "model", "kind", "status", "product", "manufacturer", "source", "description"]
        table = _FullTextFrame.from_dict(POROUS_PRESETS, orient="index")[columns].rename_axis("preset")
        table = _filter_rows(table, {"status": status, "model": model, "kind": kind}, search,
                             ["product", "manufacturer", "source", "description"],
                             {"status": PRESET_STATUSES, "model": POROUS_MODELS, "kind": POROUS_PRESET_KINDS})
        return _sort_rows(table, sort_by)

    @staticmethod
    def plate_presets(search=None, status=None, sort_by=None):
        """
        Return the ``thin_plate_layer(preset=...)`` and ``membrane_layer(preset=...)`` material constants as a table.

        Each row holds ``rho``, ``E``, ``nu`` and ``loss_factor`` in ``thin_plate_layer()`` units and the ``status`` of
        the values. ``source`` names the table and printed page of Hopkins (2007), Bies and Hansen (2009), Mechel (2008)
        or Vigran (2008), and ``description`` says what the source lists and how. A field is empty in the display and
        NaN in the data (``None`` in the preset) where the source does not give it as a single value of the kind the
        layer takes. The ``"derived"`` rows compute ``E = rho*c_L**2*(1 - nu**2)`` from the quasi-longitudinal phase
        velocity ``c_L`` that Hopkins prints, which reproduces its ``h*f_c`` column. The ``"nominal"`` rows are the
        handbook values of Bies and Hansen and of Mechel, section I.8, table 3. Rows are ordered by density.

        ``status`` says how the acoustic values of a row were obtained. The acoustic values are the flow resistivity or
        resistance, porosity, tortuosity, characteristic lengths and thermal permeability, and for plates the density
        and the elastic constants. A secondary value whose origin the source does not state, such as a thickness,
        surface mass or bulk density, is named in the description and does not change the status. A row is
        ``"measured"`` when its acoustic values were measured on the sample, by direct or ultrasonic methods. It is
        ``"derived"`` when none of its acoustic values was fitted and at least one was computed by arithmetic from
        values the source prints, and ``"fitted"`` when at least one comes from indirect characterization, that is, was
        extracted by inverting an absorber model against measured surface impedance, absorption, or dynamic density and
        bulk modulus. ``"manufacturer"`` marks values that come from the producer of the material, as the source states.
        ``"nominal"`` marks representative values for a kind of material rather than for one sample, such as handbook
        tables of typical material constants and values a source calls nominal. ``"unstated"`` marks a row whose source
        does not say how the acoustic values of the material or product were obtained, also when the package computes
        one of them by arithmetic from values of that unstated origin, which does not make the row ``"derived"``.
        ``"range"`` marks ``plasterboard_800`` of ``plate_presets()``, the one row that takes the lower ends of printed
        ranges.

        Parameters
        ----------
        search : str, optional
            Keep only rows whose preset name, source or description contains this text, ignoring case, for example
            ``"plasterboard"``.
        status : str or list of str, optional
            Keep only these statuses, for example ``"nominal"`` or ``["derived", "range"]``. A status defined above that
            no row holds, or an empty list, gives an empty table, and a value that is not one of the seven raises
            ``ValueError``.
        sort_by : str or list of str, optional
            Sort the rows ascending by this column or these columns, where ``"preset"`` is the preset name, for example
            ``"E"``. Text sorts by character code, so capitals come before lower case and numbers in text sort digit by
            digit. Rows without a value come last, and rows that tie keep the order by density.

        Returns
        -------
        pandas.DataFrame
            One row per preset, indexed by preset name. The table displays every row and column and its text columns in
            full; the global pandas display options are left unchanged.
        """
        columns = [*PLATE_PARAMETERS, "status", "source", "description"]
        table = _FullTextFrame.from_dict(PLATE_PRESETS, orient="index")[columns].rename_axis("preset")
        table = _filter_rows(table, {"status": status}, search, ["source", "description"], {"status": PRESET_STATUSES})
        return _sort_rows(table, sort_by)

    @staticmethod
    def fabric_presets(search=None, status=None, method=None, sort_by=None):
        """
        Return the ``porous_facing_layer(preset=...)`` fabric and screen parameter sets as a table.

        Each row holds ``t``, ``surface_mass``, ``resistance``, ``porosity`` and ``tortuosity`` in
        ``porous_facing_layer()`` units, ``method``, the branch the layer uses for the row when the call gives none, and
        ``status``. ``source`` gives the author and year, the table or section and the printed page. ``description``
        names the sample, with its product and manufacturer where the source names them, and says how the source
        obtained each value. The arithmetic of the package, in the ``"derived"`` rows and in an ``"unstated"`` row whose
        inputs have no stated origin, is a resistance as resistivity times thickness or a surface mass as density times
        thickness, with the inputs named in the description. A value is empty in the display and NaN in the data where
        the source gives none (``None`` in the preset), and so are the porosity and tortuosity of the Tang and Zhu
        (2024) rows, where the source gives an optical perforation ratio and a tortuosity computed from it, which the
        description gives instead. Rows are ordered by resistance.

        ``status`` says how the acoustic values of a row were obtained. The acoustic values are the flow resistivity or
        resistance, porosity, tortuosity, characteristic lengths and thermal permeability, and for plates the density
        and the elastic constants. A secondary value whose origin the source does not state, such as a thickness,
        surface mass or bulk density, is named in the description and does not change the status. A row is
        ``"measured"`` when its acoustic values were measured on the sample, by direct or ultrasonic methods. It is
        ``"derived"`` when none of its acoustic values was fitted and at least one was computed by arithmetic from
        values the source prints, and ``"fitted"`` when at least one comes from indirect characterization, that is, was
        extracted by inverting an absorber model against measured surface impedance, absorption, or dynamic density and
        bulk modulus. ``"manufacturer"`` marks values that come from the producer of the material, as the source states.
        ``"nominal"`` marks representative values for a kind of material rather than for one sample, such as handbook
        tables of typical material constants and values a source calls nominal. ``"unstated"`` marks a row whose source
        does not say how the acoustic values of the material or product were obtained, also when the package computes
        one of them by arithmetic from values of that unstated origin, which does not make the row ``"derived"``.
        ``"range"`` marks ``plasterboard_800`` of ``plate_presets()``, the one row that takes the lower ends of printed
        ranges.

        Parameters
        ----------
        search : str, optional
            Keep only rows whose preset name, source or description contains this text, ignoring case, for example
            ``"woven"``.
        status : str or list of str, optional
            Keep only these statuses, for example ``"measured"`` or ``["measured", "derived"]``. A status defined above
            that no row holds, or an empty list, gives an empty table, and a value that is not one of the seven raises
            ``ValueError``.
        method : str or list of str, optional
            Keep only rows with this default method, ``"pieren"`` or ``"rebillard"``. Another method of
            ``porous_facing_layer()``, such as ``"pieren_free"``, or an empty list, gives an empty table, and any other
            value raises ``ValueError``.
        sort_by : str or list of str, optional
            Sort the rows ascending by this column or these columns, where ``"preset"`` is the preset name, for example
            ``"surface_mass"``. Text sorts by character code, so ``"woven_114"`` comes before ``"woven_55"`` and
            capitals come before lower case. Rows without a value come last, and rows that tie keep the order by
            resistance.

        Returns
        -------
        pandas.DataFrame
            One row per preset, indexed by preset name. The table displays every row and column and its text columns in
            full; the global pandas display options are left unchanged.
        """
        columns = [*FABRIC_PARAMETERS, "method", "status", "source", "description"]
        table = _FullTextFrame.from_dict(FABRIC_PRESETS, orient="index")[columns].rename_axis("preset")
        table = _filter_rows(table, {"status": status, "method": method}, search, ["source", "description"],
                             {"status": PRESET_STATUSES, "method": FACING_METHODS})
        return _sort_rows(table, sort_by)

    @staticmethod
    def absorption_table(category=None, search=None):
        """
        Return the bundled literature absorption data as a table.

        The rows come from Cox and D'Antonio [1], Appendix A, Vorländer [2], Annex, Beranek [3], Table A3.6, Beranek
        [4], Table V, and Kuttruff [5], Tables 6.2, 6.3, 6.5 and 6.6. Each row gives its table and printed page, and in
        ``primary_ref`` the references the book names for it, resolved in ``sources.csv`` of ``LITERATURE_FOLDER``.
        After ``primary_ref`` the table adds from ``sources.csv`` the full ``citation`` of the row's ``source_id`` and
        ``primary_citation``, the citations of the ``primary_ref`` entries joined by ``" | "``, empty where the row
        names none. [2] names the absorption database of the Physikalisch-Technische Bundesanstalt (PTB), compiled by
        Bork, as the origin of most of its data and manufacturer data as the origin of the rest, without saying which
        row comes from which. The descriptions are the package's own wording. Rows are ordered by category, then id.

        The band columns ``a_125`` to ``a_8000`` hold the values as printed and are empty where the source gives none.
        Only [5], Table 6.3, fills ``a_6000``. Values above 1 are kept. ``quantity`` is ``"alpha"`` for an absorption
        coefficient, and ``"area_per_person"``, ``"area_per_object"`` or ``"area_total"`` for an absorption area [m²].
        ``kind`` says how the source obtained the values. It is ``"reverberation_room"`` or
        ``"reverberation_room_scale_model"`` where the source states a reverberation-room measurement at full or model
        scale, ``"hall_derived"`` for Sabine coefficients derived from the reverberation times of concert halls, and
        ``"compiled"`` for the rows of [1], which states no method. ``provenance`` says where the book took the row
        from: the author's own hall data, a named primary source, a compilation with or without a reference per row, a
        book table that prints no source, or, for [2], the PTB database or manufacturer data without attribution per
        row. ``flags`` names copies in other sources, disagreements, misprints, values derived from printed ones, and a
        thickness, density or air space that the source prints only as a range or an open bound, which the row leaves
        empty.

        Parameters
        ----------
        category : str or list of str, optional
            Keep only these categories, for example ``"carpet"`` or ``["masonry", "concrete"]``.
        search : str, optional
            Keep only rows whose id, material or mounting contains this text, ignoring case.

        Returns
        -------
        pandas.DataFrame
            One row per entry, indexed by ``id``. The table displays every row and column and its text columns in full;
            the global pandas display options are left unchanged.

        References
        ----------
        [1] T. J. Cox and P. D'Antonio, "Acoustic Absorbers and Diffusers: Theory, Design and Application", 3rd ed.,
            CRC Press, 2017.
        [2] M. Vorländer, "Auralization: Fundamentals of Acoustics, Modelling, Simulation, Algorithms and Acoustic
            Virtual Reality", Springer, Berlin, 2008.
        [3] L. L. Beranek, "Concert Halls and Opera Houses: Music, Acoustics, and Architecture", 2nd ed., Springer,
            New York, 2004.
        [4] L. L. Beranek, "Analysis of Sabine and Eyring equations and their application to concert hall audience and
            chair absorption", Journal of the Acoustical Society of America, vol. 120, pp. 1399-1410, 2006.
        [5] H. Kuttruff, "Room Acoustics", 6th ed., CRC Press, 2017.
        """
        return _filter_rows(_with_citations(_literature_csv("absorption.csv")), {"category": category}, search,
                            ["material", "mounting"])

    @staticmethod
    def scattering_table(category=None, search=None):
        """
        Return the bundled literature random-incidence scattering coefficients as a table.

        The rows of Cox and D'Antonio [1], Appendix D, were measured to ISO 17497-1 at full or model scale and are given
        in third-octave bands from 100 Hz to 5 kHz, columns ``s_100`` to ``s_5000``. The rows of Vorländer [2], Annex,
        are rounded values, either in octave bands, columns ``s_125`` to ``s_8000``, or against the normalized frequency
        ``a/lambda = f*a/c``, columns ``s_al_0.125`` to ``s_al_8``, where ``a`` is the mean structural length of the
        surface ([2], p. 311). ``band_type`` names the set a row uses. The geometry columns give the depth, period and
        element width [mm] where the source states them, or the ratio ``h_over_a`` of mean structural depth to length.
        ``quantity`` is ``"s"``, the random-incidence scattering coefficient, for every row. ``kind`` is
        ``"iso17497_1_measured"`` for the measurements of [1] and ``"iso17497_1_rounded"`` for the rounded values of
        [2]. Values are kept as printed, including one above 1. ``provenance`` says whether a row is a measurement with
        its own reference or one of the rounded values of [2], which the book attributes to a named publication for some
        rows and to none for the others. The other columns are those of ``absorption_table()``, with the ``citation``
        and ``primary_citation`` columns it adds from ``sources.csv``.

        Parameters
        ----------
        category : str or list of str, optional
            Keep only these categories, for example ``"battens_1d"``.
        search : str, optional
            Keep only rows whose id or surface description contains this text, ignoring case.

        Returns
        -------
        pandas.DataFrame
            One row per entry, indexed by ``id``. The table displays every row and column and its text columns in full;
            the global pandas display options are left unchanged.

        References
        ----------
        [1] T. J. Cox and P. D'Antonio, "Acoustic Absorbers and Diffusers: Theory, Design and Application", 3rd ed.,
            CRC Press, 2017.
        [2] M. Vorländer, "Auralization: Fundamentals of Acoustics, Modelling, Simulation, Algorithms and Acoustic
            Virtual Reality", Springer, Berlin, 2008.
        """
        return _filter_rows(_with_citations(_literature_csv("scattering.csv")), {"category": category}, search,
                            ["surface"])

    def _resolve_preset(self, table, preset, layer_name, given):
        """
        Return ``(values, overrides)`` for a layer call with an optional preset name.

        ``values`` has the keys of ``given``: each passed value, or the preset value where the call passed ``None``.
        ``overrides`` is ``None`` or the text of the passed values that differ from non-``None`` preset values, and one
        ``UserWarning`` reports it at the caller of the layer method. Without a preset ``given`` is returned unchanged.
        An unknown name raises a ``ValueError``. No default is applied and no missing value is reported; the layer does
        both.
        """
        if preset is None:
            return dict(given), None
        if not isinstance(preset, str) or preset not in table:
            raise ValueError(f"Unknown preset {utils.describe_value(preset)}. Choose from {sorted(table)}.")
        row = table[preset]
        for name, value in given.items():
            if value is not None and (isinstance(value, (str, bytes, bool, np.bool_, np.ma.MaskedArray))
                                      or np.ndim(value) != 0):
                raise ValueError(f"{layer_name}(preset={preset!r}): {name} must be a single number, not "
                                 f"{utils.describe_value(value)}.")
        changed = [_preset_change(name, row[name], value) for name, value in given.items()
                   if value is not None and row[name] is not None and value != row[name]]
        overrides = ", ".join(changed) or None
        if overrides:
            warnings.warn(f"{layer_name}(preset={preset!r}): overriding preset values {overrides}.", UserWarning,
                          stacklevel=3)
        return {name: row[name] if value is None else value for name, value in given.items()}, overrides

    @staticmethod
    def _require_preset_values(layer_name, preset, values, names):
        """Raise a ``ValueError`` naming each field of ``names`` that is ``None`` in ``values`` after a preset."""
        missing = [name for name in names if values[name] is None]
        if missing:
            raise ValueError(f"{layer_name}(preset={preset!r}): the preset gives no value for {', '.join(missing)}; "
                             f"pass {'it' if len(missing) == 1 else 'them'} explicitly.")

    @staticmethod
    def _check_fibre_type(method, fibre_type):
        """Return ``fibre_type`` as an int when it is 1 or 2, the two fibre types of the Mechel-Grundmann model, and
        raise a ``ValueError`` otherwise."""
        return int(utils.check_number(method, "fibre_type", "the fibre type of the Mechel-Grundmann model", fibre_type,
                                      "1 (basalt or rock wool) or 2 (glass fibre)", lambda x: x in (1, 2)))

    @staticmethod
    def _check_porous_relations(method, model, tortuosity, static_tortuosity, static_thermal_tortuosity,
                                perforation_diameter, perforation_spacing, unit):
        """
        Raise a ``ValueError`` in the shared form for a static tortuosity of ``model="jcapl"`` that is not above the
        high-frequency one, or not above 1 for the thermal one, and for meso-perforations of ``model="double_porosity"``
        wider than the side of their square cell, where they overlap. ``unit`` is that of the perforation lengths.
        """
        if model == "jcapl":
            if static_tortuosity is not None:
                value = utils.real_number(static_tortuosity)
                finite = value is not None and np.isfinite(value)
                # Both values with the digits that tell them apart; anything but a finite number is echoed as given.
                bound, shown = utils.number_texts(tortuosity, value if finite else tortuosity)
                rule = f"a finite number above tortuosity = {bound} for model='jcapl'"
                if finite and not value > tortuosity:
                    raise ValueError(f"{method}: static_tortuosity, the static viscous tortuosity, must be {rule}, not "
                                     f"{shown}.")
                utils.check_number(method, "static_tortuosity", "the static viscous tortuosity", static_tortuosity,
                                   rule, lambda x: x > tortuosity)
            if static_thermal_tortuosity is not None:
                utils.check_number(method, "static_thermal_tortuosity", "the static thermal tortuosity",
                                   static_thermal_tortuosity, "a finite number above 1 for model='jcapl'",
                                   lambda x: x > 1)
        if model == "double_porosity" and perforation_diameter is not None and perforation_spacing is not None:
            utils.check_at_most(method, "perforation_diameter", f"the diameter of the meso-perforations [{unit}]",
                                perforation_diameter, "perforation_spacing", "the side of the square cell",
                                perforation_spacing, unit, " Wider perforations overlap; touching ones are accepted.")

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
        utils.check_choice("flow_resistivity", "model", "the empirical relation", model,
                           ("bies_hansen", "garai_pompoli"))
        density = np.asarray(utils.check_values("flow_resistivity", "density", "the bulk density [kg/m³]", density,
                                                "a finite number above 0, or an array of them", lambda x: x > 0),
                             dtype=float)
        if fibre_diameter is None:
            if model == "bies_hansen":
                raise ValueError("model 'bies_hansen' needs fibre_diameter [µm].")
            sigma = 25.989 * density ** 1.404
        else:
            d = np.asarray(utils.check_values("flow_resistivity", "fibre_diameter", "the mean fibre diameter [µm]",
                                              fibre_diameter, "a finite number above 0, or an array of them",
                                              lambda x: x > 0), dtype=float)
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
            Flow resistivity in SI units [Pa*s/m2], above 0. ``porous_layer()`` accepts [kPa*s/m2] and converts before
            calling this method. A ``sigma`` that is not one finite number above 0 raises a ``ValueError``.
        model : str, optional
            Equivalent-fluid model name, one of the branches listed above.
        fibre_type : int, optional
            Fibre type for ``mechel_grundmann``. ``1`` is basalt/rock wool and ``2`` is glass fibre; another value
            raises a ``ValueError``.
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
            ``True`` when the double-porosity layer lies on a rigid wall, ``False`` when air or another layer follows
            it. Only ``model="double_porosity"`` reads it; the other models also accept ``None``.

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
        method = "equivalent_fluid_model"
        utils.check_choice(method, "model", "the equivalent-fluid model", model, POROUS_MODELS)
        utils.check_choice(method, "frame", "the frame model", frame, POROUS_FRAMES)
        sigma = utils.check_positive(method, "sigma", "the flow resistivity [Pa*s/m²]", sigma)
        fibre_type = self._check_fibre_type(method, fibre_type)
        # The layer methods pass None for the models that do not read it, and True or False for double_porosity.
        if rigid_backed is not None or model == "double_porosity":
            rigid_backed = utils.check_flag(method, "rigid_backed", rigid_backed)
        porosity = utils.check_fraction(method, "porosity", "the open porosity", porosity)
        tortuosity = utils.check_tortuosity(method, "tortuosity", "the high-frequency tortuosity", tortuosity)
        checked = {}
        for name, what, value in (
                ("viscous_length", "the viscous characteristic length [m]", viscous_length),
                ("thermal_length", "the thermal characteristic length [m]", thermal_length),
                ("thermal_permeability", "the static thermal permeability [m²]", thermal_permeability),
                ("bulk_density", "the bulk density of the frame [kg/m³]", bulk_density),
                ("pore_shape_factor", "the pore shape factor", pore_shape_factor),
                ("perforation_diameter", "the diameter of the meso-perforations [m]", perforation_diameter),
                ("perforation_spacing", "the side of the square cell of one perforation [m]", perforation_spacing),
                ("thickness", "the layer thickness [m]", thickness)):
            checked[name] = None if value is None else utils.check_positive(method, name, what, value)
        for name, what, value in (("static_tortuosity", "the static viscous tortuosity", static_tortuosity),
                                  ("static_thermal_tortuosity", "the static thermal tortuosity",
                                   static_thermal_tortuosity)):
            checked[name] = None if value is None else utils.check_tortuosity(method, name, what, value)
        viscous_length, thermal_length = checked["viscous_length"], checked["thermal_length"]
        thermal_permeability, bulk_density = checked["thermal_permeability"], checked["bulk_density"]
        pore_shape_factor, thickness = checked["pore_shape_factor"], checked["thickness"]
        perforation_diameter, perforation_spacing = checked["perforation_diameter"], checked["perforation_spacing"]
        static_tortuosity = checked["static_tortuosity"]
        static_thermal_tortuosity = checked["static_thermal_tortuosity"]
        self._check_porous_relations(method, model, tortuosity, static_tortuosity, static_thermal_tortuosity,
                                     perforation_diameter, perforation_spacing, "m")
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
        else:  # "double_porosity", the last of POROUS_MODELS
            if frame != "rigid":
                raise ValueError("model='double_porosity' supports frame='rigid' only.")
            kc, zc = self._double_porosity(sigma, porosity, tortuosity, viscous_length, thermal_length,
                                           thermal_permeability, perforation_diameter, perforation_spacing,
                                           thickness, rigid_backed)
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
        if not 0 < sigma < np.inf or viscous_length <= 0 or thermal_length <= 0:
            raise ValueError("sigma, viscous_length and thermal_length must be greater than zero, and sigma finite.")

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
            if static_tortuosity <= tortuosity:
                raise ValueError(f"static_tortuosity must exceed tortuosity for model='jcapl': static_tortuosity is "
                                 f"{static_tortuosity!r} and tortuosity {tortuosity!r}.")
            if static_thermal_tortuosity <= 1:
                raise ValueError(f"static_thermal_tortuosity must exceed 1 for model='jcapl', not "
                                 f"{static_thermal_tortuosity!r}.")
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
        if not 0 < sigma < np.inf or tortuosity < 1 or pore_shape_factor <= 0:
            raise ValueError("sigma and pore_shape_factor must be greater than zero, sigma finite, and tortuosity at "
                             "least one.")
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
            raise ValueError(f"perforation_diameter must be positive and leave material between the perforations: the "
                             f"perforations of diameter {perforation_diameter!r} m fill the square cell of side "
                             f"perforation_spacing = {perforation_spacing!r} m.")
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
        ``bulk_density`` and ``thermal_permeability`` with the values of one characterized sample, where the set gives
        them, and any of these passed explicitly overrides the preset value. An override that differs from the preset
        issues one ``UserWarning`` listing each changed value, and the record keeps that list as ``preset_overrides``;
        ``rebuild()`` and ``load()`` do not repeat it. Without ``model``, a preset selects the model in its ``model``
        column. Complete sets select ``jcal`` when they include a thermal permeability (``q0'``) and ``jca`` otherwise.
        Sigma sets give the flow resistivity and at most the bulk density, porosity and tortuosity, and select a model
        of their material class: ``komatsu`` for glass and rock wool inside the flow resistivity range of its data and
        ``miki`` outside it, ``garai_pompoli`` for polyester, ``qunli`` for melamine and polyurethane foam, ``miki`` for
        sheep wool, and ``wilson_2015`` for the wood sets, which give porosity and tortuosity. A sigma set whose flow
        resistivity the source chose by fitting a model to the sample, or whose sample the source used to derive the
        coefficients of a model, selects that model instead. A model or ``frame="limp"`` that needs a field the preset
        leaves empty raises a ``ValueError`` naming it, unless the call passes it; ``frame="limp"`` needs ``porosity``
        and ``bulk_density``. Fields the model does not read take the defaults below. An explicit ``model`` takes only
        the parameters that model uses, so ``model="jca"`` on a ``jcal`` preset ignores its thermal permeability, and
        ``jcapl`` always needs its static parameters. Names give the material class and the flow resistivity; each set
        describes one sample, not a material class. ``show_layers(validity=True)`` reports a sigma set outside the
        fitted range of its model. The record stores the resolved model and values, so ``rebuild()``, ``load()`` and
        ``log_rebuild()`` do not depend on the preset table. ``porous_presets()`` lists every set with its values,
        source, status and description. A value outside the range its entry below gives, or an array for a parameter
        that is one number, passed or taken from the preset, raises a ``ValueError`` for every model, whether or not the
        model reads it. A thermal length below the viscous length is accepted, as fitted parameter sets can give one.

        Parameters
        ----------
        sigma : float, optional
            Flow resistivity of the porous material [k*Pa*s/m²], above 0. Defaults to the preset value, or 27 without a
            preset.
        t : float or int, optional
            Thickness of the porous material [mm], above 0.
        model : str, optional
            Equivalent-fluid model name. See ``equivalent_fluid_model()`` for the options. Defaults to the preset's
            model (column ``model`` of ``porous_presets()``), or ``"mac"`` without a preset.
        fibre_type : int, optional
            Fibre type for the Mechel-Grundmann model: ``1`` for basalt or rock wool, ``2`` for glass fibre.
        porosity : float, optional
            Open pore volume fraction for models that expose it, above 0 and at most 1. Used by ``model="wilson_2015"``,
            ``model="wilson_db"``, ``model="mechel_1976"``, ``jca``, ``jcal``, ``jcapl``, ``attenborough``,
            ``double_porosity`` and ``frame="limp"``. Defaults to the preset value, or 0.95 where the preset gives none
            or without a preset.
        tortuosity : float, optional
            High-frequency tortuosity, at least 1. Used by ``jca``, ``jcal``, ``jcapl``, ``double_porosity``,
            ``attenborough`` (as Attenborough's ``q**2``) and ``model="wilson_2015"``, where Wilson's internal ``q`` is
            computed as ``sqrt(tortuosity)``. Defaults to the preset value, or 1 where the preset gives none or without
            a preset.
        layer : None or int, optional
            Optional layer index, an integer or a float with an integer value. If ``None``, the layer is appended.
        viscous_length : float, optional
            Viscous characteristic length for ``jca``, ``jcal``, ``jcapl`` and ``double_porosity`` [µm], above 0.
        thermal_length : float, optional
            Thermal characteristic length for ``jca``, ``jcal``, ``jcapl`` and ``double_porosity`` [µm], above 0.
        thermal_permeability : float, optional
            Static thermal permeability for ``model="jcal"``, ``model="jcapl"`` and ``model="double_porosity"`` [m²],
            above 0.
        static_tortuosity : float, optional
            Static viscous tortuosity for ``model="jcapl"``, at least 1.
        static_thermal_tortuosity : float, optional
            Static thermal tortuosity for ``model="jcapl"``, at least 1.
        frame : str, optional
            ``"rigid"`` (default): the frame stays still, valid for held and stiff, heavy, or weakly coupled frames.
            ``"limp"``: the frame has no stiffness and moves with the pore air, adding its mass through Panneton's
            limp-frame density; suited to soft, light or unbonded layers, and it changes the result mainly at low
            frequency. See ``equivalent_fluid_model()`` for the equations and limits.
        bulk_density : float, optional
            In-vacuo bulk density of the frame, the mass of solid per unit volume of material, for ``frame="limp"``
            [kg/m³], above 0. Defaults to the preset value where the preset gives one.
        pore_shape_factor : float, optional
            Pore shape factor ratio for ``model="attenborough"``, above 0.
        perforation_diameter : float, optional
            Diameter of the circular meso-perforations for ``model="double_porosity"`` [mm], above 0 and at most
            ``perforation_spacing``.
        perforation_spacing : float, optional
            Side of the square cell holding one perforation for ``model="double_porosity"`` [mm], above 0.
        preset : str, optional
            Name of a material parameter set listed by ``porous_presets()``.

        References
        ----------
        [1] J. F. Allard and N. Atalla, "Propagation of Sound in Porous Media: Modelling Sound Absorbing
            Materials", 2nd ed., Wiley, 2009.
        """
        layer = utils.check_index("porous_layer", layer)
        given = {"sigma": sigma, "porosity": porosity, "tortuosity": tortuosity, "viscous_length": viscous_length,
                 "thermal_length": thermal_length, "thermal_permeability": thermal_permeability,
                 "bulk_density": bulk_density}
        given, preset_overrides = self._resolve_preset(POROUS_PRESETS, preset, "porous_layer", given)
        if preset is not None and model is None:
            model = POROUS_PRESETS[preset]["model"]
        if model is None:
            model = "mac"
        utils.check_choice("porous_layer", "model", "the equivalent-fluid model", model, POROUS_MODELS)
        utils.check_choice("porous_layer", "frame", "the frame model", frame, POROUS_FRAMES)
        if preset is not None:
            needs = list(POROUS_MODEL_NEEDS.get(model, ()))
            if frame == "limp":
                needs += [name for name in ("porosity", "bulk_density") if name not in needs]
            self._require_preset_values("porous_layer", preset, given, needs)
        for name, default in (("sigma", 27), ("porosity", 0.95), ("tortuosity", 1.0)):
            if given[name] is None:
                given[name] = default
        # A passive material has a flow resistivity above zero, and the models divide by it.
        sigma = utils.check_positive("porous_layer", "sigma", "the flow resistivity [k*Pa*s/m²]", given["sigma"])
        t = utils.check_thickness("porous_layer", t)
        fibre_type = self._check_fibre_type("porous_layer", fibre_type)
        porosity = utils.check_fraction("porous_layer", "porosity", "the open porosity", given["porosity"])
        tortuosity = utils.check_tortuosity("porous_layer", "tortuosity", "the high-frequency tortuosity",
                                            given["tortuosity"])
        checked = {}
        for name, what, value in (
                ("viscous_length", "the viscous characteristic length [µm]", given["viscous_length"]),
                ("thermal_length", "the thermal characteristic length [µm]", given["thermal_length"]),
                ("thermal_permeability", "the static thermal permeability [m²]", given["thermal_permeability"]),
                ("bulk_density", "the bulk density of the frame [kg/m³]", given["bulk_density"]),
                ("pore_shape_factor", "the pore shape factor", pore_shape_factor),
                ("perforation_diameter", "the diameter of the meso-perforations [mm]", perforation_diameter),
                ("perforation_spacing", "the side of the square cell of one perforation [mm]", perforation_spacing)):
            checked[name] = None if value is None else utils.check_positive("porous_layer", name, what, value)
        for name, what, value in (("static_tortuosity", "the static viscous tortuosity", static_tortuosity),
                                  ("static_thermal_tortuosity", "the static thermal tortuosity",
                                   static_thermal_tortuosity)):
            checked[name] = None if value is None else utils.check_tortuosity("porous_layer", name, what, value)
        viscous_length, thermal_length = checked["viscous_length"], checked["thermal_length"]
        thermal_permeability, bulk_density = checked["thermal_permeability"], checked["bulk_density"]
        pore_shape_factor = checked["pore_shape_factor"]
        perforation_diameter, perforation_spacing = checked["perforation_diameter"], checked["perforation_spacing"]
        static_tortuosity = checked["static_tortuosity"]
        static_thermal_tortuosity = checked["static_thermal_tortuosity"]
        self._check_porous_relations("porous_layer", model, tortuosity, static_tortuosity, static_thermal_tortuosity,
                                     perforation_diameter, perforation_spacing, "mm")

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

        self._warn_if_terminated("porous_layer")
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
            Thickness of the air layer [mm], above 0. Another value raises a ``ValueError``.
        layer : None or int, optional
            Optional layer index, an integer or a float with an integer value. If ``None``, the layer is appended.
        """
        layer = utils.check_index("air_layer", layer)
        t = utils.check_thickness("air_layer", t)
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

        self._warn_if_terminated("air_layer")
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
        configured incidence angles. The termination replaces any stored layers, with a ``UserWarning``, and a layer
        added afterwards warns that ``compute()`` ignores it. ``self.scat`` becomes ``None``, and the file name loses
        the suffix of ``material_model()``.

        Parameters
        ----------
        z : float or complex, optional
            Specific surface impedance value to assign at every frequency and configured incidence angle [Pa*s/m], one
            number. An array or a list, a value with a negative real part or a NaN, and a value whose absorption comes
            out NaN raise a ``ValueError``; 0 and positive infinity are accepted.
        """
        discarded = self._discarded_layers_message()
        filename = self._filename_without_material_model()
        self._set_constant_z(z)
        # A new termination has no scattering coefficient, and the file name loses the suffix of material_model().
        self._scat = None
        self.filename = filename
        if discarded is not None:
            warnings.warn(f"constant_z: {discarded}", UserWarning, stacklevel=2)

    def _set_constant_z(self, z):
        """
        Set the ``constant_z()`` termination, keeping ``self.scat`` and ``self.filename``, as ``rebuild()`` needs. Every
        value is computed before the object is changed, so a call that raises leaves it as it was.
        """
        z = utils.check_impedance("constant_z", "z", "the specific surface impedance [Pa*s/m]", z)
        dtype = complex if isinstance(z, complex) else float
        self._set_locally_reacting(np.full(len(self.freq), z, dtype=dtype), {"termination": {"type": "constant_z"}},
                                   "constant_z")

    def _raise_if_nan_absorption(self, alpha, what, argument=None):
        """
        Raise a ``ValueError`` when the absorption ``alpha`` of ``what``, the stack or a termination, is NaN at some
        frequency. The inputs are checked when they are given, so NaN comes from arithmetic that overflows. For a
        termination, ``argument`` names the input that its impedance comes from.
        """
        nan = np.isnan(np.asarray(alpha, dtype=float).reshape(-1))
        if not np.any(nan):
            return
        found = (f"The absorption of {what} is NaN at {int(nan.sum())} of {nan.size} frequencies, the first at "
                 f"{float(self.freq[np.argmax(nan)]):g} Hz.")
        if argument is not None:
            raise ValueError(f"{found} Every input is checked when it is given, so the impedance of the termination "
                             f"overflowed in the arithmetic. Check {argument}.")
        raise ValueError(f"{found} Every input is checked when it is given, so the arithmetic overflowed: a value far "
                         f"outside its usual range, often a length or a resistivity in the wrong unit, such as "
                         f"viscous_length=1e-4 (metres) for 100 µm, or an extremely thick, resistive layer. Check the "
                         f"values that show_layers() prints.")

    def _set_locally_reacting(self, z, matrix, caller):
        """
        Set the impedance ``z`` [Pa*s/m] on ``self.freq`` of a locally reacting surface, the termination of
        ``constant_z()`` and ``material_model()``, and store ``matrix`` as its record. ``self.z_angle`` repeats ``z`` at
        every incidence angle. The cached absorption is the plane-wave value at the angle of an angle object and the
        Paris average over ``incidence_angle`` with ``diffuse_method="paris"``. Otherwise the ``alpha`` property gives
        the normal-incidence value of ``z``, which is also the field-incidence value, since an angle-independent
        impedance is its own field impedance. An absorption that comes out NaN raises the ``ValueError`` of
        ``compute()``, naming ``caller``. Every value is computed before the object is changed, so a call that raises
        leaves it as it was.
        """
        z_angle = np.repeat(z[:, None], len(self.incidence_angle), axis=1)
        alpha = None
        if self.incidence == "angle":
            _, alpha = self.reflection_and_absorption_coefficient(z, angle=self.incidence_angle[0])
            alpha = np.asarray(alpha, dtype=float).reshape((len(self.freq),))
        elif self.incidence == "diffuse" and self.diffuse_method == "paris":
            alpha = self.diffuse_absorption_coefficient(z_angle)
        with np.errstate(invalid="ignore", over="ignore"):
            computed = alpha if alpha is not None else self.reflection_and_absorption_coefficient(z)[1]
        self._raise_if_nan_absorption(computed, f"the {caller}() termination", TERMINATION_ARGUMENTS[caller])
        self._clear_stale_state()  # A new termination replaces whatever the stale results described.
        self.z = z
        self.z_angle = z_angle
        self._alpha = alpha
        self.matrix = matrix

    def _filename_without_material_model(self):
        """Return ``self.filename`` without the suffix that an earlier ``material_model()`` termination added."""
        if "material_model" in self.matrix and self.filename.endswith("_material_model"):
            return self.filename[:-len("_material_model")]
        return self.filename

    def _discarded_layers_message(self):
        """
        Return the sentence a termination warns with when it replaces stored layers, or ``None`` when there are none.
        """
        count = sum(1 for key, value in self.matrix.items()
                    if key not in ("termination", "material_model") and isinstance(value, dict)
                    and value.get("type") != "backing")
        if not count:
            return None
        return (f"the {count} stored layer{'s are' if count > 1 else ' is'} discarded, since a termination sets the "
                f"surface impedance of the whole treatment.")

    def _termination_name(self):
        """Return the name of the method that set the surface impedance, or ``None`` for a layer stack."""
        if "material_model" in self.matrix:
            return "material_model"
        if "termination" not in self.matrix:
            return None
        record = self.matrix["termination"]
        return "material_from_table" if "literature" in record else record["type"]

    def _has_paris_termination_alpha(self):
        """
        Return True when ``alpha`` is the Paris absorption over 0 to 90 degrees that a diffuse object takes from an
        ``impedance_from_absorption()`` termination, whatever ``diffuse_method`` and ``incidence_angle`` say.
        """
        record = self.matrix.get("termination") if isinstance(self.matrix, dict) else None
        return (self.incidence == "diffuse" and isinstance(record, dict)
                and record.get("type") == "impedance_from_absorption")

    @staticmethod
    def _record_angle_limit(record):
        """
        Return the ``angle_limit`` [deg] of an ``impedance_from_absorption()`` record: the stored value, 90 for a Sabine
        record without the key, and ``None`` for other data.
        """
        limit = record.get("angle_limit [deg]")
        if limit is None:
            return 90.0 if record.get("incidence") == "sabine" else None
        return float(limit)

    def _termination_angles(self):
        """
        Return the angles [deg] over which ``finite_sample_absorption()`` integrates by default for the termination of
        ``impedance_from_absorption()`` on a diffuse object, 0 to its ``angle_limit`` or to 90 degrees, or ``None`` for
        any other surface.
        """
        if not self._has_paris_termination_alpha():
            return None
        return self._full_range_angles(self._record_angle_limit(self.matrix["termination"]) or 90.0)

    def _warn_if_terminated(self, layer_type):
        """Warn that a layer added to a treatment whose surface impedance a termination sets has no effect."""
        name = self._termination_name()
        if name is None:
            return
        warnings.warn(f"{layer_type}: the surface impedance is set by {name}(), which compute() does not combine with "
                      f"layers, so this layer has no effect.", UserWarning, stacklevel=3)

    def impedance_from_absorption(self, bands, alpha, *, incidence="random", method="real", absorber=None, root="hard",
                                  sample_size=None, angle_limit=None, extrapolation=None, fraction=None):
        """
        Set a locally reacting surface impedance that reproduces band absorption coefficients.

        Like ``constant_z()``, this sets a termination rather than a layer. ``self.z`` holds the impedance and
        ``self.z_angle`` repeats it at every incidence angle, since the surface is taken as locally reacting, and
        ``compute()`` leaves both unchanged. ``self.scat`` becomes ``None``, and the file name loses the suffix of
        ``material_model()``. ``incidence`` names the kind of data. ``"normal"`` is normal-incidence absorption, as from
        an impedance tube, related to ``zeta = Z/(rho0*c0)`` by Kuttruff [1], eq. 2.11 at ``theta = 0``. ``"random"`` is
        the random-incidence absorption of an infinite surface, the Paris average in closed form, [1], eq. 2.54.
        ``"sabine"`` is the absorption measured in a reverberation room on a rectangular sample of sides
        ``sample_size``. Its forward model is Thomasson's statistical absorption coefficient of a finite sample [2],
        eq. 25, the size correction of Jeong [3], eqs. 5 and 6. It is evaluated over polar angles from 0 to
        ``angle_limit``, at steps of at most 1 degree, with the field impedance of ``finite_sample_absorption()``,
        ``SABINE_AZIMUTH_POINTS`` azimuth points and uniform angular weights, and averages the absorption over azimuth
        (``tmm._impedance_retrieval.finite_sample_alpha``). By default ``angle_limit`` is 90 degrees, the upper limit of
        the integrals of [2], eqs. 25 and 26, and [3], eqs. 6 and 7. Without ``sample_size`` the sample is a square of
        ``SABINE_SAMPLE_AREA``, the 11 m² that Mondet et al. [4], p. 2, assume for samples of unknown size, after
        stating that ISO 354 recommends 10 to 12 m².

        ``method="real"`` gives a purely resistive impedance. It takes each band value as a point value at the band
        centre, joins the values by a monotone cubic in ``log2(f)`` and holds them beyond the outer centres
        (``tmm._impedance_retrieval.bands_to_grid``), so the band means of the result differ from the input. It then
        inverts the forward relation at each frequency of ``self.freq``. For ``"normal"`` the inverse is
        ``zeta = (1 + s)/(1 - s)``, or its reciprocal for the soft root, with ``s = sqrt(1 - alpha)`` ([1], eq. 2.9).
        For ``"random"`` the inverse of [1], eq. 2.54, is numerical, and for ``"sabine"`` so is the inverse of the
        finite-sample model, there at ``SABINE_NODES_PER_OCTAVE`` points per octave with the reflection factor
        ``(zeta - 1)/(zeta + 1)`` interpolated between them. ``root`` picks the root above (``"hard"``) or below
        (``"soft"``) the absorption maximum. ``method="55deg"`` converts random-incidence data in the same way with the
        55 degree rule, ``Y = cos(55 deg)*(1 - s)/(1 + s)`` (Hargreaves et al. [5], section III.E; Cox and D'Antonio
        [6], section 13.1.1). ``method="model"`` fits the RMK+1 model of Mondet et al. [4], eq. 24,
        ``zeta = K/(j*omega) + R1 + M*j*omega + G*(j*omega)**gamma``, to the band means of the forward absorption
        ([4], eq. 26) under the constraints of the ``absorber`` class ([4], eqs. 28 to 32), see
        ``tmm._impedance_retrieval.fit_absorber_model``. Without a class the fit keeps ``Re(zeta) > 1``, the
        hard-material constraint of [4], eq. 28, and only a soft class lifts it. [4] fit six octave bands of Sabine
        data, so the Paris and normal-incidence forward models of the fit, and the validity limits of
        [4], eqs. 34 to 36, applied to them or to other bands, are package extensions.

        Absorption fixes only ``abs(R)``, so the phase of the result comes from the method, not from the data.
        ``"real"`` and ``"55deg"`` give no reactance and a resistance that changes with frequency, which is not causal
        ([4], eqs. 2 to 4), so they suit energy-based models fed with the same kind of coefficient, not phase-sensitive
        or time-domain use. The RMK+1 model is passive and causal ([4], p. 4), and its reactance follows from the band
        shape and the class constraints. Holding the impedance outside the bands (``extrapolation="hold"``) makes it
        non-causal. No locally reacting surface reaches a random-incidence absorption above 0.951 ([1], p. 41). A
        locally reacting termination has no field-incidence impedance distinct from ``z``, so on a diffuse object
        ``alpha`` is the Paris absorption of the result over 0 to 90 degrees, [1], eq. 2.54, whatever ``diffuse_method``
        and ``incidence_angle`` say, and ``plot()``, ``save2sheet()``, ``show_layers()`` and ``alpha_w()`` label it so.
        For random-incidence data that is the forward model of the retrieval, and for Sabine data the absorption without
        the size effect. ``finite_sample_absorption()`` without ``z_angle`` and ``angles`` integrates over the angles of
        the Sabine forward model, 0 to ``angle_limit``, or 0 to 90 degrees for other data, on field and Paris objects
        alike, so a Sabine retrieval reproduces its own band data there.

        ``self.matrix["termination"]`` records the type ``"impedance_from_absorption"``, the input bands and values, the
        options, with ``angle_limit [deg]`` the limit used (90 for Sabine data without ``angle_limit``, ``None`` for
        other data), the fitted ``parameters`` (``"model"`` only), the band means of the result under the forward model
        as ``band_alpha`` and their root-mean-square error from the input as ``band_rms``. For ``"model"`` it also holds
        the validity limit ``band_rms_limit`` that applies to the data and the verdict ``fit_valid``, true when
        ``band_rms`` is below it. ``checked_range [Hz]`` holds the lowest and highest frequency of the grid the record
        was made on. ``rebuild()`` repeats the extrapolation warning only when that grid stayed within one band of the
        bands and the current grid does not. ``rebuild()`` and ``load()`` set the termination again from this record on
        the current grid, from the stored ``parameters`` without a new fit, and a Sabine record without
        ``angle_limit [deg]`` takes 90. ``log_rebuild()`` writes the call, which fits again, after a comment line with
        the stored ``parameters``.

        Parameters
        ----------
        bands : array of float
            Nominal centre frequencies of fractional-octave bands [Hz], ascending.
        alpha : array of float
            Absorption coefficient of each band.
        incidence : str, optional
            Kind of data: ``"normal"``, ``"random"`` or ``"sabine"``. It does not change ``self.incidence``.
        method : str, optional
            ``"real"``, ``"55deg"`` (``incidence="random"`` only) or ``"model"``.
        absorber : str, optional
            Absorber class of ``method="model"``: ``"hard"``, ``"soft"``, ``"soft_porous"``, ``"hard_porous"``,
            ``"membrane"``, ``"perforated"`` or ``"microperforated"`` ([4], eqs. 28 to 32 and p. 5). ``None`` applies
            the hard-material constraint alone.
        root : str, optional
            Root of ``method="real"`` and ``"55deg"``: ``"hard"`` (resistance at or above the value of largest
            absorption) or ``"soft"``. ``method="model"`` raises for ``"soft"``.
        sample_size : float or tuple of float, optional
            Sample side lengths ``(a, b)`` [m] of ``incidence="sabine"``. A single value gives a square. A side above
            ``SAMPLE_SIZE_LIMIT`` raises a ``ValueError``, since it is taken for a side in millimetres.
        angle_limit : float, optional
            Upper polar angle [deg] of the finite-sample integral of ``incidence="sabine"``, above 0 and at most 90.
            ``None`` gives 90, as in [2] and [3]. 78 gives the range that the reverberation-room mode of NorFLAG offers
            for a closer match to reverberation-room measurements ([7], section 5.4.1, p. 40), a convention of that
            program that is not in [2] or [3].
        extrapolation : str, optional
            Values outside the bands. ``"hold"`` keeps the absorption of the outer band centres for ``"real"`` and
            ``"55deg"``, and the impedance at the outer band edges for ``"model"``. Hargreaves et al. [5], section
            III.E, hold the absorption below their lowest band, which is 20 Hz, so a hold below a higher lowest band or
            above the highest band is a package choice. ``"model"`` evaluates the fitted model there and needs
            ``method="model"``. At very low frequencies, Mondet et al. [4], p. 11, write that the RMK+1 extrapolation
            "can provide more accuracy than the simple and common assumption of constant values between the 63 Hz and
            125 Hz octave bands". Defaults to ``"model"`` for ``method="model"`` and ``"hold"`` otherwise.
        fraction : int, optional
            Bandwidth denominator: 1 for octave bands, 3 for third-octave bands. Inferred from ``bands`` when omitted,
            which needs at least two contiguous bands.

        Notes
        -----
        The method raises ``ValueError`` for an unknown ``incidence``, ``method``, ``absorber``, ``root`` or
        ``extrapolation``, for ``alpha`` without one value per band, for complex, NaN, infinite or negative ``alpha``,
        for ``alpha``, ``bands``, ``sample_size`` or ``angle_limit`` given as text, booleans or a masked array, for
        values of ``alpha`` above ``ABSORPTION_PERCENT_LIMIT``, which look like percentages, for ``alpha`` above 1 with
        ``"normal"`` or ``"random"`` data, for ``absorber`` or ``extrapolation="model"`` without ``method="model"`` and
        for ``root="soft"`` with it, for ``"55deg"`` with data other than ``"random"``, for ``sample_size`` or
        ``angle_limit`` without ``"sabine"``, for ``sample_size`` not positive or with a side above
        ``SAMPLE_SIZE_LIMIT`` [m], for ``angle_limit`` not above 0 and at most 90, and for complex ``bands``, ``bands``
        off the fractional-octave series or, without ``fraction``, not contiguous. It gives at most one ``UserWarning``
        per call, naming each of these conditions that applies: random-incidence values at or above 0.951, Sabine values
        at or above the largest a real impedance reaches with ``"real"``, band means of the result more than
        ``ABSORPTION_FIT_LIMIT`` RMS from the input, an RMK+1 fit that fails its validity limit
        (``tmm._impedance_retrieval.fit_validity_limit``), two RMK+1 fits that match the bands equally well with
        normal-incidence reflection factors more than ``ABSORPTION_FIT_SPREAD`` apart, a frequency grid more than one
        band beyond the outer band edges, with the ``fmin`` or ``fmax`` that stays within one band, or that range for a
        grid wholly above or below the bands, and stored layers that the termination discards. For ``"real"`` and
        ``"55deg"`` with the hard root, the sentence on the grid also names ``method="model"``, which evaluates the
        fitted model outside the bands. A layer added afterwards warns that ``compute()`` ignores it.

        References
        ----------
        [1] H. Kuttruff, "Room Acoustics", 6th ed., CRC Press, 2017.
        [2] S.-I. Thomasson, "On the absorption coefficient", Acustica, vol. 44, pp. 265-273, 1980.
        [3] C.-H. Jeong, "Converting Sabine absorption coefficients to random incidence absorption coefficients",
            Journal of the Acoustical Society of America, vol. 133, pp. 3951-3962, 2013.
        [4] B. Mondet, J. Brunskog, C.-H. Jeong and J. H. Rindel, "From absorption to impedance: Enhancing boundary
            conditions in room acoustic simulations", Applied Acoustics, vol. 157, 106884, 2020.
        [5] J. A. Hargreaves, L. R. Rendell and Y. W. Lam, "A framework for auralization of boundary element method
            simulations including source and receiver directivity", Journal of the Acoustical Society of America,
            vol. 145, pp. 2625-2637, 2019.
        [6] T. J. Cox and P. D'Antonio, "Acoustic Absorbers and Diffusers: Theory, Design and Application", 3rd ed.,
            CRC Press, 2017.
        [7] T. E. Vigran, "Manual for NorFLAG, version 4.0", Trondheim, 2018.
        """
        self._impedance_from_absorption(bands, alpha, incidence, method, absorber, root, sample_size, angle_limit,
                                        extrapolation, fraction, "impedance_from_absorption")

    def _impedance_from_absorption(self, bands, alpha, incidence, method, absorber, root, sample_size, angle_limit,
                                   extrapolation, fraction, caller, scat=None, literature=None, extra=None):
        """
        Check the arguments of ``impedance_from_absorption()``, set the termination and give its one warning.

        The warning starts with the name of the public method ``caller`` and points at the line that called it.
        ``material_from_table()`` passes its scattering coefficient on ``self.freq`` as ``scat`` and its record as
        ``literature``, which are set together with the termination, and in ``extra`` the sentences of its own that the
        warning ends with.
        """
        bands = retrieval._validated_bands(bands)
        alpha = np.atleast_1d(retrieval._real_values(alpha, "alpha"))
        if alpha.shape != bands.shape or not np.all(np.isfinite(alpha)):
            raise ValueError("alpha must hold one finite value per band.")
        if not isinstance(incidence, str) or incidence not in ABSORPTION_INCIDENCES:
            raise ValueError(f"incidence must be one of {ABSORPTION_INCIDENCES}, not "
                             f"{utils.describe_value(incidence)}.")
        if not isinstance(method, str) or method not in ABSORPTION_METHODS:
            raise ValueError(f"method must be one of {ABSORPTION_METHODS}, not {utils.describe_value(method)}.")
        if method == "model":
            if not isinstance(root, str) or root != "hard":
                raise ValueError("root applies to method='real' and '55deg'; method='model' takes an absorber class.")
            if absorber is not None and (not isinstance(absorber, str) or absorber not in retrieval.ABSORBER_CLASSES):
                raise ValueError(f"absorber must be None or one of {sorted(retrieval.ABSORBER_CLASSES)}, not "
                                 f"{utils.describe_value(absorber)}.")
        else:
            retrieval._validate_root(root)
            if absorber is not None:
                raise ValueError("absorber applies to method='model' only.")
        if method == "55deg" and incidence != "random":
            raise ValueError("The 55 degree rule converts random-incidence data; use incidence='random'.")
        if extrapolation is None:
            extrapolation = "model" if method == "model" else "hold"
        if (not isinstance(extrapolation, str) or extrapolation not in ("hold", "model")
                or (extrapolation == "model" and method != "model")):
            raise ValueError(f"extrapolation must be 'hold', or 'model' with method='model', not "
                             f"{utils.describe_value(extrapolation)}.")
        if sample_size is not None and incidence != "sabine":
            raise ValueError("sample_size applies to incidence='sabine' only.")
        sides = None
        if incidence == "sabine":
            sides = utils.check_sample_size(caller, np.sqrt(SABINE_SAMPLE_AREA) if sample_size is None else sample_size,
                                            SAMPLE_SIZE_LIMIT)
        if angle_limit is not None:
            if incidence != "sabine":
                raise ValueError("angle_limit applies to incidence='sabine' only.")
            limit = retrieval._real_values(angle_limit, "angle_limit")
            if limit.ndim != 0 or not 0 < float(limit) <= 90:
                raise ValueError(f"angle_limit must be a single angle above 0 and at most 90 [deg], not "
                                 f"{angle_limit!r}.")
            angle_limit = float(limit)
        elif incidence == "sabine":
            angle_limit = 90.0
        fraction = retrieval.band_fraction(bands, fraction)
        if np.any(alpha < 0):
            raise ValueError("alpha cannot be negative.")
        if np.any(alpha > ABSORPTION_PERCENT_LIMIT):
            raise ValueError(f"alpha has values above {ABSORPTION_PERCENT_LIMIT:g} (largest {np.max(alpha):g}), which "
                             f"look like percentages. Pass absorption coefficients, the percentages divided by 100.")
        if incidence != "sabine" and np.any(alpha > 1):
            raise ValueError(f"{incidence.capitalize()}-incidence absorption cannot exceed 1. Values above 1 come from "
                             f"reverberation-room (Sabine) measurements; pass incidence='sabine'.")

        discarded = self._discarded_layers_message()
        filename = self._filename_without_material_model()
        messages = self._set_absorption_termination(bands, alpha, incidence, method, absorber, root, sides,
                                                    extrapolation, fraction, angle_limit=angle_limit, caller=caller)
        messages.update(extra or {})
        # A new termination has no scattering coefficient but that of material_from_table(), and the file name loses the
        # suffix of material_model().
        self._scat = scat
        self.filename = filename
        if literature is not None:
            if literature["call"]["angle_limit"] is not None:
                literature["call"]["angle_limit"] = angle_limit
            self.matrix["termination"]["literature"] = literature
        if discarded is not None:
            messages = {"discarded": discarded[0].upper() + discarded[1:], **messages}
        if messages:
            text = " ".join(messages.values())
            warnings.warn(f"{caller}: {text[0].lower()}{text[1:]}", UserWarning, stacklevel=3)

    def _set_absorption_termination(self, bands, alpha, incidence, method, absorber, root, sides, extrapolation,
                                    fraction, parameters=None, angle_limit=None, caller="impedance_from_absorption"):
        """
        Set the termination of ``impedance_from_absorption()`` from checked arguments and return its warnings.

        Given the ``parameters`` of an earlier RMK+1 fit, as ``rebuild()`` passes them, the fit is not repeated.
        ``angle_limit`` [deg] is that of Sabine data and ``None`` for other data. The warnings come back as a dictionary
        of sentences by kind, for the caller to issue. ``caller`` is the public method that a NaN absorption names.
        """
        edges = utils.band_edges(bands, fraction)
        points = retrieval.band_points(bands, fraction)
        freq = np.asarray(self.freq, dtype=float)
        forward = self._absorption_forward_model(incidence, sides, 90.0 if angle_limit is None else angle_limit)

        fit = None
        if method == "model":
            if parameters is None:
                fit = retrieval.fit_absorber_model(bands, alpha, absorber or "hard", forward, fraction)
                parameters = fit["parameters"]
            parameters = {name: float(parameters[name]) for name in retrieval.RMK1_PARAMETERS}
            params = [parameters[name] for name in retrieval.RMK1_PARAMETERS]

            def zeta_at(f):
                f = np.asarray(f, dtype=float)
                return retrieval.rmk1_impedance(params, np.clip(f, edges[0, 0], edges[-1, 2])
                                                if extrapolation == "hold" else f)
        else:
            zeta_at = self._real_absorption_impedance(bands, alpha, incidence, method, root, forward)

        zeta = np.asarray(zeta_at(freq), dtype=complex)
        band_alpha = np.asarray(forward(zeta_at(points.ravel()), points.ravel()), dtype=float).reshape(points.shape)
        band_alpha = band_alpha.mean(axis=1)
        band_rms = float(np.sqrt(np.mean((band_alpha - alpha) ** 2)))
        rms_limit = retrieval.fit_validity_limit(alpha)[0] if method == "model" else None

        with np.errstate(invalid="ignore"):
            z = np.where(np.isinf(zeta.real), np.inf + 0j, zeta * self.z0)
        alpha_result = None
        if self.incidence == "angle":
            _, angle_alpha = self.reflection_and_absorption_coefficient(z, angle=self.incidence_angle[0])
            alpha_result = np.asarray(angle_alpha, dtype=float).reshape(-1)
        elif self.incidence == "diffuse":
            # Both diffuse methods give the Paris absorption over 0 to 90 degrees for this termination.
            alpha_result = np.asarray(retrieval.paris_alpha(zeta), dtype=float).reshape(-1)
        with np.errstate(invalid="ignore", over="ignore"):
            computed = alpha_result if alpha_result is not None else self.reflection_and_absorption_coefficient(z)[1]
        self._raise_if_nan_absorption(computed, f"the {caller}() termination", TERMINATION_ARGUMENTS[caller])
        record = {
            "type": "impedance_from_absorption",
            "incidence": incidence,
            "method": method,
            "absorber": absorber,
            "root": None if method == "model" else root,
            "sample_size [m]": None if sides is None else [float(side) for side in sides],
            "angle_limit [deg]": None if angle_limit is None else float(angle_limit),
            "extrapolation": extrapolation,
            "fraction": int(fraction),
            "bands [Hz]": [float(value) for value in bands],
            "alpha": [float(value) for value in alpha],
            "parameters": parameters,
            "band_alpha": [float(value) for value in band_alpha],
            "band_rms": band_rms,
            "band_rms_limit": rms_limit,
            "fit_valid": None if rms_limit is None else bool(band_rms < rms_limit),
            "checked_range [Hz]": [float(freq.min()), float(freq.max())],
        }

        def listed(mask):
            return ", ".join(f"{value:g}" for value in bands[mask])

        messages = {}
        if incidence == "random" and np.any(alpha >= retrieval.PARIS_ALPHA_MAX):
            outcome = (f"the real root is held at zeta = {retrieval.PARIS_ZETA_MAX:.3f} there" if method == "real"
                       else "the result stays below them")
            messages["ceiling"] = (f"No locally reacting surface reaches a random-incidence absorption above "
                                   f"{retrieval.PARIS_ALPHA_MAX:.3f} (Kuttruff 2017, p. 41), and the data reach it in "
                                   f"the bands at {listed(alpha >= retrieval.PARIS_ALPHA_MAX)} Hz, so {outcome}.")
        if incidence == "sabine" and method == "real":
            _, reachable = retrieval.real_impedance_inverse(alpha, bands, forward)
            if np.any(alpha >= reachable):
                messages["sabine_ceiling"] = (f"No real impedance reaches the Sabine absorption of the bands at "
                                              f"{listed(alpha >= reachable)} Hz for this sample size, so the real root "
                                              f"is held at the impedance of largest absorption there.")
        limit_text = (f"the validity limit of {rms_limit:.3f} RMS that Mondet et al. (2020), eqs. 34-36, set for six "
                      f"octave bands of Sabine data" if rms_limit is not None else "")
        if band_rms > ABSORPTION_FIT_LIMIT:
            worst = int(np.argmax(np.abs(band_alpha - alpha)))
            validity = ""
            if rms_limit is not None:
                validity = f" The fit {'passes' if band_rms < rms_limit else 'fails'} {limit_text}."
            elif method == "55deg":
                validity = (" The 55 degree rule approximates the random-incidence relation, which method='real' "
                            "inverts.")
            messages["band_rms"] = (f"The band means of the retrieved impedance differ from the input by "
                                    f"{band_rms:.3f} RMS, most at {bands[worst]:g} Hz ({band_alpha[worst]:.2f} against "
                                    f"{alpha[worst]:.2f}).{validity}")
        elif rms_limit is not None and band_rms >= rms_limit:
            messages["validity"] = (f"The band means of the RMK+1 fit differ from the input by {band_rms:.3f} RMS, so "
                                    f"the fit fails {limit_text}.")
        if fit is not None and fit["spread"] > ABSORPTION_FIT_SPREAD:
            messages["spread"] = (f"Another start of the RMK+1 fit, with a root-mean-square band error within "
                                  f"{retrieval.SPREAD_TOLERANCE:g} of that of the best fit, has a normal-incidence "
                                  f"reflection factor {fit['spread']:.2f} away (median), so the band data do not fix "
                                  f"the impedance and a narrower absorber class may select between them.")
        extrapolated = self._absorption_extrapolation_message(bands, fraction, method, extrapolation, freq.min(),
                                                              freq.max(), root)
        if extrapolated is not None:
            messages["extrapolation"] = extrapolated
        # The object changes only here, once every value is computed, so a call that raises leaves it as it was.
        self._clear_stale_state()  # A new termination replaces whatever the stale results described.
        self.z = z
        self.z_angle = np.repeat(z[:, None], len(self.incidence_angle), axis=1)
        self._alpha = alpha_result
        self.matrix = {"termination": record}
        return messages

    @staticmethod
    def _absorption_extrapolation_message(bands, fraction, method, extrapolation, f_low, f_high, root="hard"):
        """
        Return the extrapolation sentence of ``impedance_from_absorption()``, or ``None`` when ``f_low`` to ``f_high``
        stays within one band of the outer band edges. The pointer to ``method='model'`` is left out for the soft root,
        which that method does not take.
        """
        edges = utils.band_edges(bands, fraction)
        step = utils.bandedge(10 ** (3 / 10), fraction) ** 2  # Ratio of neighbouring band edges.
        lowest, highest = edges[0, 0] / step, edges[-1, 2] * step
        if f_low >= lowest and f_high <= highest:
            return None
        if extrapolation == "model":
            rule = "the fitted model is evaluated there"
        elif method == "model":
            rule = "the impedance at the outer band edges is held there"
        else:
            rule = "the absorption of the outer bands is held there"
        # Limits rounded to 0.1 Hz towards the bands, so a grid that starts or ends there stays within one band.
        fmin = np.ceil(lowest * 10) / 10
        fmin = fmin + 0.1 if fmin < lowest else fmin
        fmax = np.floor(highest * 10) / 10
        fmax = fmax - 0.1 if fmax > highest else fmax
        if f_low > fmax or f_high < fmin:
            # Neither limit alone can be met, since the grid lies wholly beyond that range.
            side = "above" if f_low > fmax else "below"
            advice = (f"The grid lies wholly {side} the bands. Set fmin and fmax within {fmin:g} to {fmax:g} Hz to "
                      f"stay within one band.")
        else:
            hints = []
            if f_low < lowest:
                hints.append(f"fmin to at least {fmin:g} Hz")
            if f_high > highest:
                hints.append(f"fmax to at most {fmax:g} Hz")
            advice = f"Set {' and '.join(hints)} to stay within one band."
        if method != "model" and root == "hard":
            advice = advice[:-1] + ", or use method='model', which evaluates the fitted model outside the bands."
        return (f"The frequency grid ({f_low:g} to {f_high:g} Hz) extends more than one band beyond the bands "
                f"({edges[0, 0]:.0f} to {edges[-1, 2]:.0f} Hz), and {rule}. {advice}")

    @staticmethod
    def _full_range_angles(angle_limit=90.0):
        """
        Return the angles [deg] of the Sabine forward model of ``impedance_from_absorption()``, 0 to ``angle_limit`` at
        steps of at most 1 degree. A limit of 90 is replaced by the float just below it, as ``incidence_angle`` does for
        a range that ends at 90.
        """
        upper = np.nextafter(90.0, 0.0) if angle_limit >= 90 else float(angle_limit)
        return np.linspace(0.0, upper, int(np.ceil(angle_limit)) + 1)

    def _absorption_forward_model(self, incidence, sides, angle_limit=90.0):
        """
        Return the forward model ``forward(zeta, freq)`` of ``impedance_from_absorption()`` for one kind of data.

        For ``"sabine"`` it is ``tmm._impedance_retrieval.finite_sample_alpha`` with the field impedance of
        ``finite_sample_absorption()`` on a fixed grid, 0 to ``angle_limit`` [deg] at steps of at most 1 degree and
        ``SABINE_AZIMUTH_POINTS`` azimuth points, so the result does not depend on this treatment's ``incidence_angle``.
        """
        if incidence != "sabine":
            return retrieval.forward_model(incidence)
        theta = np.deg2rad(self._full_range_angles(angle_limit))
        nodes, weights = np.polynomial.legendre.leggauss(SABINE_AZIMUTH_POINTS)
        phi = np.pi / 4 * (nodes + 1)
        phi_weights = weights / 2
        cache = {}

        def forward(zeta, freq):
            freq = np.asarray(freq, dtype=float).reshape(-1)
            key = freq.tobytes()
            if key not in cache:
                cache.clear()
                cache[key] = self._thomasson_field_impedance_grid(float(sides[0]), float(sides[1]),
                                                                  tuple(float(k) for k in 2 * np.pi * freq / self.c0),
                                                                  tuple(float(t) for t in theta),
                                                                  tuple(float(p) for p in phi))
            return retrieval.finite_sample_alpha(np.asarray(zeta).reshape(-1), cache[key], theta, phi_weights)

        return forward

    def _real_absorption_impedance(self, bands, alpha, incidence, method, root, forward):
        """Return ``zeta_at(f)``, the real normalized impedance of ``method="real"`` or ``"55deg"`` at ``f`` [Hz]."""
        if method == "55deg":
            def zeta_at(f):
                admittance = retrieval.admittance_55_degrees(retrieval.bands_to_grid(bands, alpha, f), root)
                with np.errstate(divide="ignore"):
                    return 1 / admittance
            return zeta_at
        if incidence == "normal":
            return lambda f: retrieval.real_impedance_normal(retrieval.bands_to_grid(bands, alpha, f), root)
        if incidence == "random":
            return lambda f: retrieval.real_impedance_random(retrieval.bands_to_grid(bands, alpha, f), root)
        octaves = np.log2(bands[-1] / bands[0])
        grid = np.geomspace(bands[0], bands[-1], int(np.ceil(octaves * SABINE_NODES_PER_OCTAVE)) + 1)
        apart = np.min(np.abs(np.log2(grid[:, None] / bands[None, :])), axis=1) > 1e-3
        nodes = np.union1d(bands, grid[apart])
        node_zeta, _ = retrieval.real_impedance_inverse(retrieval.bands_to_grid(bands, alpha, nodes), nodes, forward,
                                                        root)
        if nodes.size == 1:
            return lambda f: np.full(np.shape(f), node_zeta[0])
        with np.errstate(invalid="ignore"):
            reflection = np.where(np.isinf(node_zeta), 1.0, (node_zeta - 1) / (node_zeta + 1))
        interpolant = PchipInterpolator(np.log2(nodes), reflection)

        def zeta_at(f):
            r = interpolant(np.clip(np.log2(np.asarray(f, dtype=float)), np.log2(nodes[0]), np.log2(nodes[-1])))
            with np.errstate(divide="ignore"):
                return np.where(r >= 1, np.inf, (1 + r) / (1 - r))
        return zeta_at

    def _rebuild_impedance_from_absorption(self, record, stacklevel=3):
        """
        Set the ``impedance_from_absorption()`` termination again from its record, on the current grid.

        A model fit is not repeated, since the stored parameters are evaluated. The warnings that depend on the data
        alone were given when the record was made, so only the extrapolation warning can come again, when the grid now
        extends more than one band beyond the bands and the grid checked then did not. ``stacklevel`` points that
        warning at the line that called ``rebuild()`` or ``compute()``.
        """
        bands = np.asarray(record["bands [Hz]"], dtype=float)
        fraction = int(record["fraction"])
        sample_size = record.get("sample_size [m]")
        checked = record.get("checked_range [Hz]")
        caller = "material_from_table" if "literature" in record else "impedance_from_absorption"
        messages = self._set_absorption_termination(bands,
                                                    np.asarray(record["alpha"], dtype=float),
                                                    record["incidence"],
                                                    record["method"],
                                                    record.get("absorber"),
                                                    record.get("root") or "hard",
                                                    None if sample_size is None else tuple(float(side)
                                                                                           for side in sample_size),
                                                    record["extrapolation"],
                                                    fraction,
                                                    parameters=record.get("parameters"),
                                                    angle_limit=self._record_angle_limit(record),
                                                    caller=caller)
        message = messages.get("extrapolation")
        if message is not None and (checked is None or self._absorption_extrapolation_message(
                bands, fraction, record["method"], record["extrapolation"], *checked) is None):
            warnings.warn(f"{caller}: {message[0].lower()}{message[1:]}", UserWarning, stacklevel=stacklevel)

    def _log_impedance_from_absorption(self, record):
        """Return the ``log_rebuild()`` lines of an ``impedance_from_absorption()`` termination."""
        def numbers(values):
            return "[" + ", ".join(self._log_number(value) for value in np.asarray(values, dtype=float)) + "]"

        args = [f"bands={numbers(record['bands [Hz]'])}", f"alpha={numbers(record['alpha'])}",
                f"incidence={record['incidence']!r}", f"method={record['method']!r}"]
        if record.get("absorber") is not None:
            args.append(f"absorber={record['absorber']!r}")
        if record.get("root") not in (None, "hard"):
            args.append(f"root={record['root']!r}")
        if record.get("sample_size [m]") is not None:
            args.append(f"sample_size={tuple(float(side) for side in record['sample_size [m]'])!r}")
        if self._record_angle_limit(record) is not None:
            args.append(f"angle_limit={self._log_number(self._record_angle_limit(record))}")
        default_extrapolation = "model" if record["method"] == "model" else "hold"
        if record.get("extrapolation") != default_extrapolation:
            args.append(f"extrapolation={record['extrapolation']!r}")
        args.append(f"fraction={int(record['fraction'])}")
        lines = []
        if record.get("parameters"):
            stored = ", ".join(f"{name!r}: {self._log_number(value)}" for name, value in record["parameters"].items())
            lines.append(f"# impedance_from_absorption() fits again; stored RMK+1 parameters: {{{stored}}}")
        lines.append(f"{self._log_variable()}.impedance_from_absorption({', '.join(args)})")
        return lines

    def material_from_table(self, row, *, incidence=None, method="real", absorber=None, sample_size=None,
                            angle_limit=None, scattering=None, structural_length=None):
        """
        Set a locally reacting surface impedance, and optionally a scattering coefficient, from literature table rows.

        The band values of the ``absorption_table()`` row ``row`` go to ``impedance_from_absorption()`` as octave bands,
        with ``method``, ``absorber``, ``sample_size`` and ``angle_limit``. Bands the source leaves empty are skipped,
        and so is the 6 kHz band of Kuttruff [1], Table 6.3, which lies off the octave series. Unless ``incidence`` is
        given, the row's ``kind`` selects it through ``LITERATURE_INCIDENCE``. Rows whose source states a
        reverberation-room measurement, at full or model scale, are Sabine data on a sample of sides ``sample_size``, a
        square of ``SABINE_SAMPLE_AREA`` when it is omitted. Rows compiled without a stated method are random-incidence
        values, so a compiled row printed above 1 raises ``ValueError`` unless ``incidence="sabine"`` is passed. A
        coefficient derived from the reverberation times of concert halls belongs to no sample size, so no finite-sample
        conversion is applied and the values are used as random-incidence values too. ``incidence``, with
        ``sample_size`` and ``angle_limit`` for ``"sabine"``, overrides these defaults. A row that gives an absorption
        area [m²] per person, per object or for a whole installation describes no surface and raises ``ValueError``.

        ``scattering`` names a ``scattering_table()`` row. Its values are interpolated onto ``self.freq`` the way
        ``impedance_from_absorption()`` interpolates absorption, by a monotone cubic in ``log2(f)`` through the band
        centres, held beyond the outer bands. [2] and [3] print band values only, so the interpolation is a package
        choice. A row given against the normalized frequency ``a/lambda`` of Vorländer [3], p. 311, needs the mean
        structural length ``a`` as ``structural_length``, and its bands lie at ``f = (a/lambda)*c0/a``. Below its lowest
        band such a row also keeps its lowest value, a package choice that departs from [3], p. 311, by which the
        coefficient below ``a/lambda = 0.125`` is generally smaller than 0.05. Values above 1 are limited to 1. The one
        ``UserWarning`` of the call says so. When every band of a row given against ``a/lambda`` lies above
        ``self.freq``, it adds that ``structural_length`` is in millimetres, and when every band lies below, that
        ``structural_length`` looks too large for a length in millimetres. Without ``scattering``, ``self.scat`` is
        ``None``.

        ``self.matrix["termination"]`` is the ``impedance_from_absorption()`` record with a ``literature`` entry. It
        holds the absorption row id, source, table, page, kind, material and printed values, the scattering row id with
        the band frequencies and values set, and the arguments of the call. ``show_layers()`` and the setup block of
        ``save2sheet()`` print the absorption row id, its ``source_id`` and the scattering row id, and
        ``absorption_table()`` gives the full citation of the source. ``rebuild()`` and ``load()`` set the impedance and
        the scattering coefficient again from this record on the current grid, without reading the tables, and
        ``log_rebuild()`` writes the ``material_from_table()`` call.

        Parameters
        ----------
        row : str
            ``id`` of an ``absorption_table()`` row.
        incidence : str, optional
            ``"normal"``, ``"random"`` or ``"sabine"``, overriding the kind of data the row's ``kind`` gives.
        method : str, optional
            ``"real"``, ``"55deg"`` or ``"model"``, as in ``impedance_from_absorption()``.
        absorber : str, optional
            Absorber class of ``method="model"``, as in ``impedance_from_absorption()``.
        sample_size : float or tuple of float, optional
            Sample side lengths ``(a, b)`` [m] of Sabine data. A single value gives a square. A side above
            ``SAMPLE_SIZE_LIMIT`` raises a ``ValueError``, since it is taken for a side in millimetres.
        angle_limit : float, optional
            Upper polar angle [deg] of the finite-sample integral of Sabine data. ``None`` gives 90 degrees, as in
            ``impedance_from_absorption()``, which documents the option.
        scattering : str, optional
            ``id`` of a ``scattering_table()`` row.
        structural_length : float, optional
            Mean structural length ``a`` [mm] of the surface, for a scattering row given against ``a/lambda``.

        References
        ----------
        [1] H. Kuttruff, "Room Acoustics", 6th ed., CRC Press, 2017.
        [2] T. J. Cox and P. D'Antonio, "Acoustic Absorbers and Diffusers: Theory, Design and Application", 3rd ed.,
            CRC Press, 2017.
        [3] M. Vorländer, "Auralization: Fundamentals of Acoustics, Modelling, Simulation, Algorithms and Acoustic
            Virtual Reality", Springer, Berlin, 2008.
        """
        entry = _literature_row(_literature_csv("absorption.csv"), row, "absorption_table", "row", "scattering.csv")
        if entry["quantity"] != "alpha":
            raise ValueError(f"Row {row!r} gives an equivalent absorption area in m² ({entry['quantity']}), not an "
                             f"absorption coefficient. Add it to a room's total absorption area instead.")
        default_incidence = LITERATURE_INCIDENCE[entry["kind"]]
        if incidence is None:
            incidence = default_incidence
        elif not isinstance(incidence, str) or incidence not in ABSORPTION_INCIDENCES:
            raise ValueError(f"incidence must be one of {ABSORPTION_INCIDENCES}, not "
                             f"{utils.describe_value(incidence)}.")
        # Checked before the record is built, which reads them.
        if sample_size is not None:
            utils.check_sample_size("material_from_table", sample_size, SAMPLE_SIZE_LIMIT)
        if structural_length is not None:
            structural_length = utils.check_positive("material_from_table", "structural_length",
                                                     "the mean structural length a [mm]", structural_length)
        bands = np.array(LITERATURE_ABSORPTION_BANDS, dtype=float)
        printed = entry[[f"a_{band}" for band in LITERATURE_ABSORPTION_BANDS]].to_numpy(dtype=float)
        used = np.isfinite(printed) & (bands != 6000)
        if incidence in ("normal", "random") and np.any(printed[used] > 1):
            above = ", ".join(f"{band:g}" for band in bands[used][printed[used] > 1])
            raise ValueError(f"Row {row!r} has values above 1 (at {above} Hz), which {incidence}-incidence absorption "
                             f"cannot reach. It builds with incidence='sabine', with sample_size if the size of the "
                             f"measured sample is known.")

        scat_bands = scat_values = scat = None
        extra = {}
        if scattering is not None:
            scat_entry = _literature_row(_literature_csv("scattering.csv"), scattering, "scattering_table",
                                         "scattering", "absorption.csv")
            scat_bands, scat_values = self._scattering_bands(scat_entry, structural_length)
            if np.any(scat_values > 1):
                extra["scattering_cap"] = (f"Scattering row {scattering!r} has values above 1, which are limited to "
                                           f"1.")
            scat_values = np.minimum(scat_values, 1.0)
            scat = self._literature_scattering_grid(scat_bands, scat_values)
            freq = np.asarray(self.freq, dtype=float)
            if (scat_entry["band_type"] == "a_over_lambda" and scat_bands.size
                    and np.all((scat_bands < freq.min()) | (scat_bands > freq.max()))):
                def hz(value):
                    return f"{float(f'{value:.3g}'):g}"  # Three significant digits, without an exponent below 1e6.
                # Bands below the grid come from a length that is too large; any other case may be one in metres.
                hint = ("structural_length looks too large for a length in millimetres"
                        if np.all(scat_bands < freq.min()) else "structural_length is in millimetres")
                extra["structural_length"] = (
                    f"The a/lambda bands of scattering row {scattering!r} lie at {hz(scat_bands.min())} to "
                    f"{hz(scat_bands.max())} Hz for structural_length={structural_length:g}, all outside the "
                    f"frequency grid ({freq.min():g} to {freq.max():g} Hz); {hint}.")
        elif structural_length is not None:
            raise ValueError("structural_length applies to a scattering row; pass scattering.")

        literature = {
            "row": row, "source": entry["source_id"], "table": entry["table"], "page": entry["page"],
            "kind": entry["kind"], "material": entry["material"],
            "printed_bands [Hz]": [float(band) for band in bands[np.isfinite(printed)]],
            "printed_alpha": [float(value) for value in printed[np.isfinite(printed)]],
            "scattering_row": scattering,
            "scattering_bands [Hz]": None if scat_bands is None else [float(band) for band in scat_bands],
            "scattering": None if scat_values is None else [float(value) for value in scat_values],
            "call": {"row": row, "method": method, "absorber": absorber,
                     "sample_size": None if sample_size is None else [float(side) for side in np.ravel(sample_size)],
                     "angle_limit": angle_limit,  # Replaced by the checked float when the termination is set.
                     "incidence": None if incidence == default_incidence else incidence,
                     "scattering": scattering,
                     "structural_length": None if structural_length is None else float(structural_length)},
        }
        self._impedance_from_absorption(bands[used], printed[used], incidence, method, absorber, "hard", sample_size,
                                        angle_limit, None, 1, "material_from_table", scat=scat, literature=literature,
                                        extra=extra)

    def _scattering_bands(self, entry, structural_length):
        """Return the band frequencies [Hz] and values of a ``scattering_table()`` row, without its empty bands."""
        if entry["band_type"] == "a_over_lambda":
            if structural_length is None:
                raise ValueError(f"Scattering row {entry.name!r} is given against a/lambda; pass structural_length, "
                                 f"the mean structural length a [mm] of the surface.")
            columns = [f"s_al_{label}" for label in LITERATURE_SCATTERING_A_OVER_LAMBDA]
            labels = np.array([float(label) for label in LITERATURE_SCATTERING_A_OVER_LAMBDA])
            with np.errstate(over="ignore"):
                bands = labels * self.c0 / (structural_length / 1000)
            if not np.all(np.isfinite(bands)):
                raise ValueError(f"structural_length={structural_length!r} mm is too small: the a/lambda bands of "
                                 f"{entry.name!r} lie at infinite frequencies.")
        else:
            if structural_length is not None:
                raise ValueError(f"structural_length applies to rows given against a/lambda; {entry.name!r} is in "
                                 f"frequency bands.")
            columns = [f"s_{band}" for band in LITERATURE_SCATTERING_BANDS]
            bands = np.array(LITERATURE_SCATTERING_BANDS, dtype=float)
        values = entry[columns].to_numpy(dtype=float)
        given = np.isfinite(values)
        return bands[given], values[given]

    def _literature_scattering_grid(self, bands, values):
        """
        Return band values interpolated onto ``self.freq``, or ``None`` when there are none. The interpolated values are
        limited to 0 to 1, since the cubic through values of 1 can exceed 1 by a rounding error.
        """
        if bands is None:
            return None
        return np.clip(retrieval.bands_to_grid(np.asarray(bands, dtype=float), np.asarray(values, dtype=float),
                                               np.asarray(self.freq, dtype=float)), 0.0, 1.0)

    def _set_literature_scattering(self, bands, values):
        """Set ``self.scat`` from band values on ``self.freq``, or to ``None`` when there are none."""
        self.scat = self._literature_scattering_grid(bands, values)

    def _rebuild_material_from_table(self, record, stacklevel=4):
        """
        Set a ``material_from_table()`` termination again from its record on the current grid, without the tables.
        ``stacklevel`` is that of the extrapolation warning.
        """
        literature = record["literature"]
        self._rebuild_impedance_from_absorption(record, stacklevel=stacklevel)
        self._set_literature_scattering(literature["scattering_bands [Hz]"], literature["scattering"])
        self.matrix["termination"]["literature"] = literature

    def _log_material_from_table(self, record):
        """Return the ``log_rebuild()`` lines of a ``material_from_table()`` termination."""
        call = record["literature"]["call"]
        args = [repr(str(call["row"]))]
        for name, default in (("incidence", None), ("method", "real"), ("absorber", None), ("sample_size", None),
                              ("angle_limit", None), ("scattering", None), ("structural_length", None)):
            value = call.get(name)
            if value is None or (default is not None and value == default):
                continue
            if name == "sample_size":
                args.append(f"sample_size={tuple(float(side) for side in np.atleast_1d(value))!r}")
            elif name in ("angle_limit", "structural_length"):
                args.append(f"{name}={self._log_number(value)}")
            else:
                args.append(f"{name}={str(value)!r}")
        lines = []
        if record.get("parameters"):
            stored = ", ".join(f"{name!r}: {self._log_number(value)}" for name, value in record["parameters"].items())
            lines.append(f"# material_from_table() fits again; stored RMK+1 parameters: {{{stored}}}")
        lines.append(f"{self._log_variable()}.material_from_table({', '.join(args)})")
        return lines

    def membrane_layer(self, t=1, rho=None, layer=None, *, mechanical_resistance=0, preset=None):
        """
        Add a limp membrane or mass-sheet layer.

        The membrane is modeled as a local pressure-jump impedance with surface mass ``m_s = rho*t`` and specific
        impedance ``Z_m = r_m*rho0*c0 + 1j*omega*m_s``, where ``t`` is converted from millimeters to meters and ``r_m``
        is the mechanical loss resistance normalized by ``rho0*c0``. The transfer matrix is ``[[1, Z_m/S0], [0, 1]]``
        because this class uses pressure and volume velocity as the matrix state; ``S0`` is the front surface area. The
        resulting surface impedance recovered by ``compute()`` is scaled back to specific impedance.

        With ``r_m = 0`` this is the mass term of the classical limp-membrane absorber model. Brandão [1],
        section 2.3.2, writes ``Z_m = 1j*omega*m_s`` after neglecting membrane resistance and combines it in series with
        the surface impedance of the backing cavity, which here comes from the layers behind the membrane. Kang and
        Fuchs [2], equation 1, keep the resistance in the normalized impedance ``z_M = r' + 1j*omega*m'/(rho0*c0)``,
        where ``r'`` "depends mainly on mounting conditions" and is around 1 in their calculations. Sakagami et al. [3]
        call it the mechanical loss resistance. ``r_m`` is ``r'``.

        ``r_m`` is the only loss of the sheet's motion. It sets the height and width of the absorption peak at the
        resonance of the membrane mass with the layers behind it. At that resonance a membrane over a rigidly backed air
        space has the impedance ``r_m*rho0*c0``, so ``r_m = 1`` gives the highest peak at normal incidence ([3]).
        Without it, a membrane over a lossless air space absorbs nothing.

        The model neglects bending stiffness and finite-panel modal behavior. It is appropriate for a limp
        local-reaction mass sheet, not a full structural panel model. ``thin_plate_layer`` adds the bending stiffness of
        an infinite plate, which changes the impedance at oblique incidence near and above the critical frequency.

        ``preset`` takes ``rho`` from the ``thin_plate_layer`` presets listed by ``plate_presets()``; the other columns
        do not enter a membrane. A ``rho`` passed with it overrides the preset value with one ``UserWarning``, and the
        record keeps the change as ``preset_overrides``; ``rebuild()`` and ``load()`` do not repeat it. The record
        stores the resolved density, so ``rebuild()``, ``load()`` and ``log_rebuild()`` do not depend on the preset
        table.

        Parameters
        ----------
        t : float or int, optional
            Thickness of the membrane [mm], above 0.
        rho : float, optional
            Material density [kg/m³], above 0. Defaults to the preset value. Without a preset the default is 7850, mild
            steel, Bies and Hansen [4], table B.1.
        layer : None or int, optional
            Optional layer index, an integer or a float with an integer value. If ``None``, the layer is appended.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the membrane motion, mainly from mounting losses, normalized by
            ``rho0*c0`` [-], at least 0. Values outside these ranges raise a ``ValueError``.
        preset : str, optional
            Name of a material from ``plate_presets()``.

        References
        ----------
        [1] E. Brandão, "Acústica de Salas: Projeto e Modelagem", Blucher, São Paulo, 2018.
        [2] J. Kang and H. V. Fuchs, "Predicting the absorption of open weave textiles and micro-perforated
            membranes backed by an air space", Journal of Sound and Vibration, vol. 220, pp. 905-920, 1999.
        [3] K. Sakagami, M. Morimoto and M. Yairi, "A note on the relationship between the sound absorption by
            microperforated panels and panel/membrane-type absorbers", Applied Acoustics, vol. 70, pp. 1131-1136, 2009.
        [4] D. A. Bies and C. H. Hansen, "Engineering Noise Control: Theory and Practice", 4th ed., Spon Press, 2009.
        """
        values, preset_overrides = self._resolve_preset(PLATE_PRESETS, preset, "membrane_layer", {"rho": rho})
        if preset is not None:
            self._require_preset_values("membrane_layer", preset, values, ("rho",))
        layer = utils.check_index("membrane_layer", layer)
        rho = 7850 if values["rho"] is None else values["rho"]
        t = utils.check_thickness("membrane_layer", t)
        rho = utils.check_positive("membrane_layer", "rho", "the material density [kg/m³]", rho)
        mechanical_resistance = self._check_mechanical_resistance("membrane_layer", mechanical_resistance)
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

        self._warn_if_terminated("membrane_layer")
        self.matrix[layer] = {"type": "membrane_layer",
                              "thickness [mm]": t,
                              "density [kg/m³]": rho,
                              "preset": preset,
                              "preset_overrides": preset_overrides,
                              "mechanical_resistance [-]": mechanical_resistance or None,
                              "matrix": Tm,
                              }

    def thin_plate_layer(self, t=12.5, rho=None, E=None, nu=None, loss_factor=None, layer=None, *,
                         mechanical_resistance=0, preset=None):
        """
        Add an infinite thin plate with bending stiffness as a series-impedance layer.

        The plate is a Kirchhoff plate with surface mass ``m = rho*t`` and bending stiffness
        ``D = E*t**3/(12*(1 - nu**2))``, where ``t`` and ``E`` are converted to meters and pascals. At the trace
        wavenumber ``kx = k0*sin(theta)`` of the incident wave its specific impedance is
        ``Z_w = eta*A + 1j*(omega*m - A)`` with ``A = D*kx**4/omega`` and ``eta`` the loss factor, Vigran [1],
        section 6.5.1.2, equation 6.90 with the complex stiffness ``D*(1 + 1j*eta)``. With the critical frequency
        ``f_c = c0**2/(2*pi)*sqrt(m/D)`` this is ``1j*omega*m*(1 - (f/f_c)**2*(1 + 1j*eta)*sin(theta)**4)``, Vigran [1],
        equation 6.91, and Mechel [2], section I.8, equation 11. The transfer matrix is ``[[1, Z/S0], [0, 1]]`` with
        ``Z = Z_w + r_m*rho0*c0``. ``kx`` is the same in every layer, so ``A`` does not depend on the media next to the
        plate. The mounting resistance ``r_m`` is the membrane resistance ``r'`` of Kang and Fuchs [3], equation 1,
        added to the plate impedance as a package extension.

        Compared with ``membrane_layer``, the plate adds the bending term ``A``. At normal incidence ``A = 0`` and the
        layer is identical to ``membrane_layer`` with the same ``t``, ``rho`` and ``mechanical_resistance``. At oblique
        incidence the stiffness lowers the mass reactance by the fraction ``(f/f_c)**2*sin(theta)**4``, which is small
        well below ``f_c``. At ``f = f_c/sin(theta)**2`` the reactance vanishes (coincidence) and the impedance is
        ``eta*omega*m + r_m*rho0*c0``. Above it the stiffness dominates. ``eta`` acts only through ``A``, so it does not
        damp the plate at normal incidence or well below ``f_c``. A resonant panel absorber still needs
        ``mechanical_resistance`` or a damped cavity. Neither layer models finite size, edge conditions or panel modes.

        The defaults describe the natural-gypsum plasterboard of Hopkins [4], table A2. ``rho`` and ``loss_factor`` are
        the printed values, and so is ``nu``, which [4] marks as an estimate. ``E`` follows from the printed
        quasi-longitudinal phase velocity ``c_L = 1490 m/s`` and the plate relation ``c_L = sqrt(E/(rho*(1 - nu**2)))``
        of [4], equation 2.21, rounded to 1.74 GPa. ``t = 12.5`` is the plasterboard thickness of [4], figure 2.2.

        ``preset`` fills ``rho``, ``E``, ``nu`` and ``loss_factor`` with the constants of one material, and any of these
        passed explicitly overrides the preset value. An override that differs from the preset issues one
        ``UserWarning`` listing each changed value, and the record keeps that list as ``preset_overrides``;
        ``rebuild()`` and ``load()`` do not repeat it. A field the preset leaves empty must be passed, or a
        ``ValueError`` names it. The loss factor of a preset is the internal loss factor of the material, since ``eta``
        acts on the bending stiffness only. Where a source gives none, prints it only as a range, or does not say that
        it is the internal loss factor, the preset carries none. Names give the material and its density. The record
        stores the resolved values, so ``rebuild()``, ``load()`` and ``log_rebuild()`` do not depend on the preset
        table. ``plate_presets()`` lists each material with its values, source, status and description. Without a preset
        the defaults are those of ``plasterboard_860``.

        Parameters
        ----------
        t : float, optional
            Plate thickness [mm], above 0. The default is the plasterboard thickness of [4] and also applies with a
            preset, since presets hold no thickness.
        rho : float, optional
            Plate density [kg/m³], above 0. Defaults to the preset value, or 860 without a preset.
        E : float, optional
            Young's modulus [GPa], above 0. Defaults to the preset value, or 1.74 without a preset.
        nu : float, optional
            Poisson's ratio [-], above -1 and below 0.5. Defaults to the preset value, or 0.3 without a preset.
        loss_factor : float, optional
            Loss factor ``eta`` of the bending stiffness [-], at least 0. Defaults to the preset value, or 0.0141
            without a preset.
        layer : None or int, optional
            Optional layer index, an integer or a float with an integer value. If ``None``, the layer is appended.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the plate motion, mainly from mounting losses, normalized by
            ``rho0*c0`` [-], at least 0. Values outside these ranges raise a ``ValueError``.
        preset : str, optional
            Name of a material listed by ``plate_presets()``.

        Notes
        -----
        Thin-plate theory holds while the bending wavelength is longer than six plate thicknesses ([1], section 3.7.3.1;
        [5], section 5.2.7.1). A ``UserWarning`` is issued when the frequency range goes above this limit and an
        incidence angle is oblique. At normal incidence the bending term is zero and the limit does not apply.

        With a small loss factor and no other loss in the stack, the mass-spring resonance behind the plate is narrow
        and moves with the angle of incidence. Diffuse results then depend on the ``incidence_angle`` step, as they do
        for ``membrane_layer`` with a small resistance. A finer step or a porous layer in the cavity resolves the
        resonance, and a larger ``mechanical_resistance`` widens it.

        References
        ----------
        [1] T. E. Vigran, "Building Acoustics", Taylor & Francis, 2008.
        [2] F. P. Mechel (ed.), "Formulas of Acoustics", 2nd ed., Springer, 2008.
        [3] J. Kang and H. V. Fuchs, "Predicting the absorption of open weave textiles and micro-perforated
            membranes backed by an air space", Journal of Sound and Vibration, vol. 220, pp. 905-920, 1999.
        [4] C. Hopkins, "Sound Insulation", Butterworth-Heinemann, 2007.
        [5] T. E. Vigran, "Manual for NorFLAG, version 4.0", Trondheim, 2018.
        """
        resolved, preset_overrides = self._resolve_preset(
            PLATE_PRESETS, preset, "thin_plate_layer", {"rho": rho, "E": E, "nu": nu, "loss_factor": loss_factor})
        if preset is not None:
            self._require_preset_values("thin_plate_layer", preset, resolved, PLATE_PARAMETERS)
        defaults = {"rho": 860, "E": 1.74, "nu": 0.3, "loss_factor": 0.0141}
        rho, E, nu, loss_factor = (defaults[name] if resolved[name] is None else resolved[name]
                                   for name in PLATE_PARAMETERS)
        layer = utils.check_index("thin_plate_layer", layer)
        t = utils.check_thickness("thin_plate_layer", t)
        rho = utils.check_positive("thin_plate_layer", "rho", "the plate density [kg/m³]", rho)
        E = utils.check_positive("thin_plate_layer", "E", "Young's modulus [GPa]", E)
        nu = utils.check_interval("thin_plate_layer", "nu", "Poisson's ratio [-]", nu, -1, 0.5, closed=(False, False))
        loss_factor = utils.check_non_negative("thin_plate_layer", "loss_factor",
                                               "the loss factor of the bending stiffness [-]", loss_factor)
        mechanical_resistance = self._check_mechanical_resistance("thin_plate_layer", mechanical_resistance)

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

        self._warn_if_terminated("thin_plate_layer")
        self.matrix[layer] = {"type": "thin_plate_layer",
                              "thickness [mm]": t,
                              "density [kg/m³]": rho,
                              "young_modulus [GPa]": E,
                              "poisson_ratio [-]": nu,
                              "loss_factor [-]": loss_factor,
                              "preset": preset,
                              "preset_overrides": preset_overrides,
                              "mechanical_resistance [-]": mechanical_resistance or None,
                              "critical_frequency [Hz]": float(self.c0 ** 2 / (2 * np.pi) * np.sqrt(mass / stiffness)),
                              "thin_plate_limit [Hz]": float(f_thin),
                              "checked_fmax [Hz]": checked_fmax,
                              "matrix": Tm,
                              }

    def porous_facing_layer(self, t=None, surface_mass=None, resistance=None, porosity=None, tortuosity=None,
                            method=None, free=True, rho=None, d=None, s=None, open_area=None,
                            layer=None, *, in_contact=False, preset=None):
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
        circular-pore expression ``32*eta*t/(open_area*d**2)``. Without a preset, ``method="rebillard"`` with neither
        uses a nominal thin facing with ``surface_mass=0.25`` kg/m² and ``resistance=200`` Pa*s/m. A value outside the
        range its entry below gives raises a ``ValueError``, whether or not the branch reads it.

        ``preset`` fills ``t``, ``surface_mass``, ``resistance``, ``porosity`` and ``tortuosity`` with the values of one
        fabric or screen sample, and any of these passed explicitly overrides the preset value. An override that differs
        from the preset issues one ``UserWarning`` listing each changed value, and the record keeps that list as
        ``preset_overrides``; ``rebuild()`` and ``load()`` do not repeat it. Without ``method``, a preset selects the
        branch in its ``method`` column. A field that the chosen branch reads and the preset leaves empty, such as the
        ``porosity`` of a ``"pieren"`` row used with ``method="rebillard"``, must be passed, or a ``ValueError`` names
        it. A preset never falls back on the nominal facing. Fields the branch does not read take the defaults below, so
        a ``"pieren"`` row without a thickness records ``t = 1``. The record stores the resolved values, so
        ``rebuild()``, ``load()`` and ``log_rebuild()`` do not depend on the preset table. ``fabric_presets()`` lists
        each sample with its values, source, status and description.

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
        t : float, optional
            Facing thickness [mm], above 0. Defaults to the preset value, or 1 without a preset or where the preset
            gives none and the branch does not read it.
        surface_mass : float, optional
            Mass per unit area [kg/m²], above 0. Defaults to the preset value. If ``None`` and ``rho`` is supplied, it
            is estimated as ``rho*t``.
        resistance : float, optional
            Specific airflow resistance [Pa*s/m], at least 0. Defaults to the preset value. If ``None``, it is estimated
            from ``d`` and ``open_area``.
        porosity : float, optional
            Open porosity used by the Rebillard model, above 0 and at most 1. Defaults to the preset value, or 0.7.
        tortuosity : float, optional
            Tortuosity used by the Rebillard model, at least 1. Defaults to the preset value, or 1.
        method : str, optional
            ``"rebillard"``, ``"pieren"``, ``"pieren_free"``, or ``"pieren_fixed"``, in any letter case. Defaults to the
            preset's ``method``, or ``"rebillard"`` without a preset.
        free : bool, optional
            For ``method="pieren"``, choose the free-moving fabric expression when ``True`` or the motionless
            resistance-only limit when ``False``.
        rho : float, optional
            Apparent fabric density [kg/m³], above 0, used only to estimate ``surface_mass`` when needed.
        d : float, optional
            Equivalent pore diameter [mm], above 0, used only to estimate ``resistance``.
        s : float, optional
            Equivalent pore spacing [mm] of pores on a square pitch, above 0 and at least ``d``, used to estimate
            ``open_area`` as ``pi*d**2/(4*s**2)`` when needed.
        open_area : float, optional
            Open-area ratio used for the geometric resistance estimate, above 0 and at most 1.
        layer : None or int, optional
            Optional layer index, an integer or a float with an integer value. If ``None``, the layer is appended.
        in_contact : bool, optional
            If ``True``, the sheet rests on the plate directly before or after it, within about one hole diameter of its
            openings, and acts at the opening velocity, as described above.
        preset : str, optional
            Name of a fabric or screen listed by ``fabric_presets()``.

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
        values, preset_overrides = self._resolve_preset(
            FABRIC_PRESETS, preset, "porous_facing_layer",
            dict(zip(FABRIC_PARAMETERS, (t, surface_mass, resistance, porosity, tortuosity))))
        layer = utils.check_index("porous_facing_layer", layer)
        in_contact = utils.check_flag("porous_facing_layer", "in_contact", in_contact)
        free = utils.check_flag("porous_facing_layer", "free", free)
        if method is None:
            method = "rebillard" if preset is None else FABRIC_PRESETS[preset]["method"]
        if isinstance(method, str):
            method = method.lower()  # The branch names are matched in any letter case, as they always were.
        if not isinstance(method, str) or method not in FACING_METHODS:
            raise ValueError(f"porous_facing_layer method must be 'rebillard', 'pieren', 'pieren_free', or "
                             f"'pieren_fixed', not {utils.describe_value(method)}.")
        if method == "pieren_free":
            method = "pieren"
            free = True
        elif method == "pieren_fixed":
            method = "pieren"
            free = False

        if preset is not None:
            # The fields this branch reads must come from the preset or the call, never from the defaults below.
            rebillard = method == "rebillard"
            mass_estimate = values["surface_mass"] is None and rho is not None
            resistance_estimate = values["resistance"] is None and d is not None
            needs = {"t": rebillard or mass_estimate or resistance_estimate,
                     "surface_mass": (rebillard or free) and rho is None,
                     "resistance": d is None,
                     "porosity": rebillard,
                     "tortuosity": rebillard}
            self._require_preset_values("porous_facing_layer", preset, values,
                                        [name for name in FABRIC_PARAMETERS if needs[name]])
        defaults = {"t": 1, "surface_mass": None, "resistance": None, "porosity": 0.7, "tortuosity": 1.0}
        t, surface_mass, resistance, porosity, tortuosity = (defaults[name] if values[name] is None else values[name]
                                                             for name in FABRIC_PARAMETERS)

        name = "porous_facing_layer"
        t = utils.check_thickness(name, t, what="the facing thickness [mm]")
        surface_mass, rho, d, s = (None if value is None else utils.check_positive(name, key, what, value)
                                   for key, what, value in (
                                       ("surface_mass", "the mass per unit area [kg/m²]", surface_mass),
                                       ("rho", "the apparent fabric density [kg/m³]", rho),
                                       ("d", "the equivalent pore diameter [mm]", d),
                                       ("s", "the equivalent pore spacing [mm]", s)))
        if d is not None and s is not None:
            note = " Wider pores overlap on the square pitch; touching pores, d = s, are accepted."
            utils.check_at_most(name, "d", "the equivalent pore diameter [mm]", d, "s", "the equivalent pore spacing",
                                s, "mm", note)
        if resistance is not None:
            resistance = utils.check_non_negative(name, "resistance", "the specific airflow resistance [Pa*s/m]",
                                                  resistance)
        if open_area is not None:
            open_area = utils.check_fraction(name, "open_area", "the open-area ratio", open_area)
        porosity = utils.check_fraction(name, "porosity", "the open porosity of the Rebillard model", porosity)
        tortuosity = utils.check_tortuosity(name, "tortuosity", "the tortuosity of the Rebillard model", tortuosity)
        t_meters = t / 1000

        if method == "rebillard" and preset is None:
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
            if open_area is None:
                if s is None:
                    raise ValueError("open_area or s must be supplied when estimating resistance.")
                s_meters = s / 1000
                open_area = self._facing_open_area(d, s, d_meters, s_meters)
            resistance = 32 * self.air_prop["air_viscosity"] * t_meters / (open_area * d_meters**2)
        elif open_area is None and d is not None and s is not None:
            d_meters = d / 1000
            s_meters = s / 1000
            open_area = self._facing_open_area(d, s, d_meters, s_meters)

        if method == "pieren":
            if free and surface_mass is None:
                raise ValueError("surface_mass or rho must be supplied for a free Pieren fabric.")
        elif surface_mass is None:
            raise ValueError("surface_mass or rho must be supplied for the Rebillard porous-facing model.")

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
                       "preset": preset,
                       "preset_overrides": preset_overrides,
                       "rho [kg/m3]": rho,
                       "pore_diameter [mm]": d,
                       "pore_spacing [mm]": s,
                       "open_area [%]": None if open_area is None else open_area * 100,
                       "in_contact": True if in_contact else None,
                       "matrix": Tf,
                       }
        if method == "pieren":
            matrix_data["free"] = free
        self._warn_if_terminated("porous_facing_layer")
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

    @staticmethod
    def _facing_open_area(d, s, d_meters, s_meters):
        """
        Return the open area ``pi*d**2/(4*s**2)`` that ``porous_facing_layer`` estimates for circular pores of diameter
        ``d`` on a square pitch ``s`` [mm], which the layer has checked for ``d <= s``.
        """
        return np.pi / ((2 * s_meters / d_meters) ** 2)

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
            label = utils.layer_label(self.matrix, key)
            if not neighbours:
                raise ValueError(f"{label} is a porous_facing_layer with in_contact=True, but no "
                                 "perforated_panel_layer, slotted_panel_layer or rectangular_slot_layer lies directly "
                                 "before or after it.")
            if len(neighbours) > 1:
                raise ValueError(f"{label} is a porous_facing_layer with in_contact=True between two plates; place "
                                 "it directly next to the one plate it rests on.")
            plates.add(neighbours[0])
        return plates

    @staticmethod
    def _check_mechanical_resistance(method, mechanical_resistance):
        """Raise a ``ValueError`` unless the mechanical loss resistance is one finite number of at least 0."""
        return utils.check_non_negative(method, "mechanical_resistance",
                                        "the mechanical loss resistance normalized by rho0*c0 [-]",
                                        mechanical_resistance)

    @staticmethod
    def _validate_mechanical_resistance(method, mechanical_resistance, rho):
        """Return the mechanical resistance as a float; raise if it is not a finite number of at least 0 or is set on a
        rigid plate."""
        mechanical_resistance = TMM._check_mechanical_resistance(method, mechanical_resistance)
        if mechanical_resistance and not rho:
            raise ValueError("mechanical_resistance acts on the plate motion and needs rho; a rigid plate does not "
                             "move.")
        return mechanical_resistance

    @staticmethod
    def _circular_hole_end_length(d_meters, open_area, end_correction):
        """
        Return the total end-correction length of both hole ends [m] for ``perforated_panel_layer``.
        """
        xi = np.sqrt(open_area)
        if end_correction == "allard_atalla":
            # Allard and Atalla (2009), equation 9.17, one side; both ends are equal.
            return 2 * TMM._allard_atalla_end_length(d_meters, open_area)
        if end_correction == "ingard":
            # Vigran's fit to Ingard's circular aperture in a square tube, both ends: 0.96*sqrt(A0)*(...), evaluated at
            # the square root of the open area as the NorFLAG manual, annex J, states it.
            return 0.96 * np.sqrt(np.pi * d_meters ** 2 / 4) * (1 - 1.24 * xi + 0.27 * xi ** 2)
        if end_correction == "nesterov":
            # Cox and D'Antonio, Table 7.1: circular holes in a circular pattern.
            return 0.85 * d_meters * (1 - 1.47 * xi + 0.47 * xi ** 3)
        if end_correction in {"jb", "jaouen_becot"}:
            # Jaouen and Bécot (2011), equation 4: 8*r/(3*pi)*(...) per side, with xi = 2*sqrt(phi/pi) = d/s. The
            # polynomial turns negative only for overlapping holes (xi > 1), where the end length is set to zero.
            xi = 2 * np.sqrt(open_area / np.pi)
            return max(8 * d_meters / (3 * np.pi) * (1 - 1.13 * xi - 0.09 * xi ** 2 + 0.27 * xi ** 3), 0.0)
        if end_correction == "beranek":
            # Single circular hole in an infinite baffle.
            return 0.85 * d_meters
        raise ValueError(f"perforated_panel_layer end_correction must be 'allard_atalla', 'ingard', 'jb', "
                         f"'jaouen_becot', 'nesterov', or 'beranek', not {utils.describe_value(end_correction)}.")

    @staticmethod
    def _allard_atalla_end_length(d_meters, open_area):
        """
        Return the one-side added length [m] of circular holes of diameter ``d_meters`` [m] in a square lattice with
        open-area ratio ``open_area``, from Allard and Atalla (2009), equation 9.17.

        The series is summed once per open area by ``_allard_atalla_factor``. Its cache key is ``open_area`` rounded to
        12 significant digits, so a layer that ``rebuild()`` re-creates from the stored equivalent spacing finds the sum
        of the original call.
        """
        rounded = float(f"{float(open_area):.12g}")
        if not ALLARD_ATALLA_MIN_OPEN_AREA <= rounded <= np.pi / 4:  # NaN fails this test too.
            if rounded < ALLARD_ATALLA_MIN_OPEN_AREA:
                raise ValueError(f"end_correction='allard_atalla' needs an open area of at least "
                                 f"{100 * ALLARD_ATALLA_MIN_OPEN_AREA:g} %, not {100 * float(open_area):.4g} %. Use "
                                 f"end_correction='jb' for smaller open areas.")
            if rounded > np.pi / 4:
                raise ValueError("end_correction='allard_atalla' needs an open area of at most pi/4 (78.5 %), where "
                                 f"the holes of its square cell touch, not {100 * float(open_area):.4g} %.")
            raise ValueError(f"end_correction='allard_atalla' needs an open area from "
                             f"{100 * ALLARD_ATALLA_MIN_OPEN_AREA:g} % to pi/4 (78.5 %), not {float(open_area)!r}.")
        return float(d_meters) * TMM._allard_atalla_factor(rounded)

    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _allard_atalla_factor(open_area):
        """
        Return the one-side added length of ``_allard_atalla_end_length`` divided by the hole diameter.

        With hole radius ``R``, lattice period ``D`` and ``R/D = sqrt(open_area/pi)``, the length is
        ``D*sum'(v_mn*2*J1(2*pi*(R/D)*q)**2/(pi**2*q**3))`` over ``m, n >= 0`` with ``q = sqrt(m**2 + n**2)``,
        ``(m, n) != (0, 0)``, and ``v_mn`` from equation 9.6: ``1/2`` when one index is zero, else ``1``. The truncation
        is set by ``ALLARD_ATALLA_ARGUMENT`` and ``ALLARD_ATALLA_MIN_INDEX``.
        """
        ratio = np.sqrt(open_area / np.pi)
        q_max = max(float(ALLARD_ATALLA_MIN_INDEX), ALLARD_ATALLA_ARGUMENT / (2 * np.pi * ratio))
        n = np.arange(int(q_max) + 1, dtype=float)
        total = 0.0
        for m in range(int(q_max) + 1):
            n_row = n[:int(np.sqrt(q_max ** 2 - m ** 2)) + 1]
            if m == 0:
                n_row = n_row[1:]
            q = np.hypot(m, n_row)
            weight = np.where(n_row > 0, 1.0, 0.5) * (1.0 if m > 0 else 0.5)
            total += np.sum(weight * 2 * special.j1(2 * np.pi * ratio * q) ** 2 / (np.pi ** 2 * q ** 3))
        edge = 2 * np.pi * ratio * q_max
        total += ratio / (np.pi * edge ** 2) * (1 - np.cos(2 * edge) / edge)
        # D = d/(2*R/D); divided by d this leaves 1/(2*R/D).
        return float(total / (2 * ratio))

    @staticmethod
    def _validate_porous_contact(method, porous_contact):
        """Raise a ``ValueError`` unless the porous-contact factor is one finite number from 0 to 1."""
        return utils.check_interval(method, "porous_contact", "the porous-contact factor K [-]", porous_contact, 0, 1)

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
        if not isinstance(exit_radiation, str) or exit_radiation not in {"air", "porous", "mean"}:
            raise ValueError(f"exit_radiation must be 'air', 'porous', or 'mean', not "
                             f"{utils.describe_value(exit_radiation)}.")
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
        ``open_area`` is supplied, ``s`` is replaced by the equivalent square pitch for metadata. A hole diameter above
        the pitch, where the holes overlap, raises a ``ValueError``; for the square pitch that is an open area above
        ``pi/4``. Touching holes, ``d = s``, are accepted. A value outside the range its entry under Parameters gives
        raises a ``ValueError`` too, whether or not the method reads it.

        The ``end_correction`` options set ``Delta``, the end-correction length of both hole ends. ``method="barrier"``
        and ``method="eq_fluid"`` use it as ``t_eff = t + Delta``, and ``method="bessel_ingard"`` places it inside the
        Bessel factor. The layer record keeps ``Delta`` of both ends together as ``end_correction_length [mm]``, which
        ``show_layers()`` prints and ``rebuild()`` computes again from the geometry. ``method="barrier_mpp"`` and
        ``method="conical"`` have no ``Delta`` and store ``None``.

        ``"jb"`` / ``"jaouen_becot"`` is the fit of Jaouen and Bécot [10], equation 4,
        ``Delta = 8*d/(3*pi)*(1 - 1.13*xi - 0.09*xi**2 + 0.27*xi**3)`` with ``xi = 2*sqrt(phi/pi)``, which is ``d/s``
        for a square pitch. [10] fits the general expression for circular holes in a square pattern that it attributes
        to Ingard's "Notes on Sound Absorption Technology" (1994). The polynomial turns negative only for overlapping
        holes, which the method refuses; the end-length helper sets ``Delta`` to zero there. [1], table 7.1, prints the
        fit with ``0.85`` for ``8/(3*pi)`` and ``sqrt(phi)`` for ``xi``.

        ``"allard_atalla"`` is the uniform-velocity series of Allard and Atalla [11], equation 9.17, for a circular hole
        of radius ``R`` in a square cell of side ``D`` with ``R/D = sqrt(phi/pi)``. Both ends together give
        ``Delta = 2*D*sum'(v_mn*2*J1(2*pi*(R/D)*q)**2/(pi**2*q**3))`` with ``q = sqrt(m**2 + n**2)``, summed over
        ``m, n >= 0`` without ``(m, n) = (0, 0)``. ``v_mn`` is ``1/2`` when one index is zero and ``1`` otherwise ([11],
        equation 9.6). The series uses the wavenumbers of [11], equation 9.15, which hold well below ``c0/D``. An
        ``open_area`` given for another hole pattern is treated as a square cell of the same open area. The module
        constants ``ALLARD_ATALLA_ARGUMENT`` and ``ALLARD_ATALLA_MIN_INDEX`` set where the series is cut. An
        ``open_area`` below ``ALLARD_ATALLA_MIN_OPEN_AREA``, ``1e-5`` (0.001 %), or above ``pi/4``, where the holes of
        the square cell overlap, raises a ``ValueError``. The approximation of the series in [11], equation 9.18, is not
        offered.

        ``"ingard"`` is Vigran's fit [4], annex J, to Ingard's end correction for a circular aperture in a square tube
        [3], ``Delta = 0.96*sqrt(A0)*(1 - 1.24*xi + 0.27*xi**2)`` with ``A0 = pi*a**2``. The method evaluates it as [4]
        states it, with ``xi = sqrt(phi)``. Ingard's variable for this geometry is ``d/s`` ([3], table I), so at equal
        ``phi`` the ``"ingard"`` length is longer than the ``"jb"`` and ``"allard_atalla"`` lengths.

        ``"nesterov"`` is the circular-pattern correction ``Delta = 0.85*d*(1 - 1.47*sqrt(phi) + 0.47*phi**1.5)`` listed
        in [1], table 7.1, and ``"beranek"`` is the single-hole infinite-baffle approximation ``Delta = 0.85*d``.

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
            Thickness of the perforated plate [mm], above 0.
        d : float or int, optional
            Hole diameter [mm], above 0 and at most ``s``.
        s : float or None, optional
            Hole spacing from the center of one hole to the next [mm], above 0. ``None`` is accepted with ``open_area``.
        open_area : float, optional
            Ratio of open area, above 0 and at most ``pi/4``. If set to ``None`` it is calculated from the hole spacing
            ``s``. If a value is supplied, the equivalent spacing is calculated and stored in place of ``s``.
        rho : float, int or None, optional
            Plate density [kg/m3], above 0. If ``None`` is passed, a fully rigid plate is assumed; ``rho=0`` raises a
            ``ValueError`` that asks for ``None``. If supplied, ``method="bessel_ingard"`` uses the full-surface mass
            ``1j*omega*rho*t`` while ``method="barrier"`` and ``method="barrier_mpp"`` use the solid-fraction mass
            ``1j*omega*rho*t*(1 - phi)``. ``method="conical"`` uses ``1j*omega*rho*t*(1 - phi_m)`` with the mean open
            area through the plate ``phi_m = phi*(1 + d_exit/d + (d_exit/d)**2)/3``, a package extension.
        end_correction : string or None, optional
            End-correction model. Options are ``"jb"`` / ``"jaouen_becot"``, ``"allard_atalla"``, ``"ingard"``,
            ``"nesterov"``, and ``"beranek"``. If ``None``, ``method="bessel_ingard"`` uses ``"ingard"`` and
            ``method="barrier"`` and ``method="eq_fluid"`` use ``"jb"``. ``method="barrier_mpp"`` accepts only
            ``"maa1998"`` (default) and ``"maa1987"``. ``method="conical"`` takes none.
        method : string, optional
            Chooses the perforated-plate formulation. Available options are ``"barrier"``, ``"barrier_mpp"``,
            ``"eq_fluid"``, ``"bessel_ingard"``, and ``"conical"``.
        porous_contact : float, optional
            Porous-contact factor ``K`` from 0 to 1. ``0`` disables the term. Only ``method="barrier"`` and
            ``method="bessel_ingard"`` accept a nonzero value. ``compute()`` warns when a nonzero value has no
            ``porous_layer`` directly before or after the sheet, since the term is then zero.
        layer : None or int, optional
            Optional value to choose the layer level, an integer or a float with an integer value. If None is passed the
            layer will be added to the existing ones.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the plate motion, in series with the plate mass, normalized by
            ``rho0*c0`` [-], at least 0. Requires ``rho``; not accepted by ``method="eq_fluid"``.
        d_exit : float or None, optional
            Hole diameter on the exit face for ``method="conical"`` [mm]. Must be larger than ``d`` and not larger than
            ``s``.
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
        [10] L. Jaouen and F.-X. Bécot, "Acoustical characterization of perforated facings", Journal of the Acoustical
            Society of America, vol. 129, pp. 1400-1406, 2011.
        [11] J. F. Allard and N. Atalla, "Propagation of Sound in Porous Media: Modelling Sound Absorbing Materials",
            2nd ed., Wiley, 2009.
        """
        name = "perforated_panel_layer"
        layer = utils.check_index(name, layer)
        if not isinstance(method, str) or method not in {"barrier", "barrier_mpp", "eq_fluid", "bessel_ingard",
                                                         "conical"}:
            raise ValueError(f"perforated_panel_layer method must be 'barrier', 'barrier_mpp', 'eq_fluid', "
                             f"'bessel_ingard', or 'conical', not {utils.describe_value(method)}.")
        self._validate_exit_radiation(exit_radiation, method)
        t = utils.check_thickness(name, t, what="the plate thickness [mm]")
        d = utils.check_positive(name, "d", "the hole diameter [mm]", d)
        if open_area is None:
            note = " Pass the hole spacing s, or open_area to compute it from." if s is None else ""
            s = utils.check_positive(name, "s", "the hole spacing [mm]", s, note)
        else:
            # open_area replaces s, so s=None works with it.
            if s is not None:
                utils.check_positive(name, "s", "the hole spacing [mm]", s)
            open_area = utils.check_number(name, "open_area", "the ratio of open area", open_area,
                                           "a finite number above 0 and at most pi/4 (0.785398), the open area of "
                                           "touching holes on a square pitch", lambda x: 0 < x <= np.pi / 4)
        rho = utils.check_plate_density(name, rho)
        if method == "conical":
            if d_exit is None:
                raise ValueError("perforated_panel_layer(method='conical') needs the exit hole diameter d_exit.")
            d_exit = utils.check_positive(name, "d_exit", "the hole diameter on the exit face [mm]", d_exit)
            if d_exit <= d:
                raise ValueError("perforated_panel_layer(method='conical') needs d_exit > d: the narrow end of the "
                                 "holes faces the incident side.")
            if end_correction is not None:
                raise ValueError("method='conical' takes no end_correction; its only end term is the exit radiation.")
            self._warn_cone_angle(t, d, d_exit, name)
        elif d_exit is not None:
            raise ValueError("d_exit is only used by method='conical'.")
        porous_contact = self._validate_porous_contact(name, porous_contact)
        if porous_contact and method not in {"barrier", "bessel_ingard"}:
            raise ValueError("porous_contact is only supported by method='barrier' and method='bessel_ingard'.")
        mechanical_resistance = self._validate_mechanical_resistance(name, mechanical_resistance, rho)
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

        open_area_given = open_area is not None
        if open_area is None:
            s_meters = s / 1000
            open_area = np.pi / ((2 * s_meters / d_meters) ** 2)
        else:
            s = d / 2 * np.sqrt(np.pi / open_area)

        if not open_area_given and d > s:
            shown_d, shown_s = utils.number_texts(d, s)
            # The open area is named only where its float tells it from pi/4.
            area, percent, limit = "", 100 * open_area, 100 * np.pi / 4
            if percent > limit:
                shown_area, shown_limit = utils.number_texts(percent, limit)
                area = (f" For the square pattern that is an open area of {shown_area} %, above pi/4 "
                        f"({shown_limit} %), where the holes touch.")
            raise ValueError(f"perforated_panel_layer: the holes overlap, since the hole diameter d = {shown_d} mm "
                             f"exceeds the hole spacing s = {shown_s} mm.{area}")
        if method == "conical" and d_exit > s:
            shown_exit, shown_s = utils.number_texts(d_exit, s)
            raise ValueError(f"perforated_panel_layer: the exit holes of method='conical' overlap, since the exit hole "
                             f"diameter d_exit = {shown_exit} mm exceeds the hole spacing s = {shown_s} mm.")

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

        self._warn_if_terminated("perforated_panel_layer")
        self.matrix[layer] = {"type": "perforated_panel_layer",
                              "thickness [mm]": t,
                              "hole_diameter [mm]": d,
                              "exit_hole_diameter [mm]": d_exit,
                              "hole_spacing [mm]": s,
                              "open_area [%]": open_area * 100,
                              "end_correction": effective_end_correction,
                              "end_correction_length [mm]": None if end_length is None else float(end_length) * 1000,
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
        sheet. A value outside the range its entry under Parameters gives raises a ``ValueError``, whether or not the
        method reads it; an open area of 1 leaves no plate and is refused.

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
            Thickness of the slotted plate [mm], above 0.
        w : float or int, optional
            Slit width [mm], above 0 and below ``s``.
        s : float or int, optional
            Slit spacing from the center of one slit to the next [mm], above 0.
        open_area : float, optional
            Ratio of open area, above 0 and below 1. If set to ``None`` it is calculated with the slit spacing ``s``. If
            a value is supplied, the equivalent spacing is calculated and stored.
        rho : float, int or None, optional
            Plate density [kg/m3], above 0. If ``None`` is passed, a fully rigid plate is assumed; ``rho=0`` raises a
            ``ValueError`` that asks for ``None``. If supplied, the solid-fraction mass ``1j*omega*rho*t*(1 - phi)`` is
            coupled in parallel with the slot impedance, or ``1j*omega*rho*t*(1 - phi_m)`` for ``method="conical"``.
        method : string, optional
            Slotted-panel calculation method: ``"barrier"`` or ``"conical"``.
        porous_contact : float, optional
            Porous-contact factor ``K`` from 0 to 1. ``0`` disables the term. Only ``method="barrier"`` accepts a
            nonzero value. ``compute()`` warns when a nonzero value has no ``porous_layer`` directly before or after the
            sheet, since the term is then zero.
        layer : None or int, optional
            Optional layer index, an integer or a float with an integer value. If ``None``, the layer is appended.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the plate motion, in series with the plate mass, normalized by
            ``rho0*c0`` [-], at least 0. Requires ``rho``.
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
        name = "slotted_panel_layer"
        layer = utils.check_index(name, layer)
        if not isinstance(method, str) or method not in {"barrier", "conical"}:
            raise ValueError(f"slotted_panel_layer method must be 'barrier' or 'conical', not "
                             f"{utils.describe_value(method)}.")
        self._validate_exit_radiation(exit_radiation, method)
        t = utils.check_thickness(name, t, what="the plate thickness [mm]")
        w = utils.check_positive(name, "w", "the slit width [mm]", w)
        if s is not None or open_area is None:  # open_area replaces s, so s=None works with it
            note = " Pass the slit spacing s, or open_area to compute it from." if s is None else ""
            s = utils.check_positive(name, "s", "the slit spacing [mm]", s, note)
        if open_area is not None:
            open_area = utils.check_interval(name, "open_area", "the ratio of open area", open_area, 0, 1,
                                             closed=(False, False))
        rho = utils.check_plate_density(name, rho)
        porous_contact = self._validate_porous_contact(name, porous_contact)
        if method == "conical":
            if w_exit is None:
                raise ValueError("slotted_panel_layer(method='conical') needs the exit slot width w_exit.")
            w_exit = utils.check_positive(name, "w_exit", "the slit width on the exit face [mm]", w_exit)
            if w_exit <= w:
                raise ValueError("slotted_panel_layer(method='conical') needs w_exit > w: the narrow end of the slots "
                                 "faces the incident side.")
            if porous_contact:
                raise ValueError("porous_contact is only supported by method='barrier'; method='conical' takes the "
                                 "porous layer behind the plate through exit_radiation.")
            self._warn_cone_angle(t, w, w_exit, name)
        elif w_exit is not None:
            raise ValueError("w_exit is only used by method='conical'.")
        mechanical_resistance = self._validate_mechanical_resistance(name, mechanical_resistance, rho)
        if layer is None:
            layer = len(self.matrix)

        if open_area is None:
            if w >= s:
                shown_w, shown_s = utils.number_texts(w, s)
                raise ValueError(f"slotted_panel_layer: the slits leave no plate between them, since the slit width "
                                 f"w = {shown_w} mm is not below the slit spacing s = {shown_s} mm.")
            open_area = w / s
        else:
            s = w / open_area

        # Adjusting units
        t_meters = t / 1000  # Convert millimeters to meters
        w_meters = w / 1000

        contact_resistance = 0.0
        if method == "conical":
            # Wedge-shaped slots of Vigran (2004): wave solution in the slot, exit strip radiation, equation 27.
            if w_exit > s:
                shown_exit, shown_s = utils.number_texts(w_exit, s)
                raise ValueError(f"slotted_panel_layer(method='conical'): the exit slits overlap, since the exit slit "
                                 f"width w_exit = {shown_exit} mm exceeds the slit spacing s = {shown_s} mm.")
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

        self._warn_if_terminated("slotted_panel_layer")
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
            Plate thickness [mm], above 0.
        w : float, optional
            Slot width, the short side of the aperture [mm], above 0 and at most ``sw``.
        l : float, optional
            Slot length, the long side of the aperture [mm], above 0 and at most ``sl``. ``w = sw`` together with
            ``l = sl`` leaves no plate and raises a ``ValueError``.
        sw : float, optional
            Centre-to-centre spacing between slots across their width [mm], above 0.
        sl : float, optional
            Centre-to-centre spacing between slots along their length [mm], above 0.
        rho : float or None, optional
            Plate density [kg/m³], above 0. If ``None``, the plate is rigid; ``rho=0`` raises a ``ValueError`` that asks
            for ``None``.
        porous_contact : float, optional
            Porous-contact factor ``K`` from 0 to 1. ``0`` disables the term. ``compute()`` warns when a nonzero
            value has no ``porous_layer`` directly before or after the sheet, since the term is then zero.
        layer : None or int, optional
            Optional layer index, an integer or a float with an integer value. If ``None``, the layer is appended.
        mechanical_resistance : float, optional
            Mechanical loss resistance ``r_m`` of the plate motion, in series with the plate mass, normalized by
            ``rho0*c0`` [-], at least 0. Requires ``rho``. Values outside the ranges above raise a ``ValueError``.

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
        layer = utils.check_index("rectangular_slot_layer", layer)
        t = utils.check_thickness("rectangular_slot_layer", t, what="the plate thickness [mm]")
        w, l, sw, sl = (utils.check_positive("rectangular_slot_layer", name, what, value) for name, what, value in (
            ("w", "the slot width [mm]", w), ("l", "the slot length [mm]", l),
            ("sw", "the slot spacing across the width [mm]", sw), ("sl", "the slot spacing along the length [mm]", sl)))
        if w > sw:
            shown_w, shown_sw = utils.number_texts(w, sw)
            raise ValueError(f"rectangular_slot_layer: the slots overlap, since the slot width w = {shown_w} mm "
                             f"exceeds "
                             f"the spacing across the width sw = {shown_sw} mm.")
        if l > sl:
            shown_l, shown_sl = utils.number_texts(l, sl)
            raise ValueError(f"rectangular_slot_layer: the slots overlap, since the slot length l = {shown_l} mm "
                             f"exceeds "
                             f"the spacing along the length sl = {shown_sl} mm.")
        if w == sw and l == sl:
            raise ValueError(f"rectangular_slot_layer: the slots leave no plate between them, since w = sw = {w:g} mm "
                             f"and l = sl = {l:g} mm give an open area of 1.")
        rho = utils.check_plate_density("rectangular_slot_layer", rho)
        porous_contact = self._validate_porous_contact("rectangular_slot_layer", porous_contact)
        mechanical_resistance = self._validate_mechanical_resistance("rectangular_slot_layer", mechanical_resistance,
                                                                     rho)
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

        self._warn_if_terminated("rectangular_slot_layer")
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
        d : float
            Hole diameter [m], above 0. Another value raises a ``ValueError``.
        open_area : float, optional
            Deprecated compatibility argument, above 0 and at most 1. If supplied, ``zc`` is scaled by ``1/open_area``
            to reproduce the previous panel-scaled return value. New code should omit this argument and apply panel
            scaling in the layer builder.

        Returns
        -------
        Pore propagation constant array and pore characteristic acoustic impedance array.
        """
        d = utils.check_positive("viscothermal_circular", "d", "the hole diameter [m]", d)
        if open_area is not None:
            open_area = utils.check_fraction("viscothermal_circular", "open_area", "the ratio of open area", open_area)
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

    @staticmethod
    def _check_material_params(type, params):
        """
        Raise a ``ValueError`` for a ``material_model`` panel parameter in ``params`` that is not a finite number above
        0, for a ``smooth`` that is not ``True`` or ``False``, or for a crossover frequency of ``"door"`` or
        ``"window"`` that is not below half the sample rate. NumPy numbers in ``params`` become Python numbers.
        """
        texts = {"sample_rate": "the sample rate of the crossover filters [Hz]",
                 "crossover_frequency": "the crossover frequency [Hz]", "rho_m": "the panel density [kg/m³]",
                 "d": "the panel thickness [m]", "area": "the panel area [m²]",
                 "f_res": "the panel resonance frequency [Hz]"}
        for key, what in texts.items():
            if key in params:
                utils.check_positive("material_model", f"params['{key}']", what, params[key])
                params[key] = utils.plain_scalar(params[key])
        if "smooth" in params:
            params["smooth"] = utils.check_flag("material_model", "params['smooth']", params["smooth"])
        defaults = {"door": (44100, 250), "window": (44100, 200)}.get(type)
        if defaults is None and not {"sample_rate", "crossover_frequency"} <= params.keys():
            return
        sample_rate = params.get("sample_rate", defaults[0] if defaults else None)
        crossover = params.get("crossover_frequency", defaults[1] if defaults else None)
        if not crossover < sample_rate / 2:
            raise ValueError(f"material_model: params['crossover_frequency'], the crossover frequency [Hz], must be "
                             f"below half the sample rate, {sample_rate / 2:g} Hz, not {crossover!r}.")

    def material_model(self, type="door", params=None):
        """
        Set an empirical boundary impedance model adapted from the GRAS material helpers distributed with Hargreaves,
        Rendell, and Lam [1].

        This method is separate from the transfer-matrix layer builders. It directly assigns ``self.z`` and, where
        available, ``self.scat`` from empirical room-acoustics boundary models used in the GRAS auralization examples.
        The surface is locally reacting. [1], section III.E, p. 2633, states this assumption for the measured absorption
        data. Treating the ``door`` and ``window`` hybrids the same way is a package choice, which follows the use of
        their admittance in the boundary-element model of [1]. ``self.z_angle`` repeats ``self.z`` at every incidence
        angle, and ``self.alpha`` is what ``constant_z()`` gives for the same impedance. That is the plane-wave
        absorption at the angle of an angle object, the Paris average over ``incidence_angle`` with
        ``diffuse_method="paris"``, and the normal-incidence absorption otherwise. ``finite_sample_absorption()``
        integrates the impedance over the object's ``incidence_angle``, so its result depends on that angle range.

        The ``floor``, ``ceiling``, ``concrete``, ``plaster``, ``door``, and ``window`` branches are Python adaptations
        of the Scene 9 MATLAB helpers ``MaterialModel_Scene9*.m``. The ``mdf`` branch follows
        ``MaterialModel_Scene3MDF.m``. The resistive-data branches load GRAS surface-description absorption data,
        convert random-incidence absorption to a purely real normalized admittance with the 55 degree rule,
        ``Y = cos(55 deg)*(1 - sqrt(1 - alpha))/(1 + sqrt(1 - alpha))`` ([1], section III.E), and interpolate the
        admittance onto ``self.freq``. TMM fits a natural cubic spline, held at its end values outside the measured
        bands, where the MATLAB helpers call ``spline(fMeas, [YsMeas(1) YsMeas YsMeas(end)], f)``, a spline whose end
        slopes are the first and last admittance values. Where the bundled CSV files carry scattering data,
        ``self.scat`` is interpolated the same way; that output is a package addition, since the MATLAB helpers return
        admittance only. ``door`` has no such data, so ``self.scat`` stays ``None``.

        Every spline is evaluated at ``self.freq`` clipped to the measured band centres, so the admittance and the
        scattering hold their end values outside the data. Hargreaves et al. [1] hold the absorption constant below
        their lowest band, which is 20 Hz. Holding it below a higher lowest band and above the highest band is a package
        choice. At very low frequencies, Mondet et al. [3], p. 11, write that the extrapolation of their fitted model
        "can provide more accuracy than the simple and common assumption of constant values between the 63 Hz and 125 Hz
        octave bands". Without the hold, the natural spline of the ``door`` octave data would turn the admittance
        negative at high frequencies, and the crossover, which adds magnitudes, would turn that into rising absorption.

        The ``self.matrix["material_model"]`` record keeps, next to ``type`` and ``params``, the band center
        frequencies, absorption coefficients and scattering coefficients that were fitted, as lists of floats under
        ``measured_bands_hz``, ``measured_absorption`` and ``measured_scattering``. ``door`` stores its summed
        absorption and transmission coefficients and ``None`` for scattering. ``rebuild()`` refits from the bundled
        data, so the record documents the fit and is never read back. The model replaces any stored layers, with a
        ``UserWarning``, and a layer added afterwards warns that ``compute()`` ignores it.

        The ``door`` and ``window`` branches use the hybrid construction: a resistive absorption-data fit combined with
        a reactive mass-spring-damper panel admittance through a Linkwitz-Riley-style crossover, which the source
        helpers call the non-linear crossover method of Aretz et al. [2]. The crossover adds the magnitudes of the two
        filtered admittances and takes the phase of the mass-spring-damper branch. That branch is
        ``zS = 1j*omega*M + R + K/(1j*omega)`` in the ``exp(1j*omega*t)`` convention of this package. [1], section II,
        and its helpers use ``exp(-1j*omega*t)`` and write ``zS = -1j*omega*M + R + K/(-1j*omega)``. The package form is
        its complex conjugate and gives a mass reactance with the sign of ``membrane_layer``. The plane-wave absorption
        at any angle, and so ``alpha``, does not depend on the sign of the reactance, while
        ``finite_sample_absorption()`` does. TMM uses the current object's air properties rather than the helpers'
        hard-coded ``rho0=1.21 kg/m3`` and ``c0=343 m/s``. The optional ``smooth`` parameter is a package extension.

        GRAS database: https://depositonce.tu-berlin.de//handle/11303/7506
        Supplemental data: https://asa.scitation.org/doi/suppl/10.1121/1.5096171

        Parameters
        ----------
        type : str, optional
            One of ``"floor"``, ``"ceiling"``, ``"door"``, ``"concrete"``, ``"plaster"``, ``"mdf"`` or ``"window"``.
        params : dict, optional
            Panel model of ``"door"`` and ``"window"``, a dict or what ``dict()`` accepts. Missing keys take the
            defaults of the type, given here as door and window values:

            - ``sample_rate`` [Hz], of the crossover filters: 44100 and 44100.
            - ``crossover_frequency`` [Hz]: 250 and 200.
            - ``rho_m``, panel density [kg/m³]: 375 and 2500.
            - ``d``, panel thickness [m]: 0.043 and 0.0067.
            - ``area``, panel area [m²]: ``2.2*0.97`` and 5.33.
            - ``f_res``, panel resonance frequency [Hz]: 95 and 6.66.
            - ``smooth``, Savitzky-Golay smoothing of the combined admittance over ``MATERIAL_SMOOTHING_WINDOW``
              points: ``False`` and ``False``.

            The six numbers must be above 0 for any ``type``, and the crossover frequency below half the sample rate;
            another value raises a ``ValueError``, and so does ``smooth=True`` on a grid with fewer points than the
            smoothing window. Other keys are kept in the record and not read.

            The mass-spring-damper branch ``zS`` takes the total panel mass ``M = rho_m*d*area`` [kg], the stiffness
            ``K = M*(2*pi*f_res)**2`` and a fixed resistance ``R`` of 12000 for the door and 6000 for the window, as the
            helpers do. Its admittance ``rho0*c0/zS`` also carries an added resistive admittance, the 55 degree rule
            applied to the mean absorption of the 500 Hz to 4 kHz door bands or to the 125 Hz window band.

        References
        ----------
        [1] J. A. Hargreaves, L. R. Rendell and Y. W. Lam, "A framework for auralization of boundary element method
            simulations including source and receiver directivity", Journal of the Acoustical Society of America,
            vol. 145, pp. 2625-2637, 2019.
        [2] M. Aretz, R. Nöthen, M. Vorländer and D. Schröder, "Combined broadband impulse responses using FEM and
            hybrid ray-based methods", EAA Symposium on Auralization, Espoo, 2009.
        [3] B. Mondet, J. Brunskog, C.-H. Jeong and J. H. Rindel, "From absorption to impedance: Enhancing boundary
            conditions in room acoustic simulations", Applied Acoustics, vol. 157, 106884, 2020.
        """
        utils.check_choice("material_model", "type", "the material", type, MATERIAL_MODEL_TYPES)
        try:
            params = None if params is None else dict(params)
        except (TypeError, ValueError):
            raise ValueError(f"material_model: params must be None, a dict or what dict() accepts, such as a list of "
                             f"(key, value) pairs, not {utils.describe_value(params)}.") from None
        if params is not None:
            self._check_material_params(type, params)
        points = len(self.freq)
        smoothed = type in ("door", "window") and bool(params) and params.get("smooth") is True
        if smoothed and points < MATERIAL_SMOOTHING_WINDOW:
            raise ValueError(f"material_model: params['smooth'] needs a grid of at least {MATERIAL_SMOOTHING_WINDOW} "
                             f"frequencies, the window of the Savitzky-Golay smoothing, and this grid has {points}. "
                             f"Pass smooth=False or use a finer grid.")
        discarded = self._discarded_layers_message()
        scat = None

        def material_spline(x, y):
            # Natural spline through the band values, held at the end values outside the measured bands.
            spline = CubicSpline(x, y, bc_type="natural")
            return lambda f: spline(np.clip(f, x[0], x[-1]))

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
            z = 1 / material_spline(fMeas, YsMeas)(self.freq)
            scat = material_spline(fMeas, sMeas)(self.freq)

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
            # Surface impedance in the package's exp(+1j*omega*t) convention, the conjugate of the helper's form
            zS = (1j * 2 * np.pi * self.freq) * M + R + K / (1j * 2 * np.pi * self.freq)
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
                Ys_real = savgol_filter(np.real(Ys), MATERIAL_SMOOTHING_WINDOW, 3)
                Ys_imag = savgol_filter(np.imag(Ys), MATERIAL_SMOOTHING_WINDOW, 3)
                Ys = Ys_real + 1j * Ys_imag

            z = 1 / Ys

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

            scat = SsInterp

            # Model 2: reactive Mass-Spring-Damper fit to assumed fundamental panel resonance:
            M = rho_m * d * area  # Mass term
            K = M * (2 * np.pi * f_res) ** 2  # Stiffness term  - adjusted to match assumed f_res
            R = 6000  # Resistance term - adjusted to match measured coefficients
            # Surface impedance in the package's exp(+1j*omega*t) convention, the conjugate of the helper's form
            zS = (1j * 2 * np.pi * self.freq) * M + R + K / (1j * 2 * np.pi * self.freq)
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
                Ys_real = savgol_filter(np.real(Ys), MATERIAL_SMOOTHING_WINDOW, 3)
                Ys_imag = savgol_filter(np.imag(Ys), MATERIAL_SMOOTHING_WINDOW, 3)
                Ys = Ys_real + 1j * Ys_imag

            z = 1 / Ys

        record = self._material_model_record(type, params, fMeas, aMeas, sMeas)
        filename = self.filename if "_material_model" in self.filename else self.filename + "_material_model"
        # The object changes only from here, once every value is computed, so a call that raises leaves it as it was.
        self._set_locally_reacting(z * self.z0, {"material_model": record}, "material_model")
        self._scat = scat
        self.filename = filename
        if discarded is not None:
            warnings.warn(f"material_model: {discarded}", UserWarning, stacklevel=2)

    def field_impedance(self, z):
        """
        Compute the field-incidence impedance from a set of angle-dependent impedances.

        This method averages surface admittance rather than absorption, after Aretz [1], eq. 7.16. It evaluates
        ``Y_f = integral(sin(theta) / Zs(theta) dtheta) / integral(sin(theta) dtheta)`` over the angles in
        ``self.incidence_angle``, with ``Zs`` the angle-dependent surface impedance, and returns ``Z_f = 1 / Y_f``. For
        ``diffuse_method='field'``, ``compute()`` passes ``Z_f`` through the normal-incidence reflection relation to
        give ``alpha``, the absorption at field incidence taken from the averaged impedance as in Takahashi et al. [2],
        who weight the admittance with ``sin(theta)*cos(theta)``. Averaging the admittance and then applying the
        normal-incidence relation is not the same as averaging the absorption at each angle. ``diffuse_method='paris'``
        does the latter, weighting ``alpha(theta)`` with ``sin(theta)*cos(theta)``. That is the Paris form, Aretz [1],
        eq. 3.11, which Aretz evaluates up to 78 degrees for field incidence.

        The integral is a package extension of the source. Between neighbouring angles the method interpolates ``Zs``
        and ``sin(theta)`` linearly and integrates their ratio exactly. On an interval of width ``h`` from impedance
        ``Za`` to ``Zb``, with sines ``sa`` and ``sb``, that is ``h*(sa*L/q + (sb - sa)*(q - L)/q**2) / Za``, with
        ``q = (Zb - Za) / Za`` and ``L = log(Zb / Za)``. The admittance has near-poles where the reactance of a lightly
        damped stack crosses zero at an oblique angle, and point sampling misses them, while ``Zs`` itself stays smooth.
        Where ``Zs`` changes by less than a tenth of itself across an interval, the trapezoid of ``sin(theta) / Zs`` is
        used instead, which differs from the exact form by about ``abs(q)**2/6`` relative. An angle-independent ``Zs``
        is returned unchanged.

        Because this is a complex admittance average, the result depends on the impedance reference plane: a lossless
        top air layer changes it even though the angle-wise absorption is unchanged. See ``compute()``.

        Parameters
        ----------
        z : array
            Angle-dependent surface impedances, one row per frequency and one column per angle in
            ``self.incidence_angle`` [Pa*s/m].

        Returns
        -------
        array
            Field-incidence impedance for each frequency [Pa*s/m].

        Notes
        -----
        A segment from ``Za`` to ``Zb`` that misses the origin turns through less than ``pi`` as seen from it, so the
        principal logarithm is the change of ``log(Zs)`` along it. A lossless crossing through the origin takes the
        limit of a vanishing positive resistance, a turn of ``+pi`` when the reactance rises and ``-pi`` when it falls.
        A node with ``Zs = 0`` makes ``Y_f`` infinite and returns ``Z_f = 0``, while a zero between two nodes gives the
        finite limit above.

        References
        ----------
        [1] M. Aretz, "Combined Wave and Ray Based Room Acoustic Simulations of Small Rooms", Ph.D. thesis, RWTH
            Aachen University, Logos Verlag, 2012.
        [2] Y. Takahashi, T. Otsuru and R. Tomiku, "In situ measurements of surface impedance and absorption
            coefficients of porous materials using two microphones and ambient noise", Applied Acoustics, vol. 66,
            pp. 845-865, 2005.
        """
        z = np.asarray(z)
        if z.ndim != 2 or z.shape[1] != len(self.incidence_angle):
            self._raise_if_partial_z_angle("field_impedance()")
            raise ValueError(
                f"z must have one column per angle in incidence_angle: got shape {z.shape} for "
                f"{len(self.incidence_angle)} angles."
            )
        return self._field_impedance_and_near_poles(z, self.incidence_angle)[0]

    @classmethod
    def _field_impedance_and_near_poles(cls, z, angles_deg):
        """
        Return the field-incidence impedance and the number of near-pole intervals. Rows whose impedance is the same at
        every angle are returned as that impedance, without the round trip through the admittance.
        """
        z = np.asarray(z)
        admittance, near_poles = cls._field_admittance(z, angles_deg)
        impedance = cls._admittance_to_impedance(admittance)
        constant = np.all(z == z[:, :1], axis=1)
        impedance[constant] = z[constant, 0]
        return impedance, near_poles

    @staticmethod
    def _admittance_to_impedance(admittance):
        """Invert an admittance, mapping an infinite admittance to ``Z = 0`` and a zero admittance to ``Z = inf``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            impedance = 1 / admittance
        impedance = np.where(np.isinf(admittance), 0, impedance)
        return np.where(admittance == 0, np.inf, impedance).astype(complex)

    @staticmethod
    def _field_admittance(z, angles_deg, q_cut=0.1):
        """
        Return the field-incidence admittance of ``field_impedance()`` and the number of intervals where ``Zs`` changes
        by more than ``q_cut`` of itself, which used the exact form. On a fine angle grid that count is zero unless the
        admittance has a near-pole; a coarse grid can give a nonzero count without one.
        """
        z = np.asarray(z, dtype=complex)
        theta = np.deg2rad(np.asarray(angles_deg, dtype=float))
        sin = np.sin(theta)
        step = np.diff(theta)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if theta.size < 2:
                return 1 / z[:, 0], 0
            ratio_sin = sin[None, :] / z
            terms = step * (ratio_sin[:, :-1] + ratio_sin[:, 1:]) / 2
            q = z[:, 1:] / z[:, :-1] - 1
            rows, cols = np.nonzero(np.abs(q) > q_cut)
            if rows.size:
                z0 = z[rows, cols]
                z1 = z[rows, cols + 1]
                q = q[rows, cols]
                ratio = q + 1
                log_ratio = np.log(ratio)
                # A segment through the origin (Z1/Z0 negative real, to rounding) is a lossless crossing.
                crossing = (ratio.real < 0) & (np.abs(ratio.imag) <= 1e-9 * np.abs(ratio))
                log_ratio[crossing] = (np.log(-ratio.real[crossing])
                                       + 1j * np.pi * np.sign(z1.imag[crossing] - z0.imag[crossing]))
                exact = step[cols] * (sin[cols] * log_ratio / q
                                      + (sin[cols + 1] - sin[cols]) * (q - log_ratio) / q ** 2) / z0
                finite = np.isfinite(exact)
                terms[rows[finite], cols[finite]] = exact[finite]
            admittance = terms.sum(axis=1) / np.sum(step * (sin[:-1] + sin[1:]) / 2)
        admittance[(z == 0).any(axis=1)] = np.inf
        return admittance, int(rows.size)

    def _warn_if_field_overstates(self, z_field):
        """
        Warn when the absorption from the field-incidence impedance exceeds the statistical absorption over the same
        angles by more than the module constant ``FIELD_EXCESS_LIMIT``. ``compute()`` calls it only when some interval
        used the exact form of ``field_impedance()``. With an angle step coarser than ``COARSE_ANGLE_STEP`` the message
        names the step as a possible cause.
        """
        _, alpha_field = self.reflection_and_absorption_coefficient(z_field)
        excess = np.asarray(alpha_field - self.diffuse_absorption_coefficient(self._z_angle)).reshape(-1)
        over = excess > FIELD_EXCESS_LIMIT
        if not np.any(over):
            return
        worst = int(np.nanargmax(excess))
        freq = np.asarray(self.freq, dtype=float)
        angles = np.asarray(self.incidence_angle, dtype=float)
        low, high = freq[over].min(), freq[over].max()
        step = float(np.max(np.diff(angles))) if angles.size > 1 else 0.0
        band = f"between {low:g} and {high:g} Hz" if high > low else f"only at {low:g} Hz"
        if step > COARSE_ANGLE_STEP:
            cause = (f"The angle step of {step:g} deg is coarse enough to cause this difference by itself. Repeat "
                     f"with a finer incidence_angle step; if the warning remains, the admittance average is dominated "
                     f"by resonances at oblique angles. Compare with diffuse_method='paris'.")
        else:
            cause = ("The admittance average is dominated by resonances at oblique angles, typical of an air cavity "
                     "with little damping, so the field-incidence impedance and the absorption taken from it overstate "
                     "the absorption of this stack. Compare with diffuse_method='paris'.")
        warnings.warn(
            f"The absorption from the field-incidence impedance exceeds the statistical (Paris) absorption over the "
            f"same angles ({angles.min():g} to {angles.max():g} deg) by up to {excess[worst]:.2f} at "
            f"{freq[worst]:g} Hz (above {FIELD_EXCESS_LIMIT:g} {band}). " + cause,
            RuntimeWarning,
            stacklevel=4,
        )

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

        With ``diffuse_method='field'``, when some interval needs the exact form of ``field_impedance()``, as at a
        near-pole of the admittance, this method also evaluates the statistical absorption over the same angles and
        compares it with ``alpha``. If ``alpha`` exceeds it at any frequency by more than the module constant
        ``FIELD_EXCESS_LIMIT``, a difference between two absorption coefficients, it emits a ``RuntimeWarning``. The
        admittance average is then dominated by resonances at oblique angles, as in an air cavity with little damping,
        and both the field-incidence impedance and its absorption overstate the absorption of the stack. This is a
        package check that no source prescribes.

        On a termination, such as ``constant_z()`` or ``material_model()``, there is nothing to multiply. When a changed
        setting has made its results stale, this method sets the termination again from its record on the current grid,
        as ``rebuild()`` does, and otherwise leaves it unchanged.

        This method raises a ``ValueError`` when the absorption of the stack is NaN at some frequency. The layer methods
        check their inputs, so the cause is an overflow, from a value far outside its usual range, such as a length in
        the wrong unit, or an extremely thick, resistive layer. A call that raises leaves the object as it was.
        """
        rigid_backing = utils.check_flag("compute", "rigid_backing", rigid_backing)
        conj = utils.check_flag("compute", "conj", conj)
        show_layers = utils.check_flag("compute", "show_layers", show_layers)
        if backing is not None and (not isinstance(backing, str) or backing not in {"rigid", "radiation", "air"}):
            raise ValueError(f"backing must be 'rigid', 'radiation', or 'air', not {utils.describe_value(backing)}.")
        terminated = self._termination_name() is not None
        if getattr(self, "_layers_stale", False) and not terminated:
            raise RuntimeError(f"compute() cannot use stale layer matrices. Call rebuild() first."
                               f"{self._stale_details()}")

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

        state = self._state_snapshot()
        try:
            if not terminated:
                # The old backing entry is removed and only re-added once the computation succeeds, so without the
                # snapshot a failure in between would leave the treatment without its backing metadata.
                self._compute_global_matrix(backing, backing_was_omitted, conj)
            elif getattr(self, "_layers_stale", False) or getattr(self, "_results_stale", False):
                self._restore_records(self.matrix.copy())
        except BaseException:
            self._restore_state(state)
            raise

        if show_layers and not terminated:
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
                raise ValueError(f"{utils.layer_label(self.matrix, key)} ({record['type']}) has exit_radiation="
                                 f"{record['exit_radiation']!r} but no porous_layer directly after it or after its "
                                 "in-contact facing.")
        for key, record in list(self.matrix.items()):
            # A plate that folded in a facing at an earlier compute() is rebuilt too, in case that facing has gone.
            if isinstance(record, dict) and (record.get("porous_contact") or record.get("model") == "double_porosity"
                                             or key in contact_plates or record.get("contact_facing")
                                             or record.get("exit_radiation") in {"porous", "mean"}):
                self._add_layer_from_record(key, record)
        for key, record in self.matrix.items():
            if isinstance(record, dict) and record.get("porous_contact") and not any(
                    self.matrix.get(neighbour, {}).get("type") == "porous_layer" for neighbour in (key - 1, key + 1)):
                warnings.warn(f"{utils.layer_label(self.matrix, key)} ({record['type']}) has porous_contact="
                              f"{utils.number_texts(record['porous_contact'])[0]} but no porous_layer directly before "
                              f"or after it, so its "
                              "contact resistance is zero.",
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
            # Rigid termination: rear particle velocity is zero, so Zs = S0*A/C. C = 0 means nothing in the stack can
            # move against the wall (only sheet layers without a volume of their own, no cavity), so Zs is infinite.
            blocked = Cg == 0
            with np.errstate(divide="ignore", invalid="ignore"):
                self.z_angle = np.where(blocked, np.inf, self.s0 * Ag / Cg)
            if np.all(blocked):
                warnings.warn(
                    "The surface impedance is infinite at every frequency because nothing in the stack can move "
                    "against the rigid backing. Membranes, plates, perforated, slotted and rectangular-slot panels and "
                    "fabric facings are modelled without a volume of their own, so a stack made only of them, in any "
                    "combination, has no air to compress. Nothing absorbs, so alpha is 0. Add an air or porous layer "
                    "behind them, for example with air_layer().",
                    RuntimeWarning,
                    stacklevel=3,
                )
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
            zc, near_poles = self._field_impedance_and_near_poles(self._z_angle, self.incidence_angle)
            alpha = self.diffuse_absorption_coefficient(self._z_angle) if self.diffuse_method == "paris" else None
            if self.diffuse_method == "field" and near_poles:
                self._warn_if_field_overstates(zc)
        else:
            zc = self._z_angle[:, 0]
            if self.incidence == "angle":
                _, alpha = self.reflection_and_absorption_coefficient(zc, angle=self.incidence_angle[0])
            else:
                alpha = None

        # An infinite impedance gives alpha = 0, so NaN comes from overflow or invalid arithmetic in a layer.
        self._raise_if_nan_absorption(alpha if alpha is not None else self.reflection_and_absorption_coefficient(zc)[1],
                                      "the stack")

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

    def _add_layer_from_record(self, key, value, stacklevel=3, matrix=None):
        """
        Re-create one stored layer at ``key`` from its ``self.matrix`` metadata, through the layer method and its input
        checks. ``stacklevel`` is that of the thin-plate warning. A refused value raises a ``ValueError`` that names the
        layer as numbered in ``matrix``, by default ``self.matrix``.
        """
        # The record holds checked values, so a cone-angle warning was already given when the layer was made. The
        # thin-plate warning depends on the frequency range, so it is given again only when the range now goes above
        # the plate's limit and the range checked when the layer was made did not.
        limit = value.get("thin_plate_limit [Hz]") if value.get("type") == "thin_plate_layer" else None
        checked = value.get("checked_fmax [Hz]")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*method='conical'\): the aperture half-angle")
            warnings.filterwarnings("ignore", message=r"thin_plate_layer: the bending wavelength")
            try:
                self._add_layer_from_record_inner(key, value)
            except ValueError as error:
                source = self.matrix if matrix is None else matrix
                label = utils.layer_label(source, key) if key in source else f"Layer {key + 1}"
                raise ValueError(f"{label}, a stored {value.get('type')}, cannot be re-created: {error}") from error
        if (limit is not None and self._has_oblique_angle() and np.max(self.freq) > limit
                and (checked is None or checked <= limit)):
            warnings.warn(f"thin_plate_layer: the bending wavelength is shorter than 6 plate thicknesses above "
                          f"{limit:.0f} Hz, where thin-plate theory loses accuracy.", UserWarning,
                          stacklevel=stacklevel)

    def _add_layer_from_record_inner(self, key, value):
        """
        Re-create one stored layer at ``key``; see ``_add_layer_from_record``. A stored plate density of 0 is read as
        ``None``, the rigid plate it described, a stored one-element ``sigma`` array as its number, and stored on/off
        values through ``_stored_flag``.
        """
        if value["type"] in {"perforated_panel_layer", "slotted_panel_layer", "rectangular_slot_layer"}:
            if utils.real_number(value.get("rho [kg/m3]")) == 0:
                value = dict(value, **{"rho [kg/m3]": None})
        if value["type"] == "porous_layer":
            sigma = value["flow_resistivity [k*Pa*s/m²]"]
            if isinstance(sigma, np.ndarray) and sigma.shape == (1,):
                # Earlier versions accepted and saved a one-element array for sigma; it stands for that number.
                sigma = sigma[0]
            self.porous_layer(sigma=sigma,
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
                                     free=self._stored_flag(value.get("free"), True),
                                     rho=value["rho [kg/m3]"],
                                     d=value["pore_diameter [mm]"],
                                     s=value["pore_spacing [mm]"],
                                     open_area=None if open_area is None else open_area / 100,
                                     layer=key,
                                     in_contact=self._stored_flag(value.get("in_contact"), False))
        if "preset" in value:
            # The record holds the resolved values, so the layer is rebuilt from them without the preset table, which
            # may no longer know the name, and the preset fields are carried over as they were.
            self.matrix[key]["preset"] = value["preset"]
            self.matrix[key]["preset_overrides"] = value.get("preset_overrides")

    @staticmethod
    def _stored_flag(value, default):
        """
        Return an on/off value read back from a record for the checks of a new call: ``None`` gives ``default``, and a
        NumPy boolean or the integers 0 and 1, as a saved file can hold them, give the bool they stand for. Anything
        else is returned for the check to refuse.
        """
        if value is None:
            return default
        if isinstance(value, (bool, np.bool_)) or (isinstance(value, (int, np.integer)) and value in (0, 1)):
            return bool(value)
        return value

    @staticmethod
    def _check_record_supported(value):
        """Raise for a stored layer record that the current layer methods cannot rebuild."""
        if value.get("type") == "rectangular_slot_layer" and value.get("resistance [Pa*s/m]"):
            raise ValueError("This rectangular_slot_layer record has a slot resistance, which the layer no longer "
                             "takes; add porous_facing_layer(method='pieren_fixed', resistance=R, in_contact=True) "
                             "next to the plate instead.")

    def rebuild(self):
        """
        Rebuild the layer matrices from the stored metadata and recompute the results. Each stored layer is re-created
        through its layer method and that method's input checks. A stored value that a new call would refuse raises a
        ``ValueError`` that names the layer, numbered from 1 as ``show_layers()`` does; a stored plate ``rho`` of 0 is
        read as ``None``, a rigid plate. A call that raises leaves the object as it was.
        """
        state = self._state_snapshot()
        try:
            matrix = self.matrix.copy()
            for value in matrix.values():
                if isinstance(value, dict):
                    self._check_record_supported(value)
            preserve_diffuse_method = (
                getattr(self, "_results_stale", False)
                and any("diffuse_method was changed" in reason for reason in getattr(self, "_stale_reasons", []))
            )
            current_diffuse_method = self.diffuse_method
            self._restore_records(matrix)

            backing_data = self._backing_record(matrix)[1]
            if backing_data is not None:
                if "diffuse_method" in backing_data:
                    self._diffuse_method = (current_diffuse_method if preserve_diffuse_method
                                            else backing_data["diffuse_method"])
                compute_args = dict(rigid_backing=self._stored_flag(backing_data["rigid_backing"], True),
                                    backing=backing_data.get("backing"),
                                    conj=self._stored_flag(backing_data["impedance_conjugate"], False))
            else:
                compute_args = dict(rigid_backing=False)
            # A missing porous neighbour was reported when the stack was first computed.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Layer \d+ .* has porous_contact=")
                self.compute(show_layers=False, **compute_args)
            self._clear_stale_state()
        except BaseException:
            self._restore_state(state)
            raise

    def _restore_records(self, matrix):
        """
        Set the termination and re-create the layers of the records in ``matrix`` on the current grid, for ``rebuild()``
        and for ``compute()`` on a termination whose results are stale. Warnings point at the line that called either
        method.
        """
        self._clear_stale_state()
        self.matrix = {}
        # A termination replaces self.matrix, so it is set before the layers stored with it, whatever order load()
        # restores the keys in. A layer stored after a termination was reported when it was added.
        ordered = sorted(matrix.items(), key=lambda item: item[0] not in ("termination", "material_model"))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"\w+: the surface impedance is set by \w+\(\)")
            for key, value in ordered:
                if key == "termination":
                    if value["type"] == "constant_z":
                        self._set_constant_z(np.asarray(self._z).reshape(-1)[0])
                    elif value["type"] == "impedance_from_absorption" and "literature" in value:
                        self._rebuild_material_from_table(value, stacklevel=5)
                    elif value["type"] == "impedance_from_absorption":
                        self._rebuild_impedance_from_absorption(value, stacklevel=4)
                elif key != "material_model":
                    self._add_layer_from_record(key, value, stacklevel=4, matrix=matrix)
                else:
                    # material_model() adds its suffix to the file name; a stored model keeps the current name.
                    filename = self._filename
                    self.material_model(value["type"], params=value["params"])
                    self._filename = filename

    def _state_snapshot(self):
        """
        Return a copy of the attributes for ``_restore_state()``. Dictionaries and lists are copied in depth, while
        arrays are shared, since the methods that restore a snapshot replace arrays rather than write into them.
        """
        memo = {}

        def share_arrays(value):
            if isinstance(value, np.ndarray):
                memo[id(value)] = value
            elif isinstance(value, dict):
                for item in value.values():
                    share_arrays(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    share_arrays(item)

        share_arrays(self.__dict__)
        return copy.deepcopy(self.__dict__, memo)

    def _restore_state(self, state):
        """Put back the attributes of a ``_state_snapshot()``."""
        self.__dict__.clear()
        self.__dict__.update(state)

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

    def _log_variable(self):
        """Return the variable name of the object in ``log_rebuild()``, a Python identifier made from the file name."""
        name = self.filename
        if name.isidentifier() and not keyword.iskeyword(name) and name != "TMM":
            return name
        name = "".join(char if ("_" + char).isidentifier() else "_" for char in name)
        if not name.isidentifier() or keyword.iskeyword(name) or name == "TMM":
            name = f"treatment_{name}"
        return name

    def log_rebuild(self):
        """
        Return the list of calls needed to recreate this object. They run in a namespace that holds only ``TMM``, since
        the project folder is written as a string literal and an explicit ``freq`` vector comes with its numpy import.
        The log writes the project folder as it was given to the object. For an object without one it writes
        ``project_folder=None``, so the rebuilt object follows the working directory, as the original does. The logged
        variable is the file name when that is a Python identifier other than a keyword or ``TMM``. Otherwise each
        character an identifier does not allow becomes ``_``, and a name that then starts with a digit, is a keyword or
        is ``TMM`` gets the prefix ``treatment_``. The logged ``filename=`` keeps the file name.
        """
        variable = self._log_variable()
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
        # Without a project folder the object follows the working directory, and so does the logged one.
        project_folder = None if self._project_folder is None else self._path_literal(self._project_folder)

        logged_calls = [f"# {self.filename.capitalize()}",
                        *preamble,
                        f"{variable} = TMM("
                        f"{grid_args}"
                        f"project_folder={project_folder}, "
                        f"incidence='{self.incidence}', "
                        f"incidence_angle={self._incidence_angle}, "
                        f"diffuse_method='{self.diffuse_method}', "
                        f"filename={self.filename!r}"
                        f"{optional_constructor_args})"]

        for key, value in matrix.items():
            if key == "termination":
                if value["type"] == "constant_z":
                    logged_calls.append(f"{variable}.constant_z({self._log_number(self.z[0].item())})")
                elif value["type"] == "impedance_from_absorption" and "literature" in value:
                    logged_calls.extend(self._log_material_from_table(value))
                elif value["type"] == "impedance_from_absorption":
                    logged_calls.extend(self._log_impedance_from_absorption(value))
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
                        f"{variable}.porous_layer("
                        f"model='{value['model']}', "
                        f"t={float(value['thickness [mm]'])!r}, "
                        f"sigma={value['flow_resistivity [k*Pa*s/m²]']}"
                        f"{optional_args})"
                        )
                elif value["type"] == "air_layer":
                    logged_calls.append(f"{variable}.air_layer(t={float(value['thickness [mm]'])!r})")
                elif value["type"] == "perforated_panel_layer":
                    contact_arg = f", porous_contact={value['porous_contact']}" if value.get("porous_contact") else ""
                    end_correction_arg = ""
                    if value.get("method") in {"barrier", "eq_fluid", "bessel_ingard"} \
                            and value.get("end_correction") is not None:
                        end_correction_arg = f"end_correction='{value['end_correction']}', "
                    elif value.get("method") == "barrier_mpp" and value.get("end_correction") == "maa1987":
                        end_correction_arg = "end_correction='maa1987', "
                    logged_calls.append(
                        f"{variable}.perforated_panel_layer("
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
                        f"{variable}.slotted_panel_layer("
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
                        f"{variable}.rectangular_slot_layer("
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
                        f"{variable}.membrane_layer("
                        f"t={float(value['thickness [mm]'])!r}, "
                        f"rho={value['density [kg/m³]']}"
                        f"{self._mechanical_resistance_arg(value)}"
                        f")"
                        )
                elif value["type"] == "thin_plate_layer":
                    logged_calls.append(
                        f"{variable}.thin_plate_layer("
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
                        f"{variable}.porous_facing_layer("
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
                # The types without calculation parameters store params=None, and the call then omits it.
                params_arg = ""
                if value.get("params") is not None:
                    params = ", ".join(f"{name!r}: {self._log_number(item, keep_int=True)}"
                                       for name, item in value["params"].items())
                    params_arg = f", params={{{params}}}"
                logged_calls.append(f"{variable}.material_model('{value['type']}'{params_arg})")

        backing_data = self._backing_record(matrix)[1]
        if backing_data is not None:
            backing = backing_data.get("backing")
            if backing is None:
                logged_calls.append(
                    f"{variable}.compute("
                    f"rigid_backing={backing_data['rigid_backing']}, "
                    f"show_layers=True)"
                )
            else:
                logged_calls.append(
                    f"{variable}.compute("
                    f"backing='{backing}', "
                    f"conj={backing_data['impedance_conjugate']}, "
                    f"show_layers=True)"
                )
        else:
            logged_calls.append(f"{variable}.compute(rigid_backing=False, show_layers=True)")

        logged_calls += [f"{variable}.plot(figsize=(7, 4), plots=['alpha'], save_fig=True, timestamp=False)",
                         f"{variable}.save()"]

        return logged_calls

    @staticmethod
    def _path_literal(path):
        """
        Return Python source for the string ``path`` for ``log_rebuild()``. A raw-string literal keeps the backslashes
        of a Windows path readable. When no raw literal gives ``path`` back, as for a trailing backslash or a path with
        both kinds of quotes, the method returns ``repr(path)``.
        """
        for quote in ("'", '"'):
            literal = f"r{quote}{path}{quote}"
            try:
                if ast.literal_eval(literal) == path:
                    return literal
            except (SyntaxError, ValueError):
                pass
        return repr(path)

    @staticmethod
    def _log_number(value, keep_int=False):
        """
        Return ``value`` as Python source for ``log_rebuild()``.

        NumPy scalars and 0-d arrays become plain Python values, real numbers are written as floats (``float('nan')``
        and ``float('inf')`` for non-finite ones), complex numbers with a part that is not finite as
        ``complex(real, imag)`` with each part written so, and anything else uses ``repr``. With ``keep_int=True``
        integers stay integers.
        """
        if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and value.ndim == 0):
            value = value.item()
        if keep_int and isinstance(value, int) and not isinstance(value, bool):
            return repr(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            return repr(value) if np.isfinite(value) else f"float('{value}')"
        if isinstance(value, complex) and not np.isfinite(value):
            return f"complex({TMM._log_number(value.real)}, {TMM._log_number(value.imag)})"
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
            keys = ["type", "preset", "preset_overrides", "thickness [mm]", "density [kg/m³]",
                    "mechanical_resistance [-]"]
        elif layer_type == "thin_plate_layer":
            keys = ["type", "preset", "preset_overrides", "thickness [mm]", "density [kg/m³]", "young_modulus [GPa]",
                    "poisson_ratio [-]", "loss_factor [-]", "critical_frequency [Hz]", "thin_plate_limit [Hz]",
                    "mechanical_resistance [-]"]
        elif layer_type == "porous_facing_layer":
            keys = ["type", "method", "preset", "preset_overrides", "thickness [mm]", "surface_mass [kg/m2]",
                    "resistance [Pa*s/m]"]
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
            keys.extend(["end_correction", "end_correction_length [mm]", "exit_radiation"])
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
        elif layer_type == "impedance_from_absorption":
            keys = list(ABSORPTION_REPORT_KEYS)
        elif layer_type == "backing":
            keys = ["type", "backing", "rigid_backing", "diffuse_method", "impedance_conjugate"]
        else:
            keys = [key for key in layer_data.keys() if key != "matrix"]

        items = [(key, layer_data[key]) for key in keys if key in layer_data and layer_data[key] is not None]
        literature = layer_data.get("literature") if layer_type == "impedance_from_absorption" else None
        if isinstance(literature, dict):
            # A material_from_table() termination names its table rows and source after its type.
            items[1:1] = [(key, literature[key]) for key in ("row", "source", "scattering_row")
                          if literature.get(key) is not None]
        return items

    def _report_layer_keys(self):
        """Return matrix keys in the order used for layer reports."""
        return utils.report_layer_keys(self.matrix)

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
        validity = utils.check_flag("show_layers", "validity", validity)
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
        if self._has_paris_termination_alpha():
            # The absorption of this termination is the Paris value over 0 to 90 degrees, whatever the settings say.
            print(f"\tDiffuse method: paris ({self._termination_name()}() termination, whatever diffuse_method and "
                  f"incidence_angle say)")
            print("\tAngle: 0° - 90°\n")
        elif self.incidence == "diffuse":
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
        if entry is None and ("termination" in self.matrix or "material_model" in self.matrix):
            name = self._termination_name()
            raise ValueError(f"The causality bound needs a layer stack with a depth on a rigid backing, and this "
                             f"treatment is a surface impedance set by {name}().")
        if entry is None:
            raise RuntimeError("The causality bound needs a computed treatment. Call compute() first.")

        backing = entry.get("backing", "rigid" if entry.get("rigid_backing", True) else "air")
        if backing != "rigid":
            raise ValueError(
                f"The causality bound is derived for a rigid-backed layer with no transmission, but this treatment "
                f"was computed with backing={backing!r}. Recompute with backing='rigid'."
            )

        return entry

    def _static_bulk_modulus_ratio(self, bulk_modulus_ratio, thermal, method):
        """Return the static ``Beff / B0`` to use, its provenance, and the per-layer breakdown, for ``method``."""
        overridden = bulk_modulus_ratio is not None
        if overridden:
            utils.check_positive(method, "bulk_modulus_ratio", "the static bulk modulus ratio Beff/B0 [-]",
                                 bulk_modulus_ratio)
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
            Static ``Beff / B0`` overriding the value derived from the layer metadata, above 0; another value raises a
            ``ValueError``. Required for stacks containing a membrane layer, a constant impedance or a material model,
            whose static compressibility is not described by Wood's law. A porosity is not accepted in its place because
            it cannot express a ratio below unity, which the isothermal limit and a membrane's compliance both need.
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

        ratio_value, source, layers = self._static_bulk_modulus_ratio(bulk_modulus_ratio, thermal, "causality_check")
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
        show_fig = utils.check_flag("plot_causality", "show_fig", show_fig)
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
        ``angle_weights`` options. The default sample is a 3.162 m square, about 10 m2. The treatment must be computed
        with ``incidence="diffuse"``. The angular range of ``incidence_angle`` sets the integration range and is printed
        in the title. For the termination of ``impedance_from_absorption()`` or ``material_from_table()`` every curve,
        the Paris average included, uses the range that ``finite_sample_absorption()`` takes for it, and the title shows
        that range.

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
        show_fig = utils.check_flag("plot_finite_sample_absorption", "show_fig", show_fig)
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
            Slack allowed outside ``[0, 1]`` before a sample counts as non-passive, for absorbing float noise, at least
            0. Another value raises a ``ValueError``.
        angles : bool, optional
            Also screen every stored angle-dependent absorption column, not just the treatment's own ``alpha``.

        Returns
        -------
        Dictionary with ``passive``, the observed ``alpha_range``, the contiguous ``bands [Hz]`` that offend, the number
        of offending samples, and the spectra that were screened.
        """
        tolerance = utils.check_non_negative("check_passivity", "tolerance", "the slack allowed outside [0, 1] [-]",
                                             tolerance)
        angles = utils.check_flag("check_passivity", "angles", angles)
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
            Lower band edge in Hz, above 0. Ignored when solving for it.
        f_high : float or None, optional
            Upper band edge in Hz, above 0. ``None`` or positive infinity means the target extends to infinite
            frequency.
        depth : float, optional
            Treatment thickness in millimetres, above 0. Defaults to this treatment's depth. Ignored when solving for
            it.
        bulk_modulus_ratio : float, optional
            Static ``Beff / B0`` overriding the value derived from the layer metadata, above 0. See
            ``causality_check()`` for why a porosity is not accepted in its place. ``f_low``, ``f_high``, ``depth`` or
            ``bulk_modulus_ratio`` outside these ranges raises a ``ValueError``, also where the branch ignores it.
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
        causality.check_limit_inputs(f_low, f_high, depth)
        ratio_value, _, _ = self._static_bulk_modulus_ratio(bulk_modulus_ratio, thermal, "causality_limit")
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
            Fractional-octave resolution: ``1`` for octave bands, ``3`` for third-octave bands. A value that is not a
            whole number of at least 1 raises a ``ValueError``.
        view : bool, optional
            If True, plot the narrowband and band-averaged absorption together.
        show_table : bool, optional
            If True, print the band values as a table.

        Returns
        -------
        Bands' center frequency array and filtered absorption array.
        """
        utils.check_number("filter_alpha", "n_oct", "the bandwidth denominator of the bands", n_oct,
                           "a whole number of at least 1", lambda x: x >= 1 and float(x).is_integer())
        view = utils.check_flag("filter_alpha", "view", view)
        show_table = utils.check_flag("filter_alpha", "show_table", show_table)
        self._raise_if_stale_results("filter_alpha()")
        # bands, result = pytta.utils.filter_values(self.freq, self.alpha, nthOct=n_oct)
        bands, result = utils.filter_values(self.freq, self.alpha, n_oct=n_oct)

        # Plot
        if view:
            _, _ = plot.oct_filter(self.freq, self.alpha, bands, result, n_oct, "Absorption Coefficient [-]",
                                   filename=self.filename, project_folder=self.project_folder, **kwargs)
            plt.show()

        if show_table:
            freq_bands = []
            absorption = []
            absorption_percentual = []
            for i in range(len(bands)):
                freq_bands.append(float(bands[i]))
                absorption.append(float(f"{result[i]:0.2f}"))
                absorption_percentual.append(float(f"{result[i] * 100:0.0f}"))
            data = {"Bands [Hz]": freq_bands, "Absorption [-]": absorption, "Absorption [%]": absorption_percentual}
            df = pandas.DataFrame(data=data).set_index("Bands [Hz]").T

            # Two decimals for this table only; the global pandas options stay as they were.
            with pandas.option_context("display.precision", 2):
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
        absorption. When ``self.scat`` is set, every export also writes it on ``self.freq`` as a ``Scattering [-]``
        column after the other columns.

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
        timestamp = utils.check_flag("save2sheet", "timestamp", timestamp)
        metadata = utils.check_flag("save2sheet", "metadata", metadata)
        export_all = utils.check_flag("save2sheet", "export_all", export_all)
        utils.check_choice("save2sheet", "ext", "the file extension", ext, (".xlsx", ".csv"))
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
        metadata supported by the current package version. A stored layer with a value that a new call would refuse
        raises a ``ValueError`` that names the layer, as ``rebuild()`` describes. A file that cannot be read or rebuilt
        raises, and the object keeps the state it had before the call.
        """
        project_folder = self._project_folder
        # A refused or unreadable file must not leave this object half-overwritten.
        state = self._state_snapshot()
        try:
            # The loader sets only the attributes the file holds, and files written before the flag existed have none.
            self._df_from_default = False
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
            if isinstance(getattr(self, "_incidence_angle", None), np.ndarray):
                # The file returns the list the constructor stores as an array.
                self._incidence_angle = self._incidence_angle.tolist()
            self._drop_non_positive_frequencies(filename)
            self.rebuild()
        except BaseException:
            self._restore_state(state)
            raise
        print(filename + ".h5 loaded successfully.")

    def _drop_non_positive_frequencies(self, filename):
        """
        Remove 0 Hz and negative points from a grid restored by ``load()``. Earlier versions accepted them, and the
        loader restores the grid without the setters' checks. The remaining points become an explicit ``freq`` vector.
        A grid with values that are not finite, as a logarithmic grid from 0 Hz gives, is refused. A derived ``fmin`` at
        or below 0 Hz moves to the first remaining point, so clearing ``freq`` later cannot bring 0 Hz back.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            grid = np.asarray(self.freq, dtype=float)
        if not np.all(np.isfinite(grid)):
            raise ValueError(
                f"{filename}.h5 stores a frequency grid with values that are not finite numbers, as a logarithmic grid "
                f"from fmin = 0 Hz gives. Create the treatment again with a positive fmin."
            )
        keep = grid > 0
        if keep.all():
            return
        if not keep.any():
            raise ValueError(f"{filename}.h5 has no positive frequency to compute at.")
        self._freq = grid[keep]
        if self._fmin <= 0:
            self._fmin = float(self._freq[0])
        dropped = int((~keep).sum())
        warnings.warn(
            f"{filename}.h5 has {dropped} {'frequency' if dropped == 1 else 'frequencies'} at or below 0 Hz, where the "
            f"models are not defined. "
            f"They were removed, and the remaining points now form an explicit freq vector from "
            f"{self._freq[0]:g} to {self._freq[-1]:g} Hz.",
            UserWarning,
            stacklevel=3,
        )

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
        show_fig = utils.check_flag("plot", "show_fig", show_fig)
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
