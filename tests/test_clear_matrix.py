"""Tests for TMM.clear_matrix size-reduction options."""

import numpy as np
import pytest

from tmm.tmm import TMM


@pytest.fixture
def computed_tmm():
    """A computed diffuse-incidence treatment with a couple of layers."""
    treatment = TMM(fmin=100, fmax=2000, df=5, incidence="diffuse")
    treatment.perforated_panel_layer(t=19, d=6, s=60)
    treatment.porous_layer(model="mac", t=50, sigma=27)
    treatment.compute(rigid_backing=True, show_layers=False)
    return treatment


def test_clear_matrix_default_keeps_z_angle(computed_tmm):
    """The default call must stay backwards compatible and preserve z_angle."""
    z_angle = computed_tmm.z_angle.copy()

    computed_tmm.clear_matrix()

    assert computed_tmm._z_angle is not None
    np.testing.assert_array_equal(computed_tmm.z_angle, z_angle)
    assert all(entry.get("matrix") is None
               for entry in computed_tmm.matrix.values() if "matrix" in entry)


def test_clear_z_angle_keeps_normal_incidence(computed_tmm):
    """Trimming z_angle must leave alpha_angle() untouched."""
    alpha_normal = computed_tmm.alpha_angle().copy()
    alpha_diffuse = computed_tmm.alpha.copy()
    n_freq = len(computed_tmm.freq)

    computed_tmm.clear_matrix(clear_z_angle=True)

    assert computed_tmm._z_angle.shape == (n_freq, 1)
    np.testing.assert_allclose(computed_tmm.alpha_angle(), alpha_normal)
    np.testing.assert_allclose(computed_tmm.alpha, alpha_diffuse)


def test_clear_z_angle_without_normal_incidence_discards_it(computed_tmm):
    """Opting out of keep_normal_incidence drops z_angle entirely."""
    computed_tmm.clear_matrix(clear_z_angle=True, keep_normal_incidence=False)

    assert computed_tmm._z_angle is None
    # The property falls back to zeros, so alpha_angle() is no longer meaningful.
    assert np.allclose(computed_tmm.alpha_angle(), 0)


def test_clear_z_angle_shrinks_serialized_size(computed_tmm):
    """The trimmed object must be dramatically smaller once pickled."""
    import copy
    import pickle

    baseline = copy.deepcopy(computed_tmm)
    baseline.clear_matrix()
    computed_tmm.clear_matrix(clear_z_angle=True)

    trimmed_size = len(pickle.dumps(computed_tmm))
    assert trimmed_size < len(pickle.dumps(baseline)) / 10


def test_clear_z_angle_survives_pickle_round_trip(computed_tmm):
    """alpha_angle() must still be correct after a pickle round trip."""
    import pickle

    alpha_normal = computed_tmm.alpha_angle().copy()
    computed_tmm.clear_matrix(clear_z_angle=True)

    restored = pickle.loads(pickle.dumps(computed_tmm))

    np.testing.assert_allclose(restored.alpha_angle(), alpha_normal)


def test_reduce_size_can_keep_normal_incidence(computed_tmm):
    """reduce_size() keeps its old default but can now retain normal incidence."""
    alpha_normal = computed_tmm.alpha_angle().copy()

    computed_tmm.reduce_size(keep_normal_incidence=True)

    np.testing.assert_allclose(computed_tmm.alpha_angle(), alpha_normal)


def test_reduce_size_default_discards_z_angle(computed_tmm):
    """reduce_size() with no arguments behaves as it always has."""
    computed_tmm.reduce_size()

    assert computed_tmm._z_angle is None
