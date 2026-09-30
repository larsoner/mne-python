# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import subspace_angles

from mne import pick_info, pick_types
from mne._fiff.proj import make_projector
from mne.bem import _fit_sphere, _fit_spheroid, get_fitting_dig
from mne.datasets import testing
from mne.forward import _prep_meg_channels
from mne.forward._compute_forward import _magnetic_dipole_field_vec
from mne.io import RawArray, read_info
from mne.preprocessing import amm_filter
from mne.preprocessing._amm import _amm_basis
from mne.preprocessing.maxwell import _prep_mf_coils, _sss_basis

data_path = testing.data_path(download=False)
raw_fname = data_path / "SSS" / "test_move_anon_raw.fif"
opm_fname = data_path / "OPM" / "opm-evoked-ave.fif"


def _resid(b, basis):
    """Compute the relative residual of fitting b with basis."""
    return np.linalg.norm(b - basis @ np.linalg.lstsq(basis, b)[0]) / np.linalg.norm(b)


@testing.requires_testing_data
def test_amm_basis():
    """Test the prolate spheroidal harmonic basis."""
    info = read_info(raw_fname)
    info = pick_info(info, pick_types(info, meg=True))
    coils = _prep_mf_coils(info)
    exp = dict(origin=(0.0, 0.0, 0.04), int_order=9, ext_order=2, axis=(0, 1, 0))
    n_in = 99
    # sphere limit: same spans as SSS (external spans agree for any spheroid)
    S = _sss_basis(exp, coils)
    for b, tol in ((0.09, 1e-3), (0.0899, 2.0), (0.08, None)):
        H = _amm_basis(dict(exp, a=0.09, b=b), coils)
        assert np.isfinite(H).all()
        if tol is not None:
            assert np.rad2deg(subspace_angles(H[:, :n_in], S[:, :n_in])).max() < tol
        assert np.rad2deg(subspace_angles(H[:, n_in:], S[:, n_in:])).max() < 1e-6
    # physics: an internal dipole is captured by H_in, a far one by H_out
    H_in, H_out = H[:, :n_in], H[:, n_in:]
    defs = _prep_meg_channels(info, head_frame=False, verbose=False)["defs"]
    rrs = np.array([[0.01, 0.03, 0.05], [0.0, 3.0, 3.0]])
    b_int, b_ext = _magnetic_dipole_field_vec(rrs, defs)[[0, 4]]
    assert _resid(b_int, H_in) < 0.005
    assert _resid(b_int, H_out) > 0.8
    assert _resid(b_ext, H_out) < 0.1
    assert _resid(b_ext, H_in) > 0.5
    # finite (and continuous) exactly on the major axis
    rmags = np.array([[0.0, 0.12, 0.04], [1e-6, 0.12, 0.04]]).repeat(3, axis=0)
    point_coils = (rmags, np.tile(np.eye(3), (2, 1)), np.arange(6), 6)
    H = _amm_basis(dict(exp, a=0.1, b=0.08), point_coils)
    assert np.isfinite(H).all()
    assert_allclose(H[:3], H[3:], rtol=1e-3, atol=1e-3 * np.abs(H).max())


@pytest.mark.parametrize("half", (False, True))
def test_fit_spheroid(half):
    """Test that a known spheroid is recovered."""
    rng = np.random.default_rng(0)
    center, axis = np.array([0.01, -0.02, 0.04]), np.array([0.6, 0.8, 0])
    a, b = 0.1, 0.08
    pts = rng.normal(size=(500, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    if half:  # helmet-like coverage
        pts = pts[pts[:, 2] > 0]
    other = np.array([-0.8, 0.6, 0.0])
    pts = pts[:, [0]] * other * b + pts[:, [1]] * axis * a + pts[:, [2]] * [0, 0, b]
    pts += center + 1e-4 * rng.normal(size=pts.shape)
    got_center, got_axis, got_a, got_b = _fit_spheroid(pts)
    assert_allclose(got_center, center, atol=1e-3)
    assert_allclose(np.abs(got_axis @ axis), 1.0, atol=1e-3)
    assert_allclose([got_a, got_b], [a, b], atol=1e-3)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname, meg, spheroid, int_order, dist, ext_tol, corr_tol",
    [
        # 86-channel dual-axis OPM (too few channels for int_order=9), which can
        # lose up to half of the signal (Tierney et al. 2024, Sec. 4.1)
        (opm_fname, True, "meg", 6, 3.0, 0.005, 0.65),
        # Neuromag with head digitization (grads have more distant L=2 content)
        (raw_fname, "grad", "dig", 9, 10.0, 0.05, 0.9),
    ],
)
def test_amm_filter(fname, meg, spheroid, int_order, dist, ext_tol, corr_tol):
    """Test AMM filtering of simulated internal and external sources."""
    info_all = read_info(fname)
    info = pick_info(info_all, pick_types(info_all, meg=meg, ref_meg=False))
    with info._unlock():
        info["projs"], info["sfreq"] = [], 1000.0
    head_frame = spheroid == "dig"
    defs = _prep_meg_channels(info, head_frame=head_frame, verbose=False)["defs"]
    if head_frame:
        center = _fit_sphere(get_fitting_dig(info))[1]
    else:
        center = _fit_sphere(np.array([ch["loc"][:3] for ch in info["chs"]]))[1]
    rrs = center + np.array([[0.01, 0.02, 0.0], [0.6, 0.48, 0.64]]) * [[1], [dist]]
    b_int, b_ext = _magnetic_dipole_field_vec(rrs, defs)[[0, 4]]
    b_ext *= 100 * np.abs(b_int).max() / np.abs(b_ext).max()
    times = np.arange(2000) / info["sfreq"]
    sig_int = np.outer(b_int, np.sin(2 * np.pi * 7 * times))
    noise = np.random.default_rng(0).normal(
        scale=1e-2 * np.abs(b_int).max(), size=sig_int.shape
    )
    data = sig_int + np.outer(b_ext, np.sin(2 * np.pi * 3 * times)) + noise
    raw = RawArray(data, info)
    raw.info["bads"] = [raw.ch_names[0]]
    kwargs = dict(int_order=int_order, spheroid=spheroid)
    raw_amm = amm_filter(raw, st_duration=None, **kwargs)
    (proj,) = raw_amm.info["projs"]
    assert proj["active"] and proj["desc"] == f"AMM-in-L{int_order}-out-L2"
    good = [name for name in raw.ch_names if name not in raw.info["bads"]]
    P = make_projector([proj], good)[0]
    n_in = int_order * (int_order + 2)
    assert_allclose(P, P.T, atol=1e-12)
    assert_allclose(P @ P, P, atol=1e-12)
    assert np.linalg.matrix_rank(P) == n_in
    assert_allclose(
        np.var(P @ noise[1:]) / np.var(noise[1:]), n_in / len(good), rtol=0.05
    )
    assert_allclose(
        raw_amm.get_data(good), P @ data[1:], atol=1e-12 * np.abs(data).max()
    )
    assert_allclose(raw_amm.get_data(raw.info["bads"]), data[:1], rtol=1e-12)
    assert np.linalg.norm(P @ b_ext[1:]) / np.linalg.norm(b_ext[1:]) < ext_tol
    # orthogonal projection: the correlation with the internal field is ||P b|| / ||b||
    assert np.linalg.norm(P @ b_int[1:]) / np.linalg.norm(b_int[1:]) > corr_tol
    # temporal step removes the remaining interference
    raw_st = amm_filter(raw, st_duration=1.0, **kwargs)
    err_st, err = (
        np.linalg.norm(r.get_data(good) - P @ sig_int[1:]) for r in (raw_st, raw_amm)
    )
    assert err_st < 0.5 * err
    # errors
    with pytest.raises(ValueError, match="Invalid value for the 'spheroid'"):
        amm_filter(raw, spheroid="foo")
    with pytest.raises(ValueError, match="st_correlation"):
        amm_filter(raw, st_correlation=0.0, **kwargs)
    with pytest.raises(ValueError, match="st_duration"):
        amm_filter(raw, st_duration=10.0, **kwargs)
    with pytest.raises(RuntimeError, match="cannot be applied"):
        amm_filter(raw_amm, **kwargs)
    if meg == "grad":  # mixed coil types, unless only one of them is good
        raw_all = RawArray(np.zeros((len(info_all["ch_names"]), 1)), info_all)
        with pytest.raises(ValueError, match="single MEG channel type"):
            amm_filter(raw_all, st_duration=None)
        raw_all.info["bads"] = [
            raw_all.ch_names[pick] for pick in pick_types(info_all, meg="mag")
        ]
        amm_filter(raw_all, st_duration=None)
    with raw.info._unlock():
        raw.info["dev_head_t"] = None
    with pytest.raises(RuntimeError, match="dev_head_t"):
        amm_filter(raw, spheroid="dig")
