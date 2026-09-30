# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from functools import partial

import numpy as np
from numpy.polynomial import Legendre
from scipy import linalg
from scipy.special import hyp2f1

from .._fiff.proj import Projection
from .._ola import _COLA, _Storer
from ..annotations import _annotations_starts_stops
from ..bem import _fit_spheroid, get_fitting_dig
from ..io import BaseRaw
from ..surface import _normal_orth
from ..transforms import _deg_ord_idx, _get_n_moments
from ..utils import _check_option, _validate_type, logger, verbose_static
from .maxwell import (
    _check_usable,
    _do_tSSS,
    _get_mf_picks_fix_mags,
    _legendre_theta_terms,
    _prep_mf_coils,
    _remove_meg_projs_comps,
    _sph_harm_norm,
    _tabular_legendre,
)


@verbose_static("skip_by_annotation_maxwell")
def amm_filter(
    raw,
    int_order=9,
    ext_order=2,
    *,
    spheroid="dig",
    st_duration=10.0,
    st_correlation=0.98,
    skip_by_annotation=("edge", "bad_acq_skip"),
    verbose=None,
):
    """Adaptive multipole model (AMM) denoising of MEG data.

    Parameters
    ----------
    raw : instance of Raw
        Data to be filtered. Bad channels are excluded from (and left untouched by)
        the processing. The good MEG channels must all be magnetometers or all be
        gradiometers.
    int_order : int
        Order of the internal component of the prolate spheroidal harmonic
        expansion.
    ext_order : int
        Order of the external component of the prolate spheroidal harmonic
        expansion.
    spheroid : str
        What to fit the reference prolate spheroid to. Can be ``"dig"`` (default) to
        use the head digitization points (see :func:`mne.bem.get_fitting_dig`), or
        ``"meg"`` to use the MEG sensor positions, which is useful for on-scalp
        sensors when no digitization is available.
    st_duration : float | None
        Window duration in seconds for the temporal step, which removes
        components of the internal space that are temporally correlated with the
        residual (neither internal nor external) space. If None, only the spatial
        projection is performed.
    st_correlation : float
        Correlation limit between the internal and residual subspaces used to
        reject components during the temporal step.
    skip_by_annotation : str | list of str
        If a string (or list of str), any annotation segment that begins
        with the given string will not be included in filtering, and
        segments on either side of the given excluded annotated segment
        will be filtered separately (i.e., as independent signals).
        The default ``('edge', 'bad_acq_skip')`` will separately filter
        any segments that were concatenated by :func:`mne.concatenate_raws`
        or :meth:`mne.io.Raw.append`, or separated during acquisition.
        To disable, provide an empty list.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    raw_amm : instance of Raw
        The filtered data.

    See Also
    --------
    compute_proj_hfc
    maxwell_filter

    Notes
    -----
    This implements the adaptive multipole model of :footcite:`TierneyEtAl2024`.
    The magnetic field is modeled with prolate spheroidal harmonics using a
    spheroid fit to the head (or sensors). The spatial step projects the data onto
    the part of the internal space that is orthogonal to the external space, which
    unlike the oblique projection used by :func:`maxwell_filter` cannot amplify
    sensor noise. It is stored as an (applied) projector in
    ``raw_amm.info["projs"]`` so that forward modeling, source estimation, and rank
    estimation account for it. The optional temporal step is analogous to tSSS.

    The method is designed for on-scalp (e.g., OPM) magnetometer arrays. To be
    representable as a projector, the projection is computed in the native units of
    the channels, so magnetometers and gradiometers (T vs T/m) cannot be processed
    together. Only MEG channels are processed; reference channels are ignored.
    Projectors must not have been applied, and existing MEG projectors are removed.

    .. warning::
       AMM can remove part of the brain signal along with the interference: in the
       worst case up to two thirds for single-axis (radial) arrays and one half for
       dual-axis arrays, but negligibly little for triaxial arrays
       :footcite:`TierneyEtAl2024`. Because the projector is stored in the info,
       source modeling accounts for this, but take care when interpreting the
       sensor-level data.

    .. versionadded:: 1.14

    References
    ----------
    .. footbibliography::
    """
    _validate_type(raw, BaseRaw, "raw")
    _check_usable(raw, ignore_ref=True)
    _check_option("spheroid", spheroid, ("dig", "meg"))
    st_correlation = float(st_correlation)
    if not 0.0 < st_correlation <= 1.0:
        raise ValueError(f"Need 0 < st_correlation <= 1., got {st_correlation}")
    if spheroid == "dig" and raw.info["dev_head_t"] is None:
        raise RuntimeError(
            'spheroid="dig" requires info["dev_head_t"], consider using spheroid="meg"'
        )
    raw = raw.copy().load_data()
    _remove_meg_projs_comps(raw, ignore_ref=True)
    info, sfreq = raw.info, raw.info["sfreq"]
    meg_picks, mag_picks, grad_picks, good_mask, _ = _get_mf_picks_fix_mags(
        info, int_order, ext_order, ignore_ref=True, verbose=False
    )
    n_mag, n_grad = good_mask[mag_picks].sum(), good_mask[grad_picks].sum()
    if n_mag and n_grad:
        raise ValueError(
            "AMM requires a single MEG channel type, got "
            f"{n_mag} good magnetometers and {n_grad} good gradiometers. Pick one "
            "type (or mark the other as bad), or use maxwell_filter instead."
        )
    good_picks = meg_picks[good_mask]
    logger.info(
        f"    Processing {len(good_picks)} MEG channels "
        f"({len(meg_picks) - len(good_picks)} bad excluded)"
    )
    if spheroid == "dig":
        points = get_fitting_dig(info, verbose=False)
    else:
        points = np.array([info["chs"][pick]["loc"][:3] for pick in good_picks])
    origin, axis, a, b = _fit_spheroid(points)
    exp = dict(
        origin=origin, axis=axis, a=a, b=b, int_order=int_order, ext_order=ext_order
    )
    coils = _prep_mf_coils(info, ignore_ref=True, head_frame=spheroid == "dig")
    H = _amm_basis(exp, coils)[good_mask]
    n_in = _get_n_moments(int_order)
    n_out = H.shape[1] - n_in
    logger.info(f"    Using {n_in} internal and {n_out} external moments")
    # QR of [H_out, H_in] gives orthonormal bases for the external space, the
    # internal space orthogonalized with respect to it (U, Eq. 18), and the rest
    H /= np.linalg.norm(H, axis=0)
    Q = linalg.qr(np.concatenate([H[:, n_in:], H[:, :n_in]], axis=1))[0]
    U = Q[:, n_out : n_out + n_in]
    op_in = U @ U.T
    if st_duration is not None:  # temporal step on unprojected data (Eqs. 21-22)
        st_duration = int(round(float(st_duration) * sfreq))
        onsets, ends = _annotations_starts_stops(raw, skip_by_annotation, invert=True)
        max_samps = (ends - onsets).max()
        if not 0.0 < st_duration <= max_samps + 1.0:
            raise ValueError(
                f"st_duration ({st_duration / sfreq:0.1f}s) must be between 0 and the "
                "longest contiguous duration of the data "
                f"({max_samps / sfreq:0.1f}s)."
            )
        Q_inter = Q[:, n_out + n_in :]
        op_inter = Q_inter @ Q_inter.T
        step = int(round(raw.buffer_size_sec * sfreq))
        for onset, end in zip(onsets, ends):
            n = end - onset
            tsss_valid = n >= st_duration
            cola = _COLA(
                partial(
                    _do_tSSS,
                    st_correlation=st_correlation,
                    n_positions=1,
                    tsss_valid=tsss_valid,
                    sfreq=sfreq,
                ),
                _Storer(raw._data[:, onset:end], picks=good_picks),
                n,
                min(st_duration, n),
                (st_duration + 1) // 2 if tsss_valid else 0,
                sfreq,
                "hann" if tsss_valid else "boxcar",
                name="AMM-COLA",
                offset=onset,
            )
            for start in range(onset, end, step):
                data = raw._data[good_picks, start : min(start + step, end)]
                data_in = op_in @ data
                # _do_tSSS performs CCA between inner and intermediate (Sec 2.3)
                cola.feed(data_in, data_in, op_inter @ data)
    # Spatial step (Eq. 18) as an SSP projector: this is a no-op for data already
    # processed by the temporal step, which leaves it in the span of U
    # (but harmless to reapply)
    proj_data = np.delete(Q, slice(n_out, n_out + n_in), axis=1).T
    proj = Projection(
        data=dict(
            col_names=[info["ch_names"][pick] for pick in good_picks],
            row_names=None,
            data=proj_data,
            ncol=len(good_picks),
            nrow=len(proj_data),
        ),
        desc=f"AMM-in-L{int_order}-out-L{ext_order}",
    )
    raw.add_proj(proj).apply_proj(projs=proj)
    return raw


def _amm_basis(exp, all_coils):
    """Compute the prolate spheroidal harmonic basis for given conditions.

    Same contract as ``_sss_basis``, with ``exp`` additionally containing the
    spheroid major ``axis`` (unit vector) and semi-axes ``a`` and ``b``.
    """
    rmags, cosmags, bins, n_coils = all_coils[:4]
    int_order, ext_order = exp["int_order"], exp["ext_order"]
    a, b = exp["a"], exp["b"]

    # Focal distance, floored so that a (nearly) spherical fit stays finite
    c = np.sqrt(max(a * a - b * b, (1e-3 * a) ** 2))

    # Rotate so that the spheroid major axis is +z
    rot = _normal_orth(np.array(exp["axis"], float))
    x, y, z = rot @ (rmags - exp["origin"]).T
    nx, ny, nz = rot @ cosmags.T

    # Prolate spheroidal coordinates (Fig. 1a): r_maj_rel = r_maj / c, cos_pol, phi
    rho = np.hypot(x, y)
    d_plus, d_minus = np.hypot(rho, z - c), np.hypot(rho, z + c)
    r_maj_rel = (d_plus + d_minus) / (2 * c)
    cos_pol = np.clip((d_minus - d_plus) / (2 * c), -1.0, 1.0)
    sin_pol = np.sqrt(1.0 - cos_pol * cos_pol)
    phi = np.arctan2(y, x)
    sq = np.sqrt(r_maj_rel * r_maj_rel - 1.0)
    denom = c * (r_maj_rel * r_maj_rel - cos_pol * cos_pol)

    # Coil normals dotted with the gradient terms for d/dr_maj_rel, d/dtheta, and
    # d/dphi / sin(theta) (which are all finite on the major axis)
    n_rho = nx * np.cos(phi) + ny * np.sin(phi)
    n_r = (r_maj_rel * sq * sin_pol * n_rho + sq * sq * cos_pol * nz) / denom
    n_pol = (sq * cos_pol * n_rho - r_maj_rel * sin_pol * nz) / denom
    n_az = (ny * np.cos(phi) - nx * np.sin(phi)) / (c * sq)
    max_order = max(int_order, ext_order)
    L = _tabular_legendre(cos_pol, max_order + 1)
    n_in, n_out = _get_n_moments([int_order, ext_order])
    S_tot = np.empty((n_coils, n_in + n_out))
    starts = np.flatnonzero(np.diff(bins, prepend=-1))  # bins is sorted
    r_maj_rel = np.append(r_maj_rel, a / c)  # normalize radial functions at a (Eq. 6)
    for degree in range(1, max_order + 1):
        for order in range(degree + 1):
            P = L[degree][order]
            dP, P_sin = _legendre_theta_terms(L, degree, order)
            # Eq. 9
            factor = _sph_harm_norm(order, degree) * (np.sqrt(2) if order else 1.0)
            ord_phi = order * phi
            cos_order, sin_order = np.cos(ord_phi), np.sin(ord_phi)
            for offset, this_order, internal in (
                (0, int_order, True),
                (n_in, ext_order, False),
            ):
                if degree > this_order:
                    continue
                R, dR = _amm_radial(degree, order, r_maj_rel, internal)
                for sign in (1, -1) if order else (1,):
                    if sign > 0:
                        trig, dtrig = cos_order, -order * sin_order
                    else:
                        trig, dtrig = sin_order, order * cos_order
                    grad = factor * (
                        (dR * n_r * P + R * n_pol * dP) * trig
                        + R * n_az * P_sin * dtrig
                    )
                    idx = offset + _deg_ord_idx(degree, sign * order)
                    S_tot[:, idx] = np.add.reduceat(grad, starts)
    return S_tot


def _amm_radial(degree, order, x, internal):
    """Compute the radial function and its derivative, normalized by x[-1].

    With x = r_maj / c, internal uses the Legendre function of the second kind
    (Eq. 7), external the first kind (Eq. 8), each written as
    (x^2 - 1)^(m/2) * x^p * s(x).
    """
    if internal:  # s = 2F1(a1, a2; a3; 1 / x^2), using d2F1/dz = a1 a2 / a3 2F1(+1)
        power = -(degree + order + 1)
        a1, a2, a3 = (degree + order + 1) / 2, (degree + order + 2) / 2, degree + 1.5
        z = 1.0 / (x * x)
        s = hyp2f1(a1, a2, a3, z)
        ds = -2 * z / x * (a1 * a2 / a3) * hyp2f1(a1 + 1, a2 + 1, a3 + 1, z)
        # equivalent to (but slower than, and non-convergent near the foci) the
        # series in Eq. 7 that a naive implementation would sum:
        # term, s, ds = np.ones_like(x), np.ones_like(x), np.zeros_like(x)
        # for k in range(1000):
        #     term *= (a1 + k) * (a2 + k) / ((a3 + k) * (k + 1)) * z
        #     s += term
        #     ds -= 2 * (k + 1) * term / x
    else:  # polynomial: m-th derivative of the Legendre polynomial
        power = 0
        poly = Legendre.basis(degree).deriv(order)
        s, ds = poly(x), poly.deriv()(x)
    f = (x * x - 1.0) ** (order / 2) * x**power
    df = f * (s * (order * x / (x * x - 1.0) + power / x) + ds)
    f *= s
    return f[:-1] / f[-1], df[:-1] / f[-1]
