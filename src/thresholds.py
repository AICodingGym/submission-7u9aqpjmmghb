"""Rounding-threshold optimization for QWK-based essay scoring.

Ridge (and most regressors) emit a continuous score; the competition metric —
quadratic weighted kappa — is computed on integer classes 1..6. This module
turns the continuous prediction into integers by learning 5 cut points that
maximize QWK.

Public API
----------
quadratic_weighted_kappa(y_true, y_pred)   -> float
apply_thresholds(pred, thresholds)         -> np.ndarray[int]  (values 1..6)
optimize_thresholds(oof_pred, y_true, ...) -> ThresholdResult

Leakage guarantee
-----------------
``optimize_thresholds`` accepts ONLY out-of-fold predictions and their matching
ground truth. There is deliberately no `test` parameter, so test information
cannot influence the thresholds. Fit on OOF, then apply the returned cut points
to the test predictions with ``apply_thresholds`` at call sites.

No heavy deps: numpy, scipy.optimize, sklearn.metrics only.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

SCORE_MIN, SCORE_MAX = 1, 6
LABELS = [1, 2, 3, 4, 5, 6]                                  # fixed holistic scale
INITIAL_THRESHOLDS = np.array([1.5, 2.5, 3.5, 4.5, 5.5])    # midpoints between scores
_STRICT_EPS = 1e-6                                           # min gap for strict monotonicity


class ThresholdResult(NamedTuple):
    thresholds: np.ndarray   # strictly increasing cut points
    qwk: float               # best OOF QWK achieved
    initial_qwk: float       # OOF QWK with INITIAL_THRESHOLDS (for reference)
    n_iter: int              # optimizer iterations used
    success: bool            # optimizer convergence flag


# placeholder-for-append


def quadratic_weighted_kappa(y_true, y_pred) -> float:
    """Quadratic weighted kappa on the fixed 1..6 label set.

    ``labels=LABELS`` is passed explicitly so the histogram matrix is always
    6x6 even when a score is absent from ``y_pred`` — otherwise the (N-1)^2
    weight normalization would use the wrong N and distort the metric.

    Inputs are cast to int; callers must pass already-rounded integer classes,
    not continuous scores.
    """
    yt = np.asarray(y_true).round().astype(int)
    yp = np.asarray(y_pred).round().astype(int)
    return float(cohen_kappa_score(yt, yp, weights="quadratic", labels=LABELS))


def apply_thresholds(pred, thresholds) -> np.ndarray:
    """Map continuous predictions to integer classes 1..6 via cut points.

    ``thresholds`` holds the 5 boundaries between adjacent scores. A value below
    the first threshold -> 1; above the last -> 6. The array is sorted defensively
    so a mid-optimization non-monotone guess still yields a sane mapping.

    Returns an int array clipped to [1, 6].
    """
    x = np.asarray(pred, dtype=np.float64)
    cuts = np.sort(np.asarray(thresholds, dtype=np.float64))
    labels = np.searchsorted(cuts, x, side="right") + SCORE_MIN
    return np.clip(labels, SCORE_MIN, SCORE_MAX).astype(int)


def _enforce_strictly_increasing(raw: np.ndarray) -> np.ndarray:
    """Project an arbitrary vector onto a strictly increasing one.

    Cumulative-max then add tiny increasing epsilons: guarantees
    thresholds[i+1] > thresholds[i] (requirement 6) without materially moving
    well-separated cut points.
    """
    inc = np.maximum.accumulate(raw)
    inc = inc + np.arange(len(inc)) * _STRICT_EPS
    return inc


def optimize_thresholds(
    oof_pred,
    y_true,
    initial=INITIAL_THRESHOLDS,
    maxiter: int = 2000,
) -> ThresholdResult:
    """Find cut points that maximize OOF QWK.

    Parameters
    ----------
    oof_pred : array-like
        Continuous out-of-fold predictions. NEVER pass test predictions here.
    y_true : array-like
        Ground-truth integer scores aligned with ``oof_pred``.
    initial : array-like, default [1.5, 2.5, 3.5, 4.5, 5.5]
        Starting thresholds (midpoints between adjacent scores).
    maxiter : int
        Nelder-Mead iteration cap.

    Returns
    -------
    ThresholdResult
        best strictly-increasing ``thresholds``, best ``qwk``, the
        ``initial_qwk`` baseline, iteration count and convergence flag.
    """
    x = np.asarray(oof_pred, dtype=np.float64)
    y = np.asarray(y_true).round().astype(int)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"length mismatch: oof_pred={x.shape[0]} y_true={y.shape[0]}")

    init = np.asarray(initial, dtype=np.float64)

    def neg_qwk(thr: np.ndarray) -> float:
        preds = apply_thresholds(x, _enforce_strictly_increasing(thr))
        return -quadratic_weighted_kappa(y, preds)

    initial_qwk = quadratic_weighted_kappa(
        y, apply_thresholds(x, _enforce_strictly_increasing(init))
    )

    res = minimize(
        neg_qwk,
        init,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-5},
    )

    best = _enforce_strictly_increasing(res.x)
    best_qwk = quadratic_weighted_kappa(y, apply_thresholds(x, best))

    # never return something worse than the initial guess
    if best_qwk < initial_qwk:
        best = _enforce_strictly_increasing(init)
        best_qwk = initial_qwk

    return ThresholdResult(
        thresholds=best,
        qwk=best_qwk,
        initial_qwk=initial_qwk,
        n_iter=int(res.nit),
        success=bool(res.success),
    )
