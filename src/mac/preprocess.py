"""Robust spectrum preprocessing: cosmic-ray despiking + saturation flagging.

The v1 report ranked detector saturation as the worst corruption (unrecoverable)
and cosmic rays as nearly harmless given standard despiking. Implemented here:
median-filter residual despiking, and a flat-top saturation detector whose flag
is a refusal signal, never silently "corrected".
"""

import numpy as np


def moving_median(x, k=7):
    p = np.pad(x, k // 2, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(p[i : i + k])
    return out


def despike(spec, k=7, thresh=6.0):
    med = moving_median(spec, k)
    resid = spec - med
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    z = resid / (1.4826 * mad)
    out = spec.copy()
    hits = z > thresh
    out[hits] = med[hits]
    return out, int(hits.sum())


def saturation_flag(spec, flat_run=5, tol=1e-6):
    top = np.quantile(spec, 0.985)
    flat = np.abs(spec - top) <= tol + 1e-4 * top
    run = 0
    best = 0
    for v in flat:
        run = run + 1 if v else 0
        best = max(best, run)
    return best >= flat_run


def preprocess(spectra, k=7, thresh=6.0):
    out = np.empty_like(spectra)
    n_spikes = np.zeros(len(spectra), dtype=np.int32)
    sat = np.zeros(len(spectra), dtype=bool)
    for i, s in enumerate(spectra):
        out[i], n_spikes[i] = despike(s, k, thresh)
        sat[i] = saturation_flag(s)
    return out, n_spikes, sat
