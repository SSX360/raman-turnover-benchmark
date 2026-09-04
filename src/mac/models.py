import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


def bwf(x, x0, gamma, amp, q=4.0):
    t = (x - x0) / gamma
    return amp * (1.0 + t / q) ** 2 / (1.0 + t * t)


def lorentz(x, x0, gamma, amp):
    return amp * gamma * gamma / ((x - x0) ** 2 + gamma * gamma)


def peak_fit_la(grid, spec, wl_nm=514.0):
    base = np.percentile(spec, 5)
    s = spec - base
    ig = int(np.argmax(s))
    g0, g_amp = grid[ig], s[ig]
    search = np.abs(grid - 1350.0) < 120.0
    d_amp = float(s[search].max()) if search.any() else 0.0
    ratio = d_amp / max(g_amp, 1e-9)
    c1, c2 = 44.0, 0.0055
    la1 = c1 / max(ratio, 1e-9)
    la2 = np.sqrt(max(ratio, 0.0) / c2) if c2 else np.inf
    return ratio, la1, la2


class PhysicsBaseline:
    def predict_la(self, grid, spectra):
        out = []
        for spec in spectra:
            ratio, la1, _ = peak_fit_la(grid, spec)
            out.append(min(la1, 300.0))
        return np.array(out)

    def predict_stage(self, grid, spectra):
        las = self.predict_la(grid, spectra)
        return np.where(las >= 43.0, 1, np.where(las >= 6.0, 2, 3))


class RidgeModel:
    def __init__(self, n_bins=120):
        self.n_bins = n_bins
        self.ridge_la = Ridge(alpha=10.0)
        self.ridge_sp3 = Ridge(alpha=10.0)
        self.clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.1)

    def _bin(self, grid, spectra):
        edges = np.linspace(grid[0], grid[-1], self.n_bins + 1)
        idx = np.clip(np.digitize(grid, edges) - 1, 0, self.n_bins - 1)
        out = np.zeros((len(spectra), self.n_bins))
        for i, s in enumerate(spectra):
            np.add.at(out[i], idx, s)
        return out

    def fit(self, grid, X, la, sp3, stage):
        B = self._bin(grid, X)
        self.ridge_la.fit(B, np.log1p(la))
        self.ridge_sp3.fit(B, sp3)
        self.clf.fit(B, stage)
        return self

    def predict(self, grid, X):
        B = self._bin(grid, X)
        return (
            self.clf.predict(B),
            np.expm1(self.ridge_la.predict(B)),
            self.ridge_sp3.predict(B),
        )


META_KEYS = ["T_C", "ion_energy_eV", "ch4_frac", "pressure_mTorr", "bias_V"]


def features(grid, spectra, meta_rows, idx, wl):
    step = max(1, len(grid) // 200)
    S = spectra[idx][:, ::step]
    S = S / np.maximum(S.max(axis=1, keepdims=True), 1e-9)
    M = np.array([[meta_rows[i][k] for k in META_KEYS] for i in idx])
    M = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)
    W = np.zeros((len(idx), 3))
    for j, w in enumerate((325.0, 514.0, 633.0)):
        W[:, j] = (wl[idx] == w).astype(float)
    return np.hstack([S, M, W])


class GBTModel:
    def __init__(self, **overrides):
        p = dict(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=63,
            min_samples_leaf=12,
            l2_regularization=1.0,
        )
        p.update(overrides)
        self.clf = HistGradientBoostingClassifier(**p)
        self.reg_la = HistGradientBoostingRegressor(**p)
        self.reg_sp3 = HistGradientBoostingRegressor(**dict(p, max_iter=250))

    def fit(self, grid, X, meta, idx, wl, la, sp3, stage):
        F = features(grid, X, meta, idx, wl)
        self.clf.fit(F, stage[idx])
        self.reg_la.fit(F, np.log1p(la[idx]))
        self.reg_sp3.fit(F, sp3[idx])
        return self

    def predict(self, grid, X, meta, idx, wl):
        F = features(grid, X, meta, idx, wl)
        return (
            self.clf.predict(F),
            np.expm1(self.reg_la.predict(F)),
            self.reg_sp3.predict(F),
        )
