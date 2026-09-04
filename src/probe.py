"""Model interrogation: coverage (OOD refusal gate), linear probes, attribution.

Probes the finalized pipeline (v2 corpus + preprocessing) by default.
`python probe.py all` runs the full device probe and writes probe_readings.json.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mac.harness as harness
import mac.models as models
from mac.generate import generate
from mac.generate_v2 import generate as generate_v2
from mac.preprocess import preprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mac"
K = 10


def load(corpus="v2"):
    if corpus == "v2":
        if not (DATA / "mac_synthetic_v2.npz").exists():
            generate_v2(str(DATA))
        z = np.load(DATA / "mac_synthetic_v2.npz", allow_pickle=True)
        meta = json.loads((DATA / "metadata_v2.json").read_text())
        return z, meta
    if not (DATA / "mac_synthetic_v1.npz").exists():
        generate(str(DATA))
    return harness.load(str(DATA))


def prepared(z, use_preprocess=True):
    if not use_preprocess:
        return z["spectra"], None
    Xp, n_spikes, sat = preprocess(z["spectra"])
    return Xp, {
        "spikes_removed": int(n_spikes.sum()),
        "saturated_flagged": int(sat.sum()),
    }


def feature_stack(grid, X, meta, idx, wl):
    return models.features(grid, X, meta, idx, wl)


def peak_features(grid, spectra):
    feats = []
    for s in spectra:
        base = np.percentile(s, 5)
        y = s - base
        ig = int(np.argmax(y))
        g_amp = float(y[ig])
        g_pos = float(grid[ig])
        half = g_amp / 2.0
        above = np.where(y >= half)[0]
        g_fwhm = float(grid[above[-1]] - grid[above[0]]) if len(above) else 1.0
        m = (grid > 1200) & (grid < 1500)
        d_amp = float(y[m].max()) if m.any() else 0.0
        feats.append([g_amp, g_pos, g_fwhm, d_amp, d_amp / max(g_amp, 1e-9)])
    return np.array(feats)


def _predict_la(grid, X, z, meta, idx, wl):
    gb = models.GBTModel()
    tr = harness.split_idx(z, "train")
    gb.fit(grid, X, meta, tr, wl, z["la"], z["sp3"], z["stage"])
    return gb.predict(grid, X, meta, idx, wl)[1]


def cmd_coverage(args):
    z, meta = load(args.corpus)
    Xp, prep = prepared(z)
    grid = z["grid"]
    wl, la, corr = z["wavelengths"], z["la"], z["corruption"]
    tr = harness.split_idx(z, "train")
    te = harness.split_idx(z, "test")
    F_tr = feature_stack(grid, Xp, meta, tr, wl)
    F_te = feature_stack(grid, Xp, meta, te, wl)
    sc = StandardScaler().fit(F_tr)
    nn = NearestNeighbors(n_neighbors=K).fit(sc.transform(F_tr))
    d_tr, _ = nn.kneighbors(sc.transform(F_tr))
    d_te, _ = nn.kneighbors(sc.transform(F_te))
    thr = np.percentile(d_tr[:, -1], 99)
    ood = d_te[:, -1] > thr
    rel = np.abs(_predict_la(grid, Xp, z, meta, te, wl) - la[te]) / np.maximum(
        la[te], 1e-6
    )
    clean = corr[te] == "none"
    rep = {
        "corpus": args.corpus,
        "ood_fraction": round(float(ood.mean()), 4),
        "rel_err_in_dist": round(float(rel[~ood].mean()), 4) if (~ood).any() else None,
        "rel_err_ood": round(float(rel[ood].mean()), 4) if ood.any() else None,
        "corrupted_flagged_ood": round(float(ood[~clean].mean()), 4)
        if (~clean).any()
        else None,
        "clean_flagged_ood": round(float(ood[clean].mean()), 4),
        "threshold_dist": round(float(thr), 4),
    }
    if prep:
        rep["preprocess"] = prep
    rep["verdict"] = (
        "OOD carries higher error; use as a refusal gate."
        if rep["rel_err_ood"] is not None
        and rep["rel_err_in_dist"] is not None
        and rep["rel_err_ood"] > rep["rel_err_in_dist"]
        else "no separation"
    )
    print(json.dumps(rep, indent=2))
    return rep


def _linear_probe(X, y, kind, rng):
    scores, shuf = [], []
    for _ in range(5):
        m = Ridge(alpha=1.0) if kind == "reg" else LogisticRegression(max_iter=2000)
        m.fit(X, y)
        pred = m.predict(X)
        scores.append(
            r2_score(y, pred) if kind == "reg" else f1_score(y, pred, average="macro")
        )
        ys = rng.permutation(y)
        m2 = Ridge(alpha=1.0) if kind == "reg" else LogisticRegression(max_iter=2000)
        m2.fit(X, ys)
        p2 = m2.predict(X)
        shuf.append(
            r2_score(ys, p2) if kind == "reg" else f1_score(ys, p2, average="macro")
        )
    return float(np.mean(scores)), float(np.mean(shuf))


def cmd_linear(args):
    z, meta = load(args.corpus)
    Xp, _ = prepared(z)
    rng = np.random.default_rng(3407)
    wl = z["wavelengths"]
    tr = harness.split_idx(z, "train")
    grid = z["grid"]
    sp3, stage = z["sp3"][tr], z["stage"][tr]
    reps = {
        "peak_only": peak_features(grid, Xp[tr]),
        "binned": models.RidgeModel()._bin(grid, Xp[tr]),
        "process": np.array([[meta[i][k] for k in models.META_KEYS] for i in tr]),
    }
    reps["full"] = np.hstack([reps["binned"], reps["process"], reps["peak_only"]])
    table = {}
    for name, R in reps.items():
        S = StandardScaler().fit(R)
        Rs = S.transform(R)
        r_sp3, sh_sp3 = _linear_probe(Rs, sp3, "reg", rng)
        r_st, sh_st = _linear_probe(Rs, stage, "cls", rng)
        table[name] = {
            "sp3_r2": round(r_sp3, 3),
            "stage_macro_f1": round(r_st, 3),
            "shuffle_sp3": round(sh_sp3, 3),
            "shuffle_stage": round(sh_st, 3),
        }
        print(
            "%-12s sp3 R2=%.3f stage F1=%.3f | shuf %.3f/%.3f"
            % (name, r_sp3, r_st, sh_sp3, sh_st)
        )
    return table


def cmd_attribute(args):
    z, meta = load(args.corpus)
    Xp, _ = prepared(z)
    grid = z["grid"]
    wl, la, stage = z["wavelengths"], z["la"], z["stage"]
    tr = harness.split_idx(z, "train")
    te = harness.split_idx(z, "test")
    i = te[args.index]
    F_tr = feature_stack(grid, Xp, meta, tr, wl)
    F_q = feature_stack(grid, Xp, meta, np.array([i]), wl)
    sc = StandardScaler().fit(F_tr)
    nn = NearestNeighbors(n_neighbors=8).fit(sc.transform(F_tr))
    _, idxs = nn.kneighbors(sc.transform(F_q))
    nb = tr[idxs[0]]
    gb = models.GBTModel()
    gb.fit(grid, Xp, meta, tr, wl, la, z["sp3"], stage)
    pred_all = gb.predict(grid, Xp, meta, np.array([i]), wl)[1][0]
    keep = np.setdiff1d(tr, nb)
    gb2 = models.GBTModel()
    gb2.fit(grid, Xp, meta, keep, wl, la, z["sp3"], stage)
    pred_wo = gb2.predict(grid, Xp, meta, np.array([i]), wl)[1][0]
    rep = {
        "query": "test-%05d" % i,
        "true_la": round(float(la[i]), 3),
        "stage": int(stage[i]),
        "wavelength_nm": float(wl[i]),
        "corruption": str(z["corruption"][i]),
        "pred_with_all": round(float(pred_all), 3),
        "pred_without_top8": round(float(pred_wo), 3),
        "attribution_shift": round(float(abs(pred_all - pred_wo)), 3),
        "supporting_rows": [
            {"id": "train-%05d" % int(n), "la": round(float(la[n]), 3)} for n in nb
        ],
    }
    print(json.dumps(rep, indent=2))
    return rep


def cmd_all(args):
    readings = {
        "corpus": args.corpus,
        "claim_status": "MODELLED - synthetic forward model",
        "coverage": cmd_coverage(args),
        "linear": cmd_linear(args),
        "attribution": cmd_attribute(args),
    }
    out = ROOT / "probe_readings.json"
    out.write_text(json.dumps(readings, indent=2))
    print("wrote", out)
    return readings


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("coverage", "linear", "all"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", default="v2", choices=["v1", "v2"])
        p.add_argument("--index", type=int, default=12)
    a = sub.add_parser("attribute")
    a.add_argument("--corpus", default="v2", choices=["v1", "v2"])
    a.add_argument("--index", type=int, default=12)
    args = ap.parse_args()
    {
        "coverage": cmd_coverage,
        "linear": cmd_linear,
        "attribute": cmd_attribute,
        "all": cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
