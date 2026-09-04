import json
import pathlib

import numpy as np

META_KEYS = ["T_C", "ion_energy_eV", "ch4_frac", "pressure_mTorr", "bias_V"]


def load(datadir):
    d = pathlib.Path(datadir)
    z = np.load(d / "mac_synthetic_v1.npz", allow_pickle=True)
    meta = json.loads((d / "metadata.json").read_text())
    return z, meta


def split_idx(z, name):
    return np.where(z["split"] == name)[0]


def degenerate_mask(z, idx):
    wl = z["wavelengths"][idx]
    la = z["la"][idx]
    dg = np.where(la >= 20.0, 44.0 / la, 0.0055 * la * la)
    return np.where((dg > 0.35) & (dg < 2.05) & (wl == 514.0))[0]


def evaluate(pred_stage, pred_la, z, idx):
    stage = z["stage"][idx]
    la = z["la"][idx]
    f1s = []
    for s in (1, 2, 3):
        m = stage == s
        if m.sum() == 0:
            continue
        tp = ((pred_stage == s) & m).sum()
        fp = ((pred_stage == s) & ~m).sum()
        fn = ((pred_stage != s) & m).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    t1 = float(np.mean(f1s))
    rel = np.abs(pred_la - la) / np.maximum(la, 1e-6)
    t2 = float(np.mean(rel))
    deg = degenerate_mask(z, idx)
    deg_rel = rel[deg] if len(deg) else np.array([np.nan])
    return {
        "T1_stage_macro_f1": round(t1, 4),
        "T2_la_mean_relative_error": round(t2, 4),
        "T2_deg_mean_relative_error": round(float(np.mean(deg_rel)), 4),
        "n": int(len(idx)),
        "n_degenerate": int(len(deg)),
    }


def evaluate_sp3(pred_sp3, meta, idx):
    true = np.array([meta[i]["sp3_frac"] for i in idx])
    return {"T3_sp3_mae": round(float(np.mean(np.abs(pred_sp3 - true))), 4)}


def evaluate_robustness(pred_la, z, idx):
    corr = z["corruption"][idx]
    clean = corr == "none"
    la = z["la"][idx]
    out = {}
    if clean.sum():
        out["T5_la_mae_clean"] = round(
            float(np.mean(np.abs(pred_la[clean] - la[clean]))), 3
        )
    bad = ~clean
    if bad.sum():
        out["T5_la_mae_corrupted"] = round(
            float(np.mean(np.abs(pred_la[bad] - la[bad]))), 3
        )
        out["n_corrupted"] = int(bad.sum())
    return out
