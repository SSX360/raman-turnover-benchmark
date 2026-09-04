"""v3: robust preprocessing + tuned GBT + CNN (mixup + SWA) + val-fit ensemble.
Writes eval_results_v3.json.
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mac.harness as harness
import mac.models as models
from mac.preprocess import preprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mac"
OUT = ROOT / "eval_results_v3.json"

PARAM_GRID = {
    "max_iter": [400, 600, 800],
    "learning_rate": [0.03, 0.05, 0.08],
    "max_leaf_nodes": [63, 127, 255],
    "min_samples_leaf": [6, 10, 16],
    "l2_regularization": [0.3, 1.0, 3.0],
}


def objective(res):
    return (
        (1.0 - res["T1_stage_macro_f1"])
        + res["T2_la_mean_relative_error"]
        + 0.5 * res["T2_deg_mean_relative_error"]
    )


def load_v2():
    z = np.load(DATA / "mac_synthetic_v2.npz", allow_pickle=True)
    meta = json.loads((DATA / "metadata_v2.json").read_text())
    return z, meta


def tune_gbt(z, X, meta, wl, la, sp3, stage, tr, va, trials, seed=3407):
    rng = np.random.default_rng(seed)
    fit_idx = np.concatenate([tr, va])
    best_score = float("inf")
    best_params = None
    for t in range(trials):
        params = {
            k: float(rng.choice(v)) if isinstance(v[0], float) else int(rng.choice(v))
            for k, v in PARAM_GRID.items()
        }
        gb = models.GBTModel(**params).fit(
            z["grid"], X, meta, fit_idx, wl, la, sp3, stage
        )
        ps, pl, _ = gb.predict(z["grid"], X, meta, va, wl)
        res = harness.evaluate(ps, pl, z, va)
        score = objective(res)
        if score < best_score:
            best_score = score
            best_params = params
        print("  gbt trial %d score=%.4f (best %.4f)" % (t, score, best_score))
    if best_params is None:
        raise RuntimeError("no trials completed")
    return best_params, best_score


def gated_robustness(pl, sat, z, te):
    corr = z["corruption"][te] != "none"
    la = z["la"][te]
    out = {}
    clean = ~corr
    if clean.sum():
        out["T5_la_mae_clean"] = round(float(np.mean(np.abs(pl[clean] - la[clean]))), 3)
    if corr.sum():
        out["T5_la_mae_corrupted"] = round(
            float(np.mean(np.abs(pl[corr] - la[corr]))), 3
        )
        keep = corr & ~sat
        if keep.sum():
            out["T5_la_mae_corrupted_gated"] = round(
                float(np.mean(np.abs(pl[keep] - la[keep]))), 3
            )
        out["n_corrupted"] = int(corr.sum())
        out["n_saturated_flagged"] = int(sat.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--no-cnn", action="store_true")
    args = ap.parse_args()

    z, meta = load_v2()
    grid, X = z["grid"], z["spectra"]
    la, sp3, stage, wl = z["la"], z["sp3"], z["stage"], z["wavelengths"]
    tr = harness.split_idx(z, "train")
    va = harness.split_idx(z, "val")
    te = harness.split_idx(z, "test")

    Xp, n_spikes, sat = preprocess(X)
    print(
        "preprocess: total spikes removed %d, saturated spectra flagged %d"
        % (int(n_spikes.sum()), int(sat.sum()))
    )
    sat_te = sat[te]

    results: dict = {
        "preprocess": {
            "spikes_removed": int(n_spikes.sum()),
            "saturated_flagged": int(sat.sum()),
            "saturated_flagged_test": int(sat_te.sum()),
        }
    }

    print("tuning GBT (%d trials)..." % args.trials)
    best_params, best_score = tune_gbt(
        z, Xp, meta, wl, la, sp3, stage, tr, va, args.trials
    )
    gb = models.GBTModel(**best_params).fit(grid, Xp, meta, tr, wl, la, sp3, stage)
    ps, pl, psp3 = gb.predict(grid, Xp, meta, te, wl)
    r: dict = dict(harness.evaluate(ps, pl, z, te))
    r.update(harness.evaluate_sp3(psp3, meta, te))
    r.update(gated_robustness(pl, sat_te, z, te))
    r["tuned_params"] = best_params
    results["gbt_v3"] = r
    print("gbt_v3:", json.dumps(r))

    if not args.no_cnn and importlib.util.find_spec("torch"):
        from mac import nn

        device = "cuda"
        net, val_loss = nn.train_net(
            grid, Xp, wl, la, sp3, stage, tr, va, meta, device=device, epochs=160
        )
        ns, nl, nsp3 = nn.predict_net(net, grid, Xp, wl, te, meta, device=device)
        r = dict(harness.evaluate(ns, nl, z, te))
        r.update(harness.evaluate_sp3(nsp3, meta, te))
        r.update(gated_robustness(nl, sat_te, z, te))
        r["val_loss"] = round(val_loss, 4)
        results["cnn_v3"] = r
        print("cnn_v3:", json.dumps(r))

        w_grid = np.linspace(0.0, 1.0, 21)
        pv_s, pv_l, _ = gb.predict(grid, Xp, meta, va, wl)
        nv_s, nv_l, _ = nn.predict_net(net, grid, Xp, wl, va, meta, device=device)
        best_w, best_err = 0.0, float("inf")
        for w in w_grid:
            mix = np.exp(
                w * np.log(np.clip(pv_l, 1e-6, None))
                + (1 - w) * np.log(np.clip(nv_l, 1e-6, None))
            )
            err = float(np.mean(np.abs(mix - la[va]) / np.maximum(la[va], 1e-6)))
            if err < best_err:
                best_w, best_err = float(w), err
        ens_la = np.exp(
            best_w * np.log(np.clip(pl, 1e-6, None))
            + (1 - best_w) * np.log(np.clip(nl, 1e-6, None))
        )
        ens_stage = np.where(ps == ns, ps, np.where(best_w >= 0.5, ps, ns))
        ens_sp3 = 0.5 * (psp3 + nsp3)
        r = dict(harness.evaluate(ens_stage, ens_la, z, te))
        r.update(harness.evaluate_sp3(ens_sp3, meta, te))
        r.update(gated_robustness(ens_la, sat_te, z, te))
        r["ensemble_w_gbt"] = best_w
        r["ensemble_val_rel_err"] = round(best_err, 4)
        results["ensemble_v3"] = r
        print("ensemble_v3:", json.dumps(r))
        results["cross_model"] = {
            "stage_agreement": round(float((ns == ps).mean()), 4),
            "la_pearson": round(float(np.corrcoef(nl, pl)[0, 1]), 4),
        }

    payload = {
        "corpus": "mac_synthetic_v2",
        "pipeline": "v3",
        "results": results,
        "t1_macro_f1_gbt": results["gbt_v3"]["T1_stage_macro_f1"],
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
