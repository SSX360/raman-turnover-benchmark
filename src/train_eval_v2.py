"""v2 train + eval: v2 corpus (wavelength-scaled C, stress G-shift, a-C:H PL),
GBT + 1D-CNN (GPU), five-task harness, cross-model comparison.
Writes eval_results_v2.json.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mac.harness as harness
import mac.models as models
from mac.generate_v2 import generate

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mac"
OUT = ROOT / "eval_results_v21.json"


def load_v2():
    if not (DATA / "mac_synthetic_v2.npz").exists():
        generate(str(DATA))
    z = np.load(DATA / "mac_synthetic_v2.npz", allow_pickle=True)
    meta = json.loads((DATA / "metadata_v2.json").read_text())
    return z, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-cnn", action="store_true")
    args = ap.parse_args()

    z, meta = load_v2()
    grid, X = z["grid"], z["spectra"]
    la, sp3, stage, wl = z["la"], z["sp3"], z["stage"], z["wavelengths"]
    tr = harness.split_idx(z, "train")
    va = harness.split_idx(z, "val")
    te = harness.split_idx(z, "test")
    results = {}

    gb = models.GBTModel().fit(grid, X, meta, tr, wl, la, sp3, stage)
    ps, pl, psp3 = gb.predict(grid, X, meta, te, wl)
    r = harness.evaluate(ps, pl, z, te)
    r.update(harness.evaluate_sp3(psp3, meta, te))
    r.update(harness.evaluate_robustness(pl, z, te))
    results["gbt_v2"] = r
    print("gbt_v2:", json.dumps(r))

    if not args.no_cnn and __import__("importlib").util.find_spec("torch"):
        from mac import nn

        device = "cuda"
        net, val_loss = nn.train_net(
            grid, X, wl, la, sp3, stage, tr, va, meta, device=device
        )
        ns, nl, nsp3 = nn.predict_net(net, grid, X, wl, te, meta, device=device)
        r = harness.evaluate(ns, nl, z, te)
        r.update(harness.evaluate_sp3(nsp3, meta, te))
        r.update(harness.evaluate_robustness(nl, z, te))
        r["val_loss"] = round(val_loss, 4)
        results["cnn_v2"] = r
        print("cnn_v2:", json.dumps(r))
        agree_stage = float((ns == ps).mean())
        la_corr = float(np.corrcoef(nl, pl)[0, 1])
        results["cross_model"] = {
            "stage_agreement": round(agree_stage, 4),
            "la_pearson": round(la_corr, 4),
            "la_mae_between_models": round(float(np.mean(np.abs(nl - pl))), 3),
        }
        print("cross_model:", json.dumps(results["cross_model"]))

        ens_la = np.sqrt(np.clip(pl, 1e-6, None) * np.clip(nl, 1e-6, None))
        ens_stage = np.where(
            np.abs(pl - nl) < 0.25 * np.maximum(pl, nl), ps, np.where(pl < nl, ps, ns)
        )
        r = harness.evaluate(ens_stage, ens_la, z, te)
        r.update(harness.evaluate_sp3(0.5 * (psp3 + nsp3), meta, te))
        r.update(harness.evaluate_robustness(ens_la, z, te))
        results["ensemble_v2"] = r
        print("ensemble_v2:", json.dumps(r))

    payload = {
        "corpus": "mac_synthetic_v2",
        "results": results,
        "t1_macro_f1_gbt": results["gbt_v2"]["T1_stage_macro_f1"],
    }
    pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
