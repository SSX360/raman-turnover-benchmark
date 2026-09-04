"""Verification suite: determinism, eval reproducibility, cross-model agreement.

Paper subset of the platform verifier: the checks that run on the artifacts deposited
with the paper. Checks that belong to other programmes on the same platform are not
included here.

verify passes only if:
  1. regenerating the v2 corpus is byte-identical (determinism)
  2. T1 macro-F1 reproduces within 1e-6 of eval_results_v2.json (reproducibility)
  3. GBT/CNN stage agreement >= 0.95 (cross-model, when CNN present)
"""

import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mac"
_cands = [ROOT / "eval_results_v21.json", ROOT / "eval_results_v2.json"]
EVAL = next((p for p in _cands if p.exists()), ROOT / "eval_results_v2.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_determinism():
    from mac.generate_v2 import generate

    before = sha256(DATA / "mac_synthetic_v2.npz")
    generate(str(DATA))
    after = sha256(DATA / "mac_synthetic_v2.npz")
    ok = before == after
    print(
        "[%s] determinism: sha256 %s -> %s"
        % ("PASS" if ok else "FAIL", before[:12], after[:12])
    )
    return ok


def check_reproducibility():
    import mac.harness as harness
    import mac.models as models

    ref = json.loads(EVAL.read_text())["results"]["gbt_v2"]
    z = np.load(DATA / "mac_synthetic_v2.npz", allow_pickle=True)
    meta = json.loads((DATA / "metadata_v2.json").read_text())
    grid, X = z["grid"], z["spectra"]
    la, sp3, stage, wl = z["la"], z["sp3"], z["stage"], z["wavelengths"]
    tr = harness.split_idx(z, "train")
    te = harness.split_idx(z, "test")
    gb = models.GBTModel().fit(grid, X, meta, tr, wl, la, sp3, stage)
    ps, pl, psp3 = gb.predict(grid, X, meta, te, wl)
    r = harness.evaluate(ps, pl, z, te)
    r.update(harness.evaluate_sp3(psp3, meta, te))
    ok = True
    for k in ("T1_stage_macro_f1", "T2_la_mean_relative_error", "T3_sp3_mae"):
        d = abs(r[k] - ref[k])
        if d > 1e-6:
            ok = False
        print("    %s: ref=%s run=%s delta=%.2e" % (k, ref[k], r[k], d))
    print("[%s] reproducibility" % ("PASS" if ok else "FAIL"))
    return ok


def check_cross_model():
    if not EVAL.exists():
        print("[SKIP] cross-model (no eval_results_v2.json)")
        return True
    res = json.loads(EVAL.read_text())["results"]
    if "cross_model" not in res:
        print("[SKIP] cross-model (CNN not trained)")
        return True
    agree = res["cross_model"]["stage_agreement"]
    ok = agree >= 0.95
    print(
        "[%s] cross-model stage agreement %.4f (>= 0.95)"
        % ("PASS" if ok else "FAIL", agree)
    )
    return ok


def check_v3_optimization():
    v3p = ROOT / "eval_results_v3.json"
    v21p = ROOT / "eval_results_v21.json"
    if not v3p.exists() or not v21p.exists():
        print("[SKIP] v3 optimization (missing eval files)")
        return True
    v3 = json.loads(v3p.read_text())["results"]
    v21 = json.loads(v21p.read_text())["results"]
    ok = True
    g = (
        v3["gbt_v3"]["T2_la_mean_relative_error"]
        <= v21["gbt_v2"]["T2_la_mean_relative_error"]
    )
    ok &= g
    print(
        "[%s] v3 tuned GBT improves T2: %.4f <= %.4f"
        % (
            "PASS" if g else "FAIL",
            v3["gbt_v3"]["T2_la_mean_relative_error"],
            v21["gbt_v2"]["T2_la_mean_relative_error"],
        )
    )
    if "ensemble_v3" in v3:
        e = (
            v3["ensemble_v3"]["T2_la_mean_relative_error"]
            <= v3["gbt_v3"]["T2_la_mean_relative_error"]
        )
        ok &= e
        print(
            "[%s] v3 ensemble improves over tuned GBT: %.4f <= %.4f"
            % (
                "PASS" if e else "FAIL",
                v3["ensemble_v3"]["T2_la_mean_relative_error"],
                v3["gbt_v3"]["T2_la_mean_relative_error"],
            )
        )
        gated = v3["ensemble_v3"].get("T5_la_mae_corrupted_gated")
        corr = v3["ensemble_v3"].get("T5_la_mae_corrupted")
        if gated is not None and corr is not None:
            r = gated <= corr
            ok &= r
            print(
                "[%s] refusal gate lowers corrupted-row error: %.3f <= %.3f"
                % ("PASS" if r else "FAIL", gated, corr)
            )
    return ok


def check_probe_readings():
    p = ROOT / "probe_readings.json"
    if not p.exists():
        print("[SKIP] probe readings (run: python probe.py all)")
        return True
    r = json.loads(p.read_text())
    ok = True
    cov = r["coverage"]
    sep = cov["rel_err_ood"] > cov["rel_err_in_dist"]
    cf = cov["clean_flagged_ood"] == 0.0
    ok &= sep and cf
    print(
        "[%s] coverage probe: OOD err %.3f > in-dist %.3f, clean flagged %.3f"
        % (
            "PASS" if sep and cf else "FAIL",
            cov["rel_err_ood"],
            cov["rel_err_in_dist"],
            cov["clean_flagged_ood"],
        )
    )
    shuf_ok = all(
        abs(v["shuffle_sp3"]) < 0.1 and abs(v["shuffle_stage"]) < 0.35
        for v in r["linear"].values()
    )
    ok &= shuf_ok
    print(
        "[%s] linear probes: all shuffled controls near zero"
        % ("PASS" if shuf_ok else "FAIL")
    )
    shift_ok = r["attribution"]["attribution_shift"] < 1.0
    ok &= shift_ok
    print(
        "[%s] attribution: shift %.3f A (< 1.0, no single cluster dominates)"
        % ("PASS" if shift_ok else "FAIL", r["attribution"]["attribution_shift"])
    )
    return ok


def main():
    ok = True
    ok &= check_determinism()
    ok &= check_reproducibility()
    ok &= check_cross_model()
    ok &= check_v3_optimization()
    ok &= check_probe_readings()
    print()
    print("VERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
