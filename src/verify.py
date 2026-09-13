"""Verification suite: determinism, eval reproducibility, cross-model agreement.

Paper subset of the platform verifier: the checks that run on the artifacts deposited
with the paper. Checks that belong to other programmes on the same platform are not
included here.

All five gates are required. Determinism and tree-model reproducibility are
recomputed; cross-model, optimization and probe gates inspect deposited records.
Missing, malformed or non-finite evidence fails verification.
"""

import hashlib
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mac"
EVAL = ROOT / "eval_results_v21.json"


def read_record(path):
    record = json.loads(path.read_text(encoding="utf-8"))

    def validate(value):
        if isinstance(value, dict):
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)
        elif isinstance(value, float) and not np.isfinite(value):
            raise ValueError("non-finite value in %s" % path.name)

    validate(record)
    return record


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_determinism():
    from mac.generate_v2 import generate

    # Never overwrite a downloaded deposit or compare it with another build.
    with tempfile.TemporaryDirectory() as tmp:
        a, b = pathlib.Path(tmp) / "a", pathlib.Path(tmp) / "b"
        generate(str(a))
        generate(str(b))
        ok = True
        for name in ("mac_synthetic_v2.npz", "metadata_v2.json", "manifest_v2.json"):
            same = sha256(a / name) == sha256(b / name)
            ok &= same
            print("[%s] determinism: %s" % ("PASS" if same else "FAIL", name))
    return ok


def check_reproducibility():
    import mac.harness as harness
    import mac.models as models

    ref = read_record(EVAL)["results"]["gbt_v2"]
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
        if not np.isfinite(d) or d > 1e-6:
            ok = False
        print("    %s: ref=%s run=%s delta=%.2e" % (k, ref[k], r[k], d))
    print("[%s] reproducibility" % ("PASS" if ok else "FAIL"))
    return ok


def check_cross_model():
    if not EVAL.exists():
        print("[FAIL] cross-model (missing eval_results_v21.json)")
        return False
    res = read_record(EVAL)["results"]
    if "cross_model" not in res:
        print("[FAIL] cross-model (missing CNN comparison)")
        return False
    agree = res["cross_model"]["stage_agreement"]
    ok = 0.95 <= agree <= 1.0
    print(
        "[%s] cross-model stage agreement %.4f (>= 0.95)"
        % ("PASS" if ok else "FAIL", agree)
    )
    return ok


def check_v3_optimization():
    v3p = ROOT / "eval_results_v3.json"
    v21p = ROOT / "eval_results_v21.json"
    if not v3p.exists() or not v21p.exists():
        print("[FAIL] v3 optimization (missing eval files)")
        return False
    v3 = read_record(v3p)["results"]
    v21 = read_record(v21p)["results"]
    if "ensemble_v3" not in v3:
        print("[FAIL] v3 optimization (missing ensemble)")
        return False
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
        if gated is None or corr is None:
            print("[FAIL] refusal gate (missing corrupted-row metrics)")
            return False
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
        print("[FAIL] probe readings (missing deposited probe_readings.json)")
        return False
    r = read_record(p)
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
    expected = {"peak_only", "binned", "process", "full"}
    shuf_ok = expected <= r["linear"].keys() and all(
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
    for check in (check_determinism, check_reproducibility, check_cross_model,
                  check_v3_optimization, check_probe_readings):
        try:
            ok &= check()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print("[FAIL] %s: %s" % (check.__name__, exc))
            ok = False
    print()
    print("VERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
