"""Continuous re-derivation of the paper's CPU-reproducible numbers.

These tests re-run, on whatever machine executes them, the parts of the release record that need
no GPU: the corpus generator, the physics baseline (Table 2, first row) and the release gates in
src/verify.py. They are the same commands a referee is asked to run in the README.

Tolerances: metrics are compared at the precision the harness prints (four decimals for T1/T2/T4,
three for T5). The corpus file digest is NOT asserted against the published value: the digest is
specific to the NumPy build (compression and the last floating-point digit of transcendental
evaluations), whereas the metrics reproduce across builds. Determinism is asserted instead:
two generations on the same machine must be byte-identical.
"""

import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data" / "mac"
sys.path.insert(0, str(SRC))

from mac.generate_v2 import generate  # noqa: E402
import mac.harness as harness  # noqa: E402
import mac.models as models  # noqa: E402

# Table 2, first row (physics baseline on corpus v2), as printed in the record:
# T1 0.2106 / T2 9.0076 / T4 2.378 / T5 89.393
BASELINE = {
    "T1_stage_macro_f1": (0.2106, 1e-4),
    "T2_la_mean_relative_error": (9.0076, 1e-4),
    "T2_deg_mean_relative_error": (2.378, 1e-3),
    "T5_la_mae_corrupted": (89.393, 1e-2),
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="session")
def corpus():
    generate(str(DATA))
    return np.load(DATA / "mac_synthetic_v2.npz", allow_pickle=True)


def test_corpus_composition(corpus):
    stage = corpus["stage"]
    assert len(stage) == 3400
    assert {int(s): int((stage == s).sum()) for s in (1, 2, 3)} == {1: 1478, 2: 1379, 3: 543}
    assert int((corpus["corruption"] != "none").sum()) == 207
    split = corpus["split"]
    assert [int((split == k).sum()) for k in ("train", "val", "test")] == [2400, 400, 600]


def test_corpus_regenerates_byte_identically(tmp_path, corpus):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate(str(a))
    generate(str(b))
    assert _sha256(a / "mac_synthetic_v2.npz") == _sha256(b / "mac_synthetic_v2.npz")
    assert (a / "metadata_v2.json").read_bytes() == (b / "metadata_v2.json").read_bytes()
    assert (a / "manifest_v2.json").read_bytes() == (b / "manifest_v2.json").read_bytes()


def test_physics_baseline_is_table2_first_row(corpus):
    te = harness.split_idx(corpus, "test")
    base = models.PhysicsBaseline()
    pl = base.predict_la(corpus["grid"], corpus["spectra"][te])
    ps = base.predict_stage(corpus["grid"], corpus["spectra"][te])
    r = dict(harness.evaluate(ps, pl, corpus, te))
    r.update(harness.evaluate_robustness(pl, corpus, te))
    assert r["n"] == 600 and r["n_degenerate"] == 100 and r["n_corrupted"] == 45
    for key, (expected, tol) in BASELINE.items():
        assert abs(r[key] - expected) <= tol, (key, r[key], expected)


def test_default_tree_model_reproduces_release_record(corpus):
    ref = json.loads((ROOT / "eval_results_v21.json").read_text())["results"]["gbt_v2"]
    meta = json.loads((DATA / "metadata_v2.json").read_text())
    grid, X = corpus["grid"], corpus["spectra"]
    la, sp3, stage, wl = corpus["la"], corpus["sp3"], corpus["stage"], corpus["wavelengths"]
    tr = harness.split_idx(corpus, "train")
    te = harness.split_idx(corpus, "test")
    gb = models.GBTModel().fit(grid, X, meta, tr, wl, la, sp3, stage)
    ps, pl, psp3 = gb.predict(grid, X, meta, te, wl)
    r = harness.evaluate(ps, pl, corpus, te)
    r.update(harness.evaluate_sp3(psp3, meta, te))
    for key in ("T1_stage_macro_f1", "T2_la_mean_relative_error", "T3_sp3_mae"):
        assert abs(r[key] - ref[key]) <= 1e-6, (key, r[key], ref[key])


def test_verify_script_passes(corpus):
    proc = subprocess.run(
        [sys.executable, str(SRC / "verify.py")], capture_output=True, text=True, timeout=1800
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "VERIFY: PASS" in proc.stdout
