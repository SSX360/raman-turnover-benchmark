"""Physics baseline on corpus v2: Tuinstra-Koenig inversion with stage 1 assumed.

Writes eval_results_baseline_v2.json (the baseline row of Table 2). Deterministic; no training.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mac.harness as harness
import mac.models as models

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mac"
OUT = ROOT / "eval_results_baseline_v2.json"


def main():
    z = np.load(DATA / "mac_synthetic_v2.npz", allow_pickle=True)
    te = harness.split_idx(z, "test")
    base = models.PhysicsBaseline()
    pl = base.predict_la(z["grid"], z["spectra"][te])
    ps = base.predict_stage(z["grid"], z["spectra"][te])
    r = dict(harness.evaluate(ps, pl, z, te))
    r.update(harness.evaluate_robustness(pl, z, te))
    payload = {"corpus": "mac_synthetic_v2", "pipeline": "physics_baseline", "results": {"physics_baseline": r}}
    OUT.write_text(json.dumps(payload, indent=2))
    print("physics_baseline:", json.dumps(r))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
