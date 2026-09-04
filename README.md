# raman-turnover-benchmark

Code and evaluation records for

> R. J. York, *Resolving the Raman crystallite size turnover in nanocrystalline graphite with a synthetic benchmark* (manuscript, 2026; the citation will be updated when it is published).

The disorder ratio I(D)/I(G) of carbon Raman spectra is not invertible: it rises as 1/L_a to a turnover at 20 Å and falls as L_a² beyond it, so one ratio maps to two crystallite sizes. This repository holds the generator of a deterministic synthetic corpus built on that forward model (3,400 spectra at 325, 514 and 633 nm with planted acquisition failures), the five benchmark tasks, the inversion pipelines scored on it, the probes, and the verification script that re-runs the release gates. Every number in the paper is either the output of these scripts or is ledgered in the deposited record.

All results are **modelled**: they are outputs of a declared forward model and make no claim about physical films.

## Layout

```
src/mac/generate_v2.py   corpus generator (mac_synthetic_v2: 3,400 rows, seed 3407, PCG64)
src/mac/generate.py      the earlier v1 generator (kept because probe.py imports it; not used by the paper)
src/mac/harness.py       splits, the five task metrics, the degenerate subset
src/mac/models.py        physics baseline, gradient-boosted tree model (scikit-learn HistGradientBoosting), features
src/mac/nn.py            1-D convolutional network (PyTorch)
src/mac/preprocess.py    moving-median despiker and flat-top saturation flag
src/eval_baseline_v2.py  physics baseline on the test split -> eval_results_baseline_v2.json (Table 2, first row)
src/train_eval_v2.py     revision 2.1: default tree model, network, cross-model agreement -> eval_results_v21.json
src/train_eval_v3.py     revision 3: preprocessing, tuned tree model, network, ensemble -> eval_results_v3.json
src/probe.py             coverage probe, linear probes, single-query attribution -> probe_readings.json
src/verify.py            release gates on the deposited artifacts (determinism, reproducibility, cross-model, optimisation, probes)
src/provenance.py        append-only hash-chained Ed25519 ledger: keygen / add / verify / show
ledger/operator.pub      public key that verifies the deposited ledger
data/mac/manifest_v2.json           corpus manifest (declared physics and parameters)
data/mac/mac_synthetic_v2.npz.dvc   DVC pointer (md5, size) for the corpus file
eval_results_v2.json, eval_results_v21.json, eval_results_v3.json, probe_readings.json   the records behind Tables 2 and 3
```

The corpus itself (`data/mac/mac_synthetic_v2.npz`, `metadata_v2.json`), the per-row predictions of the release run, the deposited ledger and the pinned environment file are in the Zenodo data record cited in the paper's Data availability statement. The corpus can also be regenerated here in a few seconds.

## Reproduce

```
pip install -r requirements.txt          # pinned versions of the environment that produced the records (requirements.in lists the packages)
python src/mac/generate_v2.py --out data/mac        # regenerates the corpus and metadata_v2.json
python src/eval_baseline_v2.py                      # Table 2, physics baseline row (deterministic, seconds)
python src/train_eval_v2.py --no-cnn                # Table 2, untuned tree model (seconds); drop --no-cnn for the network (GPU)
python src/train_eval_v3.py                         # Table 2, tuned tree model, network and ensemble (GPU, ~minutes)
python src/probe.py all                             # Fig. 3, probe_readings.json
python src/verify.py                                # release gates
python src/provenance.py verify                     # ledger (place the deposited ledger.jsonl in ledger/)
```

`verify.py` regenerates the corpus and compares its digest, retrains the default tree model from the seed and compares T1, T2 and T3 with the ledgered record to 1e-6, checks cross-model agreement ≥ 0.95, checks that revision 3 improves on revision 2.1 and that the refusal gate lowers the corrupted-row error, and checks the probe readings. On the machine that produced the records every check passes; on an independent machine with a different NumPy build the regenerated `metadata_v2.json` differs in the last floating-point digit of transcendental evaluations in about 3 % of rows, with identical splits, labels, corruption assignments and counts, and the reproducibility check matches the ledgered T1, T2 and T3 to every printed digit.

## Notes a referee should know

* `train_eval_v3.py` selects the tree model's hyperparameters from 20 random draws; in the run behind `eval_results_v3.json` each candidate was fitted on the training and validation rows together and scored on the validation rows, so the selection score is optimistic. The selected setting was refitted on the training rows alone before test scoring, so the reported test numbers are from a model that never saw the test split. The paper states this.
* The despiker in `preprocess.py` replaces every point whose residual against a seven-point moving median exceeds six robust standard deviations. It replaced 205,147 points across the corpus, most of them ordinary noise excursions in the brighter parts of clean traces; the count is of points replaced, not of spike events.
* The stage labels for task T1 are assigned from L_a at 43 Å and 6 Å; the forward model switches branch at 20 Å.
* The seven release gates of the paper's Table 3 include two gates on the acquisition journal (journal validity, hypothesis determinism). Those are exercised on a simulated acquisition campaign that belongs to a separate programme on the same platform and is not part of this repository or the deposit; `verify.py` here runs the five gates that apply to the deposited artifacts.

## Licence

Apache License 2.0 (see `LICENSE`). Copyright 2026 SSX360 Corp.
