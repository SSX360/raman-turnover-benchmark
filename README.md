# raman-turnover-benchmark

[![verify](https://github.com/SSX360/raman-turnover-benchmark/actions/workflows/verify.yml/badge.svg)](https://github.com/SSX360/raman-turnover-benchmark/actions/workflows/verify.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22729662.svg)](https://doi.org/10.5281/zenodo.22729662)

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

The corpus itself (`data/mac/mac_synthetic_v2.npz`, `metadata_v2.json`), the per-row predictions of the release run, the deposited ledger and the pinned environment file are in the Zenodo data record cited in the paper's Data availability statement ([10.5281/zenodo.22729728](https://doi.org/10.5281/zenodo.22729728), CC BY 4.0; the same files, with the same digests, are published at https://ryanjamesyork.com/raman-turnover). The corpus can also be regenerated here in a few seconds.

## Reproduce

```
pip install -r requirements.txt          # pinned versions of the environment that produced the records (requirements.in lists the packages)
pip install -r requirements-ci.txt       # or: the CPU-only subset (no PyTorch) that the tests and verify.py need
python src/mac/generate_v2.py --out data/mac        # regenerates the corpus and metadata_v2.json
python src/eval_baseline_v2.py                      # Table 2, physics baseline row (deterministic, seconds)
python src/train_eval_v2.py --no-cnn                # Table 2, untuned tree model (seconds); drop --no-cnn for the network (GPU)
python src/train_eval_v3.py                         # Table 2, tuned tree model, network and ensemble (GPU, ~minutes)
python src/probe.py all                             # Fig. 3, probe_readings.json
python src/verify.py                                # release gates
python src/provenance.py verify                     # ledger (place the deposited ledger.jsonl in ledger/)
python -m pytest tests -q                           # composition, determinism, Table 2 baseline row, tree-model reproducibility, verify.py
```

`verify.py` regenerates the corpus and compares its digest, retrains the default tree model from the seed and compares T1, T2 and T3 with the ledgered record to 1e-6, checks cross-model agreement ≥ 0.95, checks that revision 3 improves on revision 2.1 and that the refusal gate lowers the corrupted-row error, and checks the probe readings. On the machine that produced the records every check passes; on an independent machine with a different NumPy build the regenerated `metadata_v2.json` differs in the last floating-point digit of transcendental evaluations in about 3 % of rows, with identical splits, labels, corruption assignments and counts, and the reproducibility check matches the ledgered T1, T2 and T3 to every printed digit.

### Independent re-derivations

| Date | Platform | Environment | Result |
|---|---|---|---|
| 2026-09-05 | Apple M-series, macOS | Python 3.12, `requirements.txt` (numpy 1.26.4, scikit-learn 1.5.1) | `VERIFY: PASS`; baseline row and T1/T2/T3 to every printed digit ([record](https://ryanjamesyork.com/raman-turnover)) |
| 2026-09-12 | Linux x86-64 (cloud sandbox) | Python 3.11, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0 | `VERIFY: PASS` in 18 s; baseline row T1 0.2106 / T2 9.0076 / T4 2.378 / T5 89.393; T1/T2/T3 retrain delta 0.00e+00 |
| 2026-09-12 | GitHub Actions `ubuntu-latest`, Python 3.12.14 | pinned: numpy 1.26.4, scipy 1.13.1, scikit-learn 1.5.1 (`requirements-ci.txt`) | `VERIFY: PASS`; all five tests pass (run #1, job "pinned") |
| 2026-09-12 | GitHub Actions `ubuntu-latest`, Python 3.12.14 | latest: numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1 | Baseline row exact; corpus deterministic; **default tree model drifts**: T1 0.9877 (record 0.9918), T2 0.0331 (0.0327), T3 0.0149 (0.0147), so the 1e-6 reproducibility gate fails by design; every other gate passes |
| continuous | GitHub Actions `ubuntu-latest` | the two rows above, on every push and weekly; the pinned job is the gate, the latest job is informational | [![verify](https://github.com/SSX360/raman-turnover-benchmark/actions/workflows/verify.yml/badge.svg)](https://github.com/SSX360/raman-turnover-benchmark/actions/workflows/verify.yml) |

The learned models are reproducible to 1e-6 on scikit-learn 1.5.1 (the record) and 1.8.0, and move at the third decimal on scikit-learn 1.9.1; the physics baseline and the corpus composition reproduce on every environment tried. The record therefore pins its environment, and the continuous job that must pass is the pinned one.

The corpus file digest is build-specific: on the pinned macOS environment `mac_synthetic_v2.npz` has SHA-256 `604e9bf4…` (the published record); on the Linux runs above it is `13054a9f…` (sandbox) and `c500106283…` (GitHub runner), with identical composition, splits, labels, corruption assignments and every reported metric. Determinism is therefore asserted as byte-identity between two generations on the same machine, and the metrics are asserted at printed precision; the published digest identifies the deposited file, not the only correct output of the generator.

## Notes a referee should know

* `train_eval_v3.py` selects the tree model's hyperparameters from 20 random draws; in the run behind `eval_results_v3.json` each candidate was fitted on the training and validation rows together and scored on the validation rows, so the selection score is optimistic. The selected setting was refitted on the training rows alone before test scoring, so the reported test numbers are from a model that never saw the test split. The paper states this.
* The despiker in `preprocess.py` replaces every point whose residual against a seven-point moving median exceeds six robust standard deviations. It replaced 205,147 points across the corpus, most of them ordinary noise excursions in the brighter parts of clean traces; the count is of points replaced, not of spike events.
* The stage labels for task T1 are assigned from L_a at 43 Å and 6 Å; the forward model switches branch at 20 Å.
* `verify.py` compares the corpus digest before and after regeneration on the same machine. Across NumPy builds the digest differs while the metrics do not (see Independent re-derivations); a digest mismatch against the published value on a different build is expected and is not a failed gate.
* The seven release gates of the paper's Table 3 include two gates on the acquisition journal (journal validity, hypothesis determinism). Those are exercised on a simulated acquisition campaign that belongs to a separate programme on the same platform and is not part of this repository or the deposit; `verify.py` here runs the five gates that apply to the deposited artifacts.

## Cite

`CITATION.cff` carries the machine-readable form. In text:

> R. J. York, *Resolving the Raman crystallite size turnover in nanocrystalline graphite with a synthetic benchmark*, preprint v2.0, 2026-09-05. Code: github.com/SSX360/raman-turnover-benchmark, release v2.0, doi:10.5281/zenodo.22729662. Data record: https://ryanjamesyork.com/raman-turnover, doi:10.5281/zenodo.22729728

The code DOI identifies release v2.0; the concept DOI 10.5281/zenodo.22729661 resolves to the latest release. The data-record DOI identifies the v2.0 deposit; its concept DOI is 10.5281/zenodo.22729727. Both records were published on Zenodo on 2026-09-12. Later releases of this repository are archived on Zenodo automatically.

## Licence

Apache License 2.0 (see `LICENSE`). Copyright 2026 SSX360 Corp.
