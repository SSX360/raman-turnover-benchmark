# Submission verification scope

The September 13, 2026 maintenance revision preserves the published v2.0 corpus generator, model definitions and evaluation records. It repairs verification and reproduction tooling and documents the scope of the evidence. Use the exact Git commit when citing this revision; the existing versioned Zenodo code DOI identifies v2.0, not this later commit.

## Reproduced locally

Windows x86-64, Python 3.11, NumPy 1.26.4, SciPy 1.13.1 and scikit-learn 1.5.1, installed in a fresh virtual environment from `requirements-ci.txt`:

| Check | Result |
| --- | --- |
| Test suite, including missing evidence and ledger tampering regressions | 20 passed |
| Corpus composition | 3,400 rows; 2,400 / 400 / 600 train / validation / test; 207 corrupted |
| Physics baseline, 600 test rows | T1 0.2106; T2 9.0076; degenerate T2 2.378; corrupted-row MAE 89.393 |
| Default tree model | T1 0.9918; T2 0.0327; T3 0.0147; exact match to reported reference precision |
| Determinism | Two local generations have identical corpus, metadata and manifest bytes |
| Fresh probe run | Coverage error 0.0969 OOD / 0.0299 in-distribution; no clean rows flagged; attribution shift 0.014 angstrom |
| Reference records | Fresh training and probe outputs use `outputs/`; committed evidence is preserved |

The [verify workflow](https://github.com/SSX360/raman-turnover-benchmark/actions/workflows/verify.yml) reruns the tests and release checks on Linux. The pinned job is required for reproducibility; the latest-library job is informational because the learned model changes across library versions. CI stores the test report and verifier log as artifacts, including when a check fails.

## What a passing verifier establishes

The verifier regenerates the corpus in temporary directories and retrains the default tree model. It checks the committed cross-model, optimization/refusal and probe readings against their declared thresholds. All five gates must succeed; missing evidence, empty probe controls and non-finite values cannot produce a pass. The comparison tolerance of 1e-6 applies to metrics already rounded to four decimals by the harness.

The signed-ledger CLI is tested using temporary signing keys, including a modified-artifact failure. Verifying the actual deposited ledger requires its original artifacts and public key. The deposited ledger was not downloaded or verified during this maintenance pass; Zenodo API requests timed out. The public key is preserved, and the CLI rejects a missing or empty ledger. The release's attached manuscript PDF and its declared SHA-256 were confirmed through GitHub release metadata.

## Scientific and submission boundaries

These are synthetic benchmark results. CNN retraining and physical-sample validation were not performed in this maintenance pass. The historical tuning procedure fits candidate models on training plus validation rows; its selection score is optimistic. Metadata normalization depends on the requested batch, and the linear probes use training-set scores. These limitations are detailed in the README and must accompany claims based on these results. Correcting those analytical choices requires new model evaluations and a separately versioned scientific record.

The two acquisition-journal gates described in the paper concern a separate programme and are outside this repository. A repository verification pass does not establish NIST endorsement, grant eligibility, application completeness or Grants.gov acceptance. The grant narrative should distinguish this existing synthetic evidence from proposed experimental validation and future inference improvements.
