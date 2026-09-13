"""Regression tests for verification that must fail when evidence is absent."""

import json
import pathlib
import sys
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import provenance
import verify


@pytest.mark.parametrize("check", ["check_cross_model", "check_v3_optimization", "check_probe_readings"])
def test_missing_release_evidence_fails(tmp_path, monkeypatch, check):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    monkeypatch.setattr(verify, "EVAL", tmp_path / "eval_results_v21.json")
    assert getattr(verify, check)() is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_record_is_rejected(tmp_path, value):
    record = tmp_path / "record.json"
    record.write_text(json.dumps({"results": {"metric": value}}))
    with pytest.raises(ValueError, match="non-finite"):
        verify.read_record(record)


def test_partial_cpu_record_cannot_pass_cross_model(tmp_path, monkeypatch):
    record = tmp_path / "eval_results_v21.json"
    record.write_text(json.dumps({"results": {"gbt_v2": {}}}))
    monkeypatch.setattr(verify, "EVAL", record)
    assert verify.check_cross_model() is False


@pytest.mark.parametrize("missing", ["ensemble_v3", "T5_la_mae_corrupted_gated"])
def test_missing_optimization_evidence_fails(tmp_path, monkeypatch, missing):
    v3 = json.loads((ROOT / "eval_results_v3.json").read_text())
    if missing == "ensemble_v3":
        del v3["results"][missing]
    else:
        del v3["results"]["ensemble_v3"][missing]
    (tmp_path / "eval_results_v3.json").write_text(json.dumps(v3))
    (tmp_path / "eval_results_v21.json").write_bytes((ROOT / "eval_results_v21.json").read_bytes())
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    assert verify.check_v3_optimization() is False


def test_empty_linear_probes_fail(tmp_path, monkeypatch):
    readings = json.loads((ROOT / "probe_readings.json").read_text())
    readings["linear"] = {}
    (tmp_path / "probe_readings.json").write_text(json.dumps(readings))
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    assert verify.check_probe_readings() is False


@pytest.fixture
def ledger_paths(tmp_path, monkeypatch):
    directory = tmp_path / "ledger"
    directory.mkdir()
    for name, value in {"LEDGER_DIR": directory, "LEDGER": directory / "ledger.jsonl",
                        "KEY": directory / "operator.key", "PUB": directory / "operator.pub"}.items():
        monkeypatch.setattr(provenance, name, value)
    return directory


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_ledger_fails(ledger_paths, empty):
    key = Ed25519PrivateKey.generate()
    provenance.PUB.write_bytes(key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    if empty:
        provenance.LEDGER.write_text("")
    with pytest.raises(SystemExit, match="empty ledger|no ledger"):
        provenance.cmd_verify(None)


def test_keygen_preserves_deposited_public_key(ledger_paths):
    provenance.PUB.write_bytes(b"existing identity")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        provenance.cmd_keygen(None)
    assert provenance.PUB.read_bytes() == b"existing identity"
    assert not provenance.KEY.exists()


def test_signed_ledger_passes_and_modified_artifact_fails(ledger_paths):
    provenance.cmd_keygen(None)
    artifact = ledger_paths.parent / "artifact.txt"
    artifact.write_text("original")
    provenance.cmd_add(SimpleNamespace(file=str(artifact), type="test", parent=[], note=""))
    with pytest.raises(SystemExit) as result:
        provenance.cmd_verify(None)
    assert result.value.code == 0
    artifact.write_text("modified")
    with pytest.raises(SystemExit) as result:
        provenance.cmd_verify(None)
    assert result.value.code == 1


def test_default_outputs_preserve_reference_records():
    import train_eval_v2
    import train_eval_v3
    assert train_eval_v2.OUT == ROOT / "outputs" / "eval_results_v21.json"
    assert train_eval_v3.OUT == ROOT / "outputs" / "eval_results_v3.json"
