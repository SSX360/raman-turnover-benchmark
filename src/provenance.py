"""Append-only, hash-chained, Ed25519-signed artifact ledger (Layer E).

Artifact paths inside the repository are recorded relative to the repository root, so a
deposited ledger verifies on any machine; absolute paths from older entries are accepted."""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

LEDGER_DIR = pathlib.Path(__file__).resolve().parent.parent / "ledger"
LEDGER = LEDGER_DIR / "ledger.jsonl"
KEY = LEDGER_DIR / "operator.key"
PUB = LEDGER_DIR / "operator.pub"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "no-git"
    except Exception:
        return "no-git"


def load_private():
    k = serialization.load_pem_private_key(KEY.read_bytes(), password=None)
    if not isinstance(k, Ed25519PrivateKey):
        sys.exit("operator.key is not an Ed25519 private key")
    return k


def load_public():
    k = serialization.load_pem_public_key(PUB.read_bytes())
    if not isinstance(k, Ed25519PublicKey):
        sys.exit("operator.pub is not an Ed25519 public key")
    return k


def canonical(entry):
    e = {k: v for k, v in entry.items() if k not in ("entry_hash", "sig")}
    return json.dumps(e, sort_keys=True, separators=(",", ":")).encode()


def cmd_keygen(_):
    LEDGER_DIR.mkdir(exist_ok=True)
    if KEY.exists() or PUB.exists():
        sys.exit("key material already exists; refusing to overwrite the signing identity")
    priv = Ed25519PrivateKey.generate()
    KEY.write_bytes(
        priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    os.chmod(KEY, 0o600) if os.name != "nt" else None
    PUB.write_bytes(
        priv.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    print("wrote", KEY, "(back this up SEPARATELY from the data)")
    print("wrote", PUB)


def read_entries():
    if not LEDGER.exists():
        return []
    return [
        json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()
    ]


def find_by_sha(prefix):
    for e in read_entries():
        if e["sha256"].startswith(prefix):
            return e
    return None


def cmd_add(args):
    if not KEY.exists():
        sys.exit("run: python provenance.py keygen")
    path = pathlib.Path(args.file).resolve()
    if not path.exists():
        sys.exit("no such file: %s" % path)
    sha = sha256_file(path)
    parents = []
    for p in args.parent or []:
        pe = None
        pp = pathlib.Path(p)
        if pp.exists():
            want = sha256_file(pp)
            for e in read_entries():
                if e["sha256"] == want:
                    pe = e
                    break
            if pe is None:
                sys.exit("parent %s is not in the ledger; add it first" % p)
            parents.append(want)
        else:
            pe = find_by_sha(p)
            if pe is None:
                sys.exit("parent %s not found as path or ledger sha" % p)
            parents.append(pe["sha256"])
    entries = read_entries()
    prev = entries[-1]["entry_hash"] if entries else "GENESIS"
    root = LEDGER_DIR.parent
    try:
        artifact = path.relative_to(root).as_posix()
    except ValueError:
        artifact = str(path)
    entry = {
        "seq": len(entries),
        "artifact": artifact,
        "sha256": sha,
        "size": path.stat().st_size,
        "type": args.type,
        "parents": parents,
        "note": args.note or "",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "prev_hash": prev,
    }
    entry_hash = hashlib.sha256(canonical(entry)).hexdigest()
    entry["entry_hash"] = entry_hash
    entry["sig"] = load_private().sign(entry_hash.encode()).hex()
    LEDGER_DIR.mkdir(exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    print("[%03d] added %s %s" % (entry["seq"], entry["sha256"][:8], path.name))


def check_entry(e, pub, prev_expected):
    sig_ok = False
    try:
        pub.verify(bytes.fromhex(e["sig"]), e["entry_hash"].encode())
        sig_ok = True
    except Exception:
        pass
    entry_ok = hashlib.sha256(canonical(e)).hexdigest() == e["entry_hash"]
    chain_ok = e["prev_hash"] == prev_expected
    p = pathlib.Path(e["artifact"])
    if not p.is_absolute():
        p = LEDGER_DIR.parent / p
    if p.exists():
        file_state = "OK" if sha256_file(p) == e["sha256"] else "MODIFIED"
    else:
        file_state = "absent(remote)"
    return sig_ok, entry_ok, chain_ok, file_state


def cmd_verify(_):
    if not LEDGER.exists():
        sys.exit("no ledger; place the deposited ledger.jsonl in ledger/ before verifying")
    if not PUB.exists():
        sys.exit("no public key; obtain the public key for the deposited ledger")
    pub = load_public()
    entries = read_entries()
    if not entries:
        sys.exit("empty ledger; no provenance evidence to verify")
    failed = 0
    prev = "GENESIS"
    for e in entries:
        sig_ok, entry_ok, chain_ok, file_state = check_entry(e, pub, prev)
        ok = sig_ok and entry_ok and chain_ok and file_state == "OK"
        if not ok:
            failed += 1
        print(
            "[%03d] %-4s %-28s sig=%s chain=%s entry=%s file=%s"
            % (
                e["seq"],
                "PASS" if ok else "FAIL",
                pathlib.Path(e["artifact"]).name[:28],
                "ok" if sig_ok else "BAD",
                "ok" if chain_ok else "BAD",
                "ok" if entry_ok else "BAD",
                file_state,
            )
        )
        prev = e["entry_hash"]
    print()
    print("%d entries, %d failed" % (len(entries), failed))
    sys.exit(1 if failed else 0)


def cmd_show(args):
    e = find_by_sha(args.sha)
    if e is None:
        sys.exit("no artifact with sha prefix %s" % args.sha)
    entries = {x["sha256"]: x for x in read_entries()}
    chain = []
    cur = e
    depth = 0
    while cur is not None and depth < 64:
        chain.append(cur)
        if not cur["parents"]:
            break
        cur = entries.get(cur["parents"][0])
        depth += 1
    for i, c in enumerate(chain):
        print(
            "%s[%03d] %s %s (%s, %d bytes) %s"
            % (
                "  " * i,
                c["seq"],
                c["sha256"][:8],
                pathlib.Path(c["artifact"]).name,
                c["type"],
                c["size"],
                c["note"],
            )
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keygen")
    a = sub.add_parser("add")
    a.add_argument("file")
    a.add_argument("--type", required=True)
    a.add_argument("--parent", action="append")
    a.add_argument("--note", default="")
    sub.add_parser("verify")
    s = sub.add_parser("show")
    s.add_argument("sha")
    args = ap.parse_args()
    {"keygen": cmd_keygen, "add": cmd_add, "verify": cmd_verify, "show": cmd_show}[
        args.cmd
    ](args)


if __name__ == "__main__":
    main()
