#!/usr/bin/env python
"""Download the ungated Whisper checkpoints and AMI shards of area "audio" into plain directories (host step,
network on, no GPU).

Models go to `models/<name>/`, dataset shards to `data-source/ami-ihm/`, as plain files (no symbolic links,
MM-20). huggingface_hub's own download metadata (`<local_dir>/.cache/huggingface/`) is moved out to
`hf-meta/<name>/`, so each directory holds exactly the repository files at the pinned commit. No token is used or
read: `token=False` on every call, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and `HF_HOME` points to the fresh
`hf-home/` of this directory (never an existing cache); a token in the environment is removed from this process's
environment before huggingface_hub is imported (its value is never read). Xet transfer is disabled (plain HTTPS).

After the download, `models/MODELS.json` and `data-source/DATA.json` record per repository: repo, revision (40-hex
commit), license, the files and their SHA-256 and bytes; `--verify` re-hashes and compares.

Usage (host):
  .venv/bin/python fetch_models.py            # download what is missing, write MODELS.json and DATA.json
  .venv/bin/python fetch_models.py --verify   # re-hash every file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
DATA_SOURCE = HERE / "data-source"

# Revisions resolved 2026-11-12 from the Hub API (`sha` of `main`); every repo reported `gated: false`.
WHISPER_FILES = ["config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
                 "added_tokens.json", "merges.txt", "normalizer.json", "special_tokens_map.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json"]
MODEL_SPECS = {
    # name -> (repo, revision, files, license). Only safetensors weights (no .bin, .h5, .msgpack).
    # whisper-tiny.en (openai/whisper-tiny.en @ 87c7102498dcde7456f24cfd30239ca606ed9063) was downloaded for the CPU
    # pilot and removed after the choice of base.en (README.md "Model choice").
    "whisper-base.en": ("openai/whisper-base.en", "911407f4214e0e1d82085af863093ec0b66f9cd6", WHISPER_FILES,
                        "apache-2.0"),
}
# AMI meeting corpus, individual headset microphones, Kaldi s5 segmentation and text (dataset card). Only the
# shards listed are downloaded (README.md "Data").
AMI_REPO, AMI_REVISION, AMI_LICENSE = "edinburghcstr/ami", "46f28f2503e2ec48f8867a84eef356c70476beab", "cc-by-4.0"
# Training: one shard per recording room/series (EN2001, ES2006-ES2007, IS1002-IS1003, TS3006-TS3007; meeting ranges
# from the shards' parquet statistics); items: the whole test split (EN2002, ES2004, IS1009, TS3003).
AMI_FILES = ["README.md",
             "ihm/train-00000-of-00042.parquet", "ihm/train-00010-of-00042.parquet",
             "ihm/train-00026-of-00042.parquet", "ihm/train-00034-of-00042.parquet",
             "ihm/test-00000-of-00004.parquet", "ihm/test-00001-of-00004.parquet",
             "ihm/test-00002-of-00004.parquet", "ihm/test-00003-of-00004.parquet"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def plain_files(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the directory must be plain (MM-20)")
        if p.is_file():
            out[p.relative_to(root).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def _hub_env() -> None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        os.environ.pop(name, None)          # removed, never read
    os.environ["HF_HOME"] = str(HERE / "hf-home")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    (HERE / "hf-home").mkdir(parents=True, exist_ok=True)


def _move_meta(dest: Path, name: str) -> None:
    meta = dest / ".cache"
    if meta.exists():
        target = HERE / "hf-meta" / name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(meta), str(target))


def download() -> int:
    _hub_env()
    from huggingface_hub import hf_hub_download

    record = {"schema": "ptc_mm.whisper_models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
              "models": {}}
    for name, (repo, revision, files, lic) in MODEL_SPECS.items():
        dest = MODELS / name
        for f in files:
            hf_hub_download(repo, f, revision=revision, local_dir=dest, token=False)
        _move_meta(dest, name)
        got = plain_files(dest)
        if sorted(got) != sorted(files):
            raise SystemExit(f"{dest} holds {sorted(got)}, not {sorted(files)}")
        record["models"][name] = {"repo": repo, "revision": revision, "license": lic, "files": got,
                                  "bytes": sum(f["bytes"] for f in got.values())}
        print(f"{name}: {len(got)} files, {record['models'][name]['bytes'] / 1e6:.1f} MB", flush=True)
    (MODELS / "MODELS.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    dest = DATA_SOURCE / "ami-ihm"
    for f in AMI_FILES:
        hf_hub_download(AMI_REPO, f, repo_type="dataset", revision=AMI_REVISION, local_dir=dest, token=False)
    _move_meta(dest, "ami-ihm")
    got = plain_files(dest)
    data = {"schema": "ptc_mm.whisper_data_source", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "datasets": {"ami-ihm": {"repo": AMI_REPO, "repo_type": "dataset", "revision": AMI_REVISION,
                                     "config": "ihm", "license": AMI_LICENSE, "files": got,
                                     "bytes": sum(f["bytes"] for f in got.values())}}}
    (DATA_SOURCE / "DATA.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ami-ihm: {len(got)} files, {data['datasets']['ami-ihm']['bytes'] / 1e6:.1f} MB", flush=True)
    return 0


def verify() -> int:
    bad = 0
    for record_path, key, root in ((MODELS / "MODELS.json", "models", MODELS),
                                   (DATA_SOURCE / "DATA.json", "datasets", DATA_SOURCE)):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for name, m in record[key].items():
            got = plain_files(root / name)
            ok = got == m["files"]
            bad += not ok
            print(f"{name}: {'ok' if ok else 'MISMATCH'} ({len(got)} files)", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verify", action="store_true")
    args = p.parse_args(argv)
    return verify() if args.verify else download()


if __name__ == "__main__":
    sys.exit(main())
