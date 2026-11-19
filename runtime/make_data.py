#!/usr/bin/env python
"""Build the training shard and the registered item set of area "audio" from the downloaded AMI IHM parquet shards
(host step, no network, no GPU).

Inputs: `data-source/ami-ihm/ihm/*.parquet` (checked against `data-source/DATA.json`), the Whisper English text
normalizer (transformers `EnglishTextNormalizer` with the base model's `normalizer.json`).

Outputs (both directories plain files only):
  data/train.jsonl        the training shard: one row per utterance {utt_id, text, speaker_id, meeting_id, audio,
                          audio_sha256, duration, n_words}; `audio` is relative to data/
  data/audio/<utt>.wav    the utterance's audio, the exact WAV bytes of the parquet cell (16 kHz, float32)
  data/data_meta.json     source, filters, split rule, counts, hashes
  items/items.jsonl       the registered items: one row per utterance {item, text, reference_normalized, audio,
                          audio_sha256, speaker_id, meeting_id, duration, n_words}
  items/audio/<item>.wav  as above
  items/item_set_meta.json

Split (README.md "Data"): training rows come only from the AMI train split (4 shards, one per recording series:
EN2001, ES2006-ES2007, IS1002-IS1003, TS3006-TS3007), items only from the AMI test split (EN2002, ES2004, IS1009,
TS3003): disjoint by meeting and, checked here, by speaker; exact raw-text and normalized-text overlap between
items and training rows is removed from the items (count recorded; 0 expected for utterances of 8+ words).

A fresh checkout already holds the committed `data/data_meta.json`, `items/items.jsonl` and `items/item_set_meta.json`
(COMMITTED). The script refuses only when generated content exists (GENERATED: `data/audio`, `data/train.jsonl`,
`items/audio`); after writing, it compares the rewritten COMMITTED files with the bytes they had before the run and
exits non-zero, naming them, if any differs (the registered item-set hash covers `items/`).

Usage:
  .venv/bin/python make_data.py [--n-train 3000] [--n-items 500] [--seed 20261112]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data-source"
AMI_DIR = SOURCE / "ami-ihm"
DATA = HERE / "data"
ITEMS = HERE / "items"
GENERATED = (DATA / "audio", DATA / "train.jsonl", ITEMS / "audio")
COMMITTED = (DATA / "data_meta.json", ITEMS / "items.jsonl", ITEMS / "item_set_meta.json")
NORMALIZER_JSON = HERE / "models" / "whisper-base.en" / "normalizer.json"   # the scorer's (eval_asr.py)
SPLIT_NAMESPACE = 0xA4D10   # numpy SeedSequence([SPLIT_NAMESPACE, seed])

# Filters (README.md "Data"). Training: utterances of at least 4 words (drops the backchannels "YEAH", "OKAY",
# "MM-HMM" that dominate AMI by count) and at most 30 s (Whisper's window). Items: at least 8 words after the
# Whisper English normalizer (fillers removed) and at most 30 s, so the per-utterance WER of the baseline is
# continuous rather than piled up at 0 or 1 (MM-38).
TRAIN_MIN_WORDS = 4
ITEM_MIN_NORM_WORDS = 8
MAX_SECONDS = 30.0


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def english_normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer(json.loads(NORMALIZER_JSON.read_text(encoding="utf-8")))


def read_split(split: str, record: dict) -> list[dict]:
    import pyarrow.parquet as pq

    rows = []
    names = sorted(k for k in record["files"] if k.startswith(f"ihm/{split}-"))
    for name in names:
        path = AMI_DIR / name
        if sha256_file(path) != record["files"][name]["sha256"]:
            raise SystemExit(f"{path} differs from its SHA-256 in {SOURCE / 'DATA.json'}")
        for r in pq.read_table(path).to_pylist():
            rows.append({"utt_id": r["audio_id"], "text": r["text"], "speaker_id": r["speaker_id"],
                         "meeting_id": r["meeting_id"], "microphone_id": r["microphone_id"],
                         "duration": round(float(r["end_time"]) - float(r["begin_time"]), 3),
                         "wav": r["audio"]["bytes"], "shard": name})
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-items", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261112)
    args = p.parse_args(argv)
    for t in GENERATED:
        if t.exists():
            raise SystemExit(f"{t} exists: remove the generated data first (the committed files may stay)")
    committed = {t: t.read_bytes() for t in COMMITTED if t.is_file()}
    import numpy as np
    import soundfile as sf
    import io

    record = json.loads((SOURCE / "DATA.json").read_text(encoding="utf-8"))["datasets"]["ami-ihm"]
    norm = english_normalizer()
    train_pool = [r for r in read_split("train", record)
                  if len(r["text"].split()) >= TRAIN_MIN_WORDS and r["duration"] <= MAX_SECONDS]
    test_all = read_split("test", record)
    for r in test_all:
        r["reference_normalized"] = norm(r["text"]).strip()
    item_pool = [r for r in test_all
                 if len(r["reference_normalized"].split()) >= ITEM_MIN_NORM_WORDS and r["duration"] <= MAX_SECONDS]
    rng = np.random.default_rng(np.random.SeedSequence([SPLIT_NAMESPACE, args.seed]))
    train_pool.sort(key=lambda r: r["utt_id"])
    item_pool.sort(key=lambda r: r["utt_id"])
    train_idx = sorted(int(i) for i in rng.permutation(len(train_pool))[: args.n_train])
    train = [train_pool[i] for i in train_idx]

    # Disjointness and overlap: speakers, raw text, normalized text.
    train_speakers = {r["speaker_id"] for r in train_pool}
    train_raw = {r["text"] for r in train_pool}
    train_norm = {norm(r["text"]).strip() for r in train_pool}
    before = len(item_pool)
    item_pool = [r for r in item_pool if r["speaker_id"] not in train_speakers]
    speaker_dropped = before - len(item_pool)
    before = len(item_pool)
    item_pool = [r for r in item_pool if r["text"] not in train_raw and r["reference_normalized"] not in train_norm]
    overlap_dropped = before - len(item_pool)
    item_idx = sorted(int(i) for i in rng.permutation(len(item_pool))[: args.n_items])
    items = [item_pool[i] for i in item_idx]

    for d in (DATA / "audio", ITEMS / "audio"):
        d.mkdir(parents=True, exist_ok=True)
    out_train, out_items = [], []
    for r in train:
        rel = f"audio/{r['utt_id']}.wav"
        (DATA / rel).write_bytes(r["wav"])
        info = sf.info(io.BytesIO(r["wav"]))
        if info.samplerate != 16000 or info.channels != 1:
            raise SystemExit(f"{r['utt_id']}: {info.samplerate} Hz, {info.channels} channels")
        out_train.append({"utt_id": r["utt_id"], "text": r["text"], "speaker_id": r["speaker_id"],
                          "meeting_id": r["meeting_id"], "audio": rel, "audio_sha256": sha256_bytes(r["wav"]),
                          "duration": r["duration"], "n_words": len(r["text"].split())})
    for r in items:
        rel = f"audio/{r['utt_id']}.wav"
        (ITEMS / rel).write_bytes(r["wav"])
        out_items.append({"item": r["utt_id"], "text": r["text"], "reference_normalized": r["reference_normalized"],
                          "speaker_id": r["speaker_id"], "meeting_id": r["meeting_id"], "audio": rel,
                          "audio_sha256": sha256_bytes(r["wav"]), "duration": r["duration"],
                          "n_words": len(r["reference_normalized"].split())})
    write_jsonl(DATA / "train.jsonl", out_train)
    write_jsonl(ITEMS / "items.jsonl", out_items)

    def series(rows):
        out = {}
        for r in rows:
            out[r["meeting_id"][:6]] = out.get(r["meeting_id"][:6], 0) + 1
        return dict(sorted(out.items()))

    import transformers

    source = {"repo": "edinburghcstr/ami", "revision": "46f28f2503e2ec48f8867a84eef356c70476beab", "config": "ihm",
              "license": "cc-by-4.0", "files": {k: v["sha256"] for k, v in record["files"].items()
                                                if k.endswith(".parquet")},
              "audio": "exact WAV bytes of the parquet `audio.bytes` cell (16 kHz mono float32, Kaldi s5 segments)",
              "text": "the dataset's `text` (Kaldi AMI s5 normalization: upper case, no punctuation)"}
    split = {"seed": args.seed, "generator": f"numpy default_rng(SeedSequence([{SPLIT_NAMESPACE:#x}, seed])): first "
             "permutation(len(train_pool))[:n_train] picks the training rows, then permutation(len(item_pool))"
             "[:n_items] picks the items; both pools sorted by utt_id first",
             "train_pool": {"split": "train", "min_words": TRAIN_MIN_WORDS, "max_seconds": MAX_SECONDS,
                            "rows": len(train_pool)},
             "item_pool": {"split": "test", "min_normalized_words": ITEM_MIN_NORM_WORDS, "max_seconds": MAX_SECONDS,
                           "rows_after_filters": len(item_pool), "dropped_shared_speaker": speaker_dropped,
                           "dropped_text_overlap": overlap_dropped},
             "n_train": len(out_train), "n_items": len(out_items),
             "train_speakers": sorted(train_speakers),
             "item_speakers": sorted({r["speaker_id"] for r in items}),
             "shared_speakers": sorted(train_speakers & {r["speaker_id"] for r in items}),
             "train_series": series(out_train), "item_series": series(out_items),
             "exact_overlap_raw": sum(r["text"] in train_raw for r in items),
             "exact_overlap_normalized": sum(r["reference_normalized"] in train_norm for r in items),
             "numpy": np.__version__}
    normalizer = {"name": "transformers EnglishTextNormalizer (Whisper's English normalizer)",
                  "transformers": transformers.__version__, "normalizer_json": str(NORMALIZER_JSON.relative_to(HERE)),
                  "normalizer_json_sha256": sha256_file(NORMALIZER_JSON)}
    write_json(DATA / "data_meta.json", {
        "schema": "ptc_mm.whisper_train_data", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "source": source, "split": split, "normalizer": normalizer,
        "train_sha256": sha256_file(DATA / "train.jsonl"),
        "audio_seconds": round(sum(r["duration"] for r in out_train), 1)})
    write_json(ITEMS / "item_set_meta.json", {
        "schema": "ptc_mm.whisper_item_set", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "source": source, "split": split, "normalizer": normalizer,
        "items_sha256": sha256_file(ITEMS / "items.jsonl"), "train_sha256": sha256_file(DATA / "train.jsonl"),
        "audio_seconds": round(sum(r["duration"] for r in out_items), 1)})
    print(json.dumps({"train": len(out_train), "items": len(out_items), "split": split}, indent=2, sort_keys=True))
    changed = changed_files(committed)
    if changed:
        raise SystemExit("regenerated files differ from the committed ones: "
                         + ", ".join(str(t.relative_to(HERE)) for t in changed)
                         + " (git diff shows how; the registered item-set and data hashes no longer hold)")
    if committed:
        print(f"reproduced the committed {', '.join(str(t.relative_to(HERE)) for t in committed)} byte for byte")
    return 0


def changed_files(before: dict[Path, bytes]) -> list[Path]:
    """The files of `before` (path -> bytes before the run) whose bytes now differ or that are missing."""
    return [t for t, b in before.items() if not t.is_file() or t.read_bytes() != b]


if __name__ == "__main__":
    sys.exit(main())
