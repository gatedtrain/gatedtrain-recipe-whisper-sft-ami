#!/usr/bin/env python
"""Per-utterance WER evaluation of a Whisper cut (or the base alone) for prototype area "audio"
(row 5 of the multimodal prototype scope; prototype, unpromotable).

An item is one registered utterance (design 05 section 3.3): an AMI IHM test-split segment held out from training
by meeting and by speaker (README.md "Data"). Per item: one transcript from the deployment decoder (the cut's
decoding.json: greedy, one beam, no sampling, no temperature fallback, short-form), the same items in the same order
and batch composition for every arm. K = 1: greedy decoding is deterministic, so a second repeat would repeat the
first. Every transcript is kept (`outputs/<item>.json`, SHA-256 in `output_ref`).

Metric (README.md "Metric and tier"):
  wer   deciding, lower is better, [0, null]: word error rate of the utterance, (S + D + I) / N with N the number of
        reference words, after the Whisper English text normalizer (transformers EnglishTextNormalizer with the
        base model's normalizer.json) on both reference and hypothesis; word-level Levenshtein distance. An empty
        transcript is a score (all deletions, WER 1); a hallucinated transcript can exceed 1.
Descriptive sub-metrics (never deciding): substitutions, deletions, insertions, ref_words, hyp_words, empty. The
corpus WER (sum of errors / sum of reference words, a ratio of sums) is reported in summary.json only.

  python eval_asr.py (--checkpoint CUT | --base-only) --items DIR --out DIR --eval-seeds 11
                     [--model-hash H] [--base-model-hash H] [--item-set-hash H] [--limit N] [--device cuda]
      writes DIR/outputs/<item>.json, DIR/scores.jsonl, DIR/summary.json, DIR/env-runtime.json and one MM-1/MM-2 arm
      DIR/arm-wer/{items.jsonl, eval_meta.json, outputs/ (hard links)}

The arm needs ptc-hub hashes this runtime does not compute (the cut's `model_hash`, MM-17, and the item set's
manifest hash, MM-6): pass them, or let ptc-hub fill `eval_meta.json` (README.md "What ptc-hub needs").
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_JSON = HERE / "models" / "MODELS.json"
BASE_NAME = "whisper-base.en"
DEFAULT_BASE = HERE / "models" / BASE_NAME
NORMALIZER_JSON = HERE / "models" / BASE_NAME / "normalizer.json"     # the scorer's, never the candidate's
ADAPTER = "hf_seq2seq"
EVALUATOR_ID = "asr.whisper_items@0.1"
METRIC = "wer"
SCORER_VERSION = "eval_asr@0.1"
ERROR_CLASSES = ("decode_error", "scorer_error")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# BEGIN SHARED DECODER (byte-identical in train_whisper.py and eval_asr.py; its SHA-256 is `decoder_code_sha256`)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
DECODING = {
    # The deployment decoder (05 section 3.3), stored in every cut as decoding.json; eval_asr.py checks it.
    "strategy": "greedy",
    "num_beams": 1,
    "do_sample": False,
    "temperature_fallback": None,       # no fallback: one decode at temperature 0 (argmax)
    "max_new_tokens": 200,              # batch bound: >= 3x the longest reference (65 tokens)
    # Per utterance: at most 16 + ceil(12 x seconds) text tokens (the first ones of the greedy output; identical to
    # generating with that bound, greedy tokens do not depend on later ones). The fastest reference is 9.7
    # tokens/s, every reference is under half its bound; a repetition loop is cut at a length proportional to the
    # audio instead of 200 tokens (README.md "Deployment decoder").
    "max_text_tokens_rule": "16 + ceil(12 * len(audio) / 16000)",
    "return_timestamps": False,
    "long_form": False,                 # every utterance <= 30 s: one window, zero padded to 30 s
    "language": "en",                   # English-only checkpoint (.en): no language or task tokens
    "input": "WhisperFeatureExtractor log-mel, 80 bins x 3000 frames, from the checkpoint's processor",
    "sample_rate": SAMPLE_RATE,
}


def decoding_doc() -> dict:
    return dict(DECODING)


def load_audio(path, want_sha256):
    """16 kHz mono float32 samples of a WAV file whose SHA-256 must equal the recorded one."""
    import soundfile as sf

    raw = Path(path).read_bytes()
    got = "sha256:" + hashlib.sha256(raw).hexdigest()
    if got != want_sha256:
        raise ValueError(f"{path}: SHA-256 {got} is not the recorded {want_sha256}")
    import io

    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE or audio.ndim != 1:
        raise ValueError(f"{path}: {sr} Hz, shape {audio.shape}; 16 kHz mono expected")
    return audio


def max_text_tokens(audio) -> int:
    return min(DECODING["max_new_tokens"], 16 + math.ceil(12 * len(audio) / SAMPLE_RATE))


def transcribe(model, processor, audios, device, dtype):
    """Greedy short-form transcripts of one batch (DECODING): the text tokens (ids below the tokenizer's first
    special token, i.e. before end-of-text) cut to the utterance's bound, decoded without special tokens.
    Returns (texts, truncated flags)."""
    import torch

    feats = processor.feature_extractor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")["input_features"]
    with torch.no_grad():
        out = model.generate(input_features=feats.to(device, dtype), num_beams=DECODING["num_beams"],
                             do_sample=DECODING["do_sample"], max_new_tokens=DECODING["max_new_tokens"],
                             return_timestamps=DECODING["return_timestamps"])
    eot = processor.tokenizer.eos_token_id          # 50256 '<|endoftext|>'; every special id is >= it
    texts, cut = [], []
    for seq, audio in zip(out.tolist(), audios):
        ids = [t for t in seq if t < eot]
        bound = max_text_tokens(audio)
        cut.append(len(ids) > bound)
        texts.append(processor.tokenizer.decode(ids[:bound], skip_special_tokens=True))
    return texts, cut

# ---------------------------------------------------------------------------
# END SHARED DECODER
# ---------------------------------------------------------------------------


def decoder_code_sha256() -> str:
    text = Path(__file__).read_text(encoding="utf-8")
    start = text.index("# BEGIN SHARED DECODER")
    end = text.index("# END SHARED DECODER")
    return sha256_bytes(text[start:end].encode("utf-8"))


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def english_normalizer():
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer(json.loads(NORMALIZER_JSON.read_text(encoding="utf-8")))


def edit_counts(ref: list[str], hyp: list[str]) -> dict:
    """Word-level Levenshtein alignment: minimal S + D + I; among minimal alignments the backtrace prefers a
    match or substitution, then a deletion, then an insertion (deterministic)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    s = de = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            de += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    assert s + de + ins == d[n][m]
    return {"substitutions": s, "deletions": de, "insertions": ins, "errors": d[n][m]}


def score(norm, reference_text: str, hypothesis: str) -> dict:
    ref = norm(reference_text).split()
    hyp = norm(hypothesis).split()
    if not ref:
        raise ValueError("the normalized reference is empty")
    c = edit_counts(ref, hyp)
    return {"wer": c["errors"] / len(ref), "ref_words": len(ref), "hyp_words": len(hyp), "empty": int(not hyp),
            "reference_normalized": " ".join(ref), "hypothesis_normalized": " ".join(hyp), **c}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def check_cut(cut: Path) -> dict:
    marker = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
    for name, want in marker["files"].items():
        if sha256_file(cut / name) != want:
            raise SystemExit(f"{cut / name} differs from its SHA-256 in cut.json")
    if marker.get("decoder_code_sha256") != decoder_code_sha256():
        raise SystemExit("the cut was written with another decoder code than this evaluator's")
    if json.loads((cut / "decoding.json").read_text(encoding="utf-8")) != decoding_doc():
        raise SystemExit("the cut's decoding.json is not this evaluator's deployment decoder")
    base_ref = json.loads((cut / "base_ref.json").read_text(encoding="utf-8"))
    record = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"][BASE_NAME]
    if base_ref["revision"] != record["revision"] or base_ref["files"] != {k: v["sha256"] for k, v in
                                                                           record["files"].items()}:
        raise SystemExit(f"the cut names another base model than models/{BASE_NAME}")
    return marker


def read_items(items_dir: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in (items_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    ids = [r["item"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("item ids are not unique")
    return rows


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())


def stats(vals) -> dict:
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if not n:
        return {"n": 0}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    counts: dict[float, int] = {}
    for v in vals:
        counts[round(v, 9)] = counts.get(round(v, 9), 0) + 1
    mode, mode_n = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals), "distinct": len(counts),
            "mode": mode, "mode_share": mode_n / n, "share_zero": sum(v == 0 for v in vals) / n,
            "share_ge_1": sum(v >= 1 for v in vals) / n}


def evaluate(args) -> int:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    t0 = time.time()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    cut = args.checkpoint.resolve() if args.checkpoint else None
    marker = check_cut(cut) if cut else None
    items = read_items(args.items)
    if args.limit:
        items = items[: args.limit]
    seeds = [int(x) for x in args.eval_seeds.split(",")]
    if len(seeds) != 1:
        raise SystemExit("K = 1: greedy decoding is deterministic; pass exactly one evaluation seed")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seeds[0])                     # greedy decoding draws nothing; recorded for completeness

    model_dir = cut if cut is not None else args.base.resolve()
    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir), torch_dtype=dtype,
                                                            local_files_only=True).to(device).eval()
    norm = english_normalizer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    scores = []
    for b0 in range(0, len(items), args.batch_size):
        batch = items[b0:b0 + args.batch_size]
        audios, rows = [], []
        for it in batch:
            audios.append(load_audio(args.items / it["audio"], it["audio_sha256"]))
        try:
            hyps, truncated = transcribe(model, processor, audios, device, dtype)
            err = None
        except Exception as e:                       # decoder failure: an error record per item (MM-4)
            hyps, truncated = [None] * len(batch), [None] * len(batch)
            err = {"class": "decode_error", "message": f"{type(e).__name__}: {e}"[:300]}
        for it, hyp, trunc in zip(batch, hyps, truncated):
            row = {"item": it["item"], "repeat": 0, "seed": seeds[0], "hypothesis": hyp, "error": err,
                   "truncated": None if trunc is None else int(trunc),
                   "speaker_id": it["speaker_id"], "meeting_id": it["meeting_id"]}
            if err is None:
                try:
                    sc = score(norm, it["text"], hyp)
                    if sc["reference_normalized"] != it["reference_normalized"]:
                        raise ValueError("the normalized reference differs from items.jsonl reference_normalized")
                    row.update(sc)
                except Exception as e:
                    row["error"] = {"class": "scorer_error", "message": f"{type(e).__name__}: {e}"[:300]}
            rel = f"outputs/{it['item']}.json"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            write_json(out / rel, {"item": it["item"], "hypothesis": hyp,
                                   "hypothesis_normalized": row.get("hypothesis_normalized"),
                                   "reference_normalized": it["reference_normalized"]})
            row["output_ref"] = {"path": rel, "sha256": sha256_file(out / rel)}
            rows.append(row)
        scores.extend(rows)
        print(f"{len(scores)}/{len(items)} items, {time.time() - t0:.0f} s", flush=True)
    wall = time.time() - t0
    write_jsonl(out / "scores.jsonl", scores)

    code_sha = sha256_file(Path(__file__).resolve())
    import transformers

    sampler = {**decoding_doc(), "batch_size": args.batch_size, "dtype": args.dtype,
               "order": "items.jsonl order, fixed batches"}
    sampler_hash = sha256_bytes(canonical(sampler))
    normalizer_sha = sha256_file(NORMALIZER_JSON)
    scorer = {"name": "per_utterance_wer",
              "version": f"{SCORER_VERSION} transformers {transformers.__version__} EnglishTextNormalizer "
                         f"normalizer.json {normalizer_sha}; word-level Levenshtein",
              "weights_sha256": None, "code_sha256": code_sha}
    arm = out / f"arm-{METRIC}"
    arm.mkdir()
    lines = []
    for row in scores:
        link_or_copy(out / row["output_ref"]["path"], arm / row["output_ref"]["path"])
        err = row["error"]
        sub = None if err else {k: row[k] for k in ("substitutions", "deletions", "insertions", "ref_words",
                                                     "hyp_words", "empty", "truncated")}
        lines.append({"item": row["item"], "repeat": 0, "seed": row["seed"], "score": None if err else row["wer"],
                      "error": err, "sub_metrics": sub, "output_ref": row["output_ref"]})
    write_jsonl(arm / "items.jsonl", lines)
    write_json(arm / "eval_meta.json", {
        "schema": "ptc_hub.item_eval", "schema_version": "1.0",
        "evaluator_id": EVALUATOR_ID, "adapter": ADAPTER, "metric": METRIC, "direction": "lower_is_better",
        "score_range": [0, None],
        "checkpoint": {"model_hash": args.model_hash, "base_model_hash": args.base_model_hash},
        "normalizer_sha256": None, "env_version_sha256": None,
        "item_set_hash": args.item_set_hash, "eval_seeds": seeds,
        "sampler": sampler, "sampler_hash": sampler_hash, "scorer": scorer,
        "framework": {"name": "transformers", "version": transformers.__version__},
        "created_at": utc_now(),
    })

    ok = [r for r in scores if r["error"] is None]
    tot_err, tot_ref = sum(r["errors"] for r in ok), sum(r["ref_words"] for r in ok)
    wer_stats = stats([r["wer"] for r in ok])
    summary = {"checkpoint": str(cut) if cut else None, "base_only": cut is None,
               "global_step": marker["global_step"] if marker else 0, "items": len(items), "repeats": 1,
               "errors": len(scores) - len(ok), "empty_transcripts": sum(r["empty"] for r in ok),
               "truncated_transcripts": sum(r["truncated"] for r in ok),
               "wer": wer_stats, "corpus_wer": tot_err / tot_ref if tot_ref else None,
               "totals": {"errors": tot_err, "ref_words": tot_ref,
                          **{k: sum(r[k] for r in ok) for k in ("substitutions", "deletions", "insertions")}},
               "degeneracy_mm38": {"mode_share_lt_0_5": wer_stats.get("mode_share", 1) < 0.5,
                                   "sd_gt_0": bool(wer_stats.get("sd")),
                                   "mean_ge_0_05": wer_stats.get("mean", 0) >= 0.05},
               "wall_seconds": round(wall, 1), "seconds_per_item": round(wall / max(1, len(scores)), 3),
               "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
               "sampler_hash": sampler_hash, "device": str(device), "dtype": str(dtype),
               "missing_hashes": [k for k, v in (("model_hash", args.model_hash),
                                                  ("item_set_hash", args.item_set_hash)) if v is None]}
    if device.type == "cuda":
        summary["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
        summary["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
    lock = HERE / "uv.lock"
    write_json(out / "env-runtime.json", {
        "python": sys.version.split()[0], "executable": sys.executable,
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None, "script_sha256": code_sha,
        "decoder_code_sha256": decoder_code_sha256(), "torch": torch.__version__,
        "transformers": transformers.__version__, "soundfile": importlib.metadata.version("soundfile"),
        "normalizer_json_sha256": normalizer_sha, "model_dir": str(model_dir),
        "checkpoint": str(cut) if cut else None, "items": str(args.items.resolve()),
        "items_sha256": sha256_file(args.items / "items.jsonl"), "uid": os.getuid(), "gid": os.getgid()})
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--base-only", action="store_true")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--items", type=Path, default=HERE / "items")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-seeds", default="11")
    p.add_argument("--model-hash")
    p.add_argument("--base-model-hash")
    p.add_argument("--item-set-hash")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("fp16", "fp32"), default="fp32")
    args = p.parse_args(argv)
    if (args.checkpoint is None) == (not args.base_only):
        p.error("exactly one of --checkpoint / --base-only")
    return args


def main(argv=None) -> int:
    return evaluate(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
