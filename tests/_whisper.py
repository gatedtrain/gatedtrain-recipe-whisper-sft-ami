"""Area "audio" test helpers (verifier-owned; tests/README.md section 11.21, EXT-138 to EXT-153).

Written from the runtime (`runtime/`: README.md "Trainer", "Cut layout",
"Outputs of `train_whisper.py`", "Evaluator outputs", "What gatedtrain needs", "Launch request facts";
`train_whisper.py` `save_cut`, `base_binding`, `main`; `eval_asr.py` `evaluate`, `check_cut`; read 2026-11-13; where the
two differ the code is followed) and its CPU sanity evidence, not from gatedtrain. Small real pieces are copied into
`fixtures/ext/whisper/`: `env-freeze.txt` (the area venv's distributions, listed with `importlib.metadata` as
`write_env_freeze` does), `sampler.json` (the evaluation sampler of `sanity-evidence/cpu-sanity-base-arm-wer`),
`decoding.json` (the deployment decoder: that sampler without `batch_size`, `dtype`, `order`), and three real rows each
of `items/items.jsonl` and `data/train.jsonl` (`items.sample.jsonl`, `train.sample.jsonl`, shape checks only). Items,
training rows and audio are synthetic; weights are minimal safetensors stand-ins: nothing in gatedtrain loads them
(MM-12).

Layout of `WhisperScenario(root)`:
- `<root>/area4-whisper/` the runtime stand-in (mounted read-only at `/inputs/a4`, EXT-112): `train_whisper.py`,
  `eval_asr.py`, `.venv/bin/python` (stand-ins, never executed), `uv.lock`, `data/{train.jsonl, data_meta.json,
  audio/}` (N_TRAIN synthetic utterances), `items/{items.jsonl, item_set_meta.json, audio/}`,
  `models/whisper-base.en/` (the 11 file names of the real base, stand-in contents) and `models/MODELS.json`;
- `<root>/out/area4-whisper/full-1/`: one run of the fixture profile (batch 8, 100 rows: 13 steps per epoch, the 13th
  carrying 4 examples; 30 steps, cuts 10, 20, 30) with run_args.json, env-freeze.txt, env-runtime.json, samples.jsonl,
  train_steps.jsonl, train_log.jsonl, memory.jsonl, attempts.jsonl;
- `<root>/eval/{base,cand}/`: two evaluation runs as `eval_asr.py` writes them (`outputs/`, `scores.jsonl`,
  `summary.json`, `env-runtime.json`, `arm-wer/` with hard-linked outputs); by default the gatedtrain hashes are filled;
- spec, registration (`wer`, lower, superiority, margin 0, sigma 0.12 fixed, planning effect 0.05, K 1, seed 11,
  `shared_annotation`), trainer.yaml, a GPU launch record, a smoke record and the host work's GPU log.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pwd
import random
import shutil
import unittest
from pathlib import Path
from typing import Any, Callable

import yaml

EXT = Path(__file__).resolve().parent / "fixtures"     # the recipe's fixtures (gatedtrain: tests/fixtures/ext)
from gatedtrain.testing.ext import ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER, ENV_INTERPRETER_TREES, ENV_OUTPUT_ROOT, IMAGE_ID, gpu_step_script, host_config_sha256, launch_label, sha_of, write_canonical
from gatedtrain.testing.ext import safetensors_bytes
from gatedtrain.ext.arm import META_SCHEMA
from gatedtrain.ext.launch import RECORD_SCHEMA
from gatedtrain.ext.smoke import SMOKE_SCHEMA
EXT = Path(__file__).resolve().parent / "fixtures"     # the recipe's fixtures (gatedtrain: tests/fixtures/ext)
from gatedtrain.testing.support import canonical_bytes, file_sha256, manifest_entries, manifest_hash, run_cli

WHISPER = EXT / "whisper"
ADAPTER = "hf_seq2seq"
TRAINER_ID = "sft.hf_seq2seq@0.1"
EVALUATOR_ID = "asr.whisper_items@0.1"
METRIC = "wer"
REPO = Path(__file__).resolve().parents[1]
AREA4 = Path(__import__("gatedtrain_recipe_whisper_sft_ami.rows", fromlist=["RUNTIME"]).RUNTIME)  # the runtime directory
EVIDENCE = REPO / "results" / "2026-11-13" / "sanity-evidence"
EVAL_ASR = AREA4 / "eval_asr.py"
TRAIN_WHISPER = AREA4 / "train_whisper.py"
EVAL_ASR_SHA256 = "sha256:b53355619b480361ddbcde8d8ad5eee072275797aa2b908ae576275f08b21f84"
# The SHARED DECODER block of both scripts (README "Deployment decoder", `decoder_code_sha256`).
DECODER_CODE_SHA256 = "sha256:478dd30eae25241a8e63b4437a89f1f8a064a1fccb77990cdef1a4fc6088cc5f"
NORMALIZER_SHA256 = "sha256:bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"
SCORER = {"name": "per_utterance_wer",
          "version": (f"eval_asr@0.1 transformers 5.21.0 EnglishTextNormalizer normalizer.json {NORMALIZER_SHA256}; "
                      "word-level Levenshtein"),
          "weights_sha256": None, "code_sha256": EVAL_ASR_SHA256}
SAMPLER = json.loads((WHISPER / "sampler.json").read_text(encoding="utf-8"))
DECODING = json.loads((WHISPER / "decoding.json").read_text(encoding="utf-8"))
EVAL_ONLY_KEYS = ("batch_size", "dtype", "order")
SEEDS = [11]
# train_whisper.py DEPLOYABLE and CUT_FILES; cut.json is written last.
DEPLOYABLE = ("model.safetensors", "config.json", "generation_config.json", "processor_config.json", "tokenizer.json",
              "tokenizer_config.json", "decoding.json")
CUT_FILES = DEPLOYABLE + ("base_ref.json", "trainer_state.json")
# The 11 files of models/whisper-base.en (MODELS.json).
BASE_FILES = ("added_tokens.json", "config.json", "generation_config.json", "merges.txt", "model.safetensors",
              "normalizer.json", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
              "tokenizer_config.json", "vocab.json")
BASE_NAME = "whisper-base.en"
REPO = "openai/whisper-base.en"
REVISION = "911407f4214e0e1d82085af863093ec0b66f9cd6"
RUNTIME_CONTAINER = "/inputs/a4"
SEED = 4000
SCRIPT_TEXT = "# fixture stand-in for runtime/train_whisper.py; never executed\n"
EVAL_TEXT = "# fixture stand-in for runtime/eval_asr.py; never executed\n"
INTERPRETER_TEXT = "# fixture stand-in for the area-4 venv interpreter; never executed\n"
LABEL = "mm-a4-whisper-full"
AREA = 5                                              # decision scope row 5; the directory is named after the run order
ENV_NAMES = ["HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "TOKENIZERS_PARALLELISM",
             "XDG_CACHE_HOME"]
N_TRAIN = 100
# train_whisper.py COMMON with a small profile: batch 8 (100 rows: 12 x 8 + 4, 13 steps per epoch), 30 steps, cuts 10,
# 20, 30.
COMMON = {"learning_rate": 1.0e-5, "lr_scheduler_type": "linear", "warmup_steps": 30, "weight_decay": 0.0,
          "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1.0e-8, "max_grad_norm": 1.0,
          "per_device_train_batch_size": 32, "gradient_accumulation_steps": 1, "precision": "bf16",
          "gradient_checkpointing": False, "dataloader_num_workers": 4, "logging_steps": 10, "full_determinism": False,
          "label_text": "AMI text lower-cased, one leading space (Whisper's transcript tokens start with a space)"}
PROFILE = {"max_steps": 30, "save_steps": 10, "per_device_train_batch_size": 8}


def run_user() -> str:
    """The container's `<uid>:<gid>` in fixture requests and records: the uid running the tests and its primary group,
    so the output directories the tests create are owned by the container's uid (EXT-63) whoever runs them. The launch
    rules refuse root, so a test that needs it skips under uid 0."""
    uid = os.getuid()
    if uid == 0:
        raise unittest.SkipTest("the launch rules refuse a container running as root (uid 0); run the tests as a "
                                "non-root user")
    return f"{uid}:{pwd.getpwuid(uid).pw_gid}"


def sha256_text(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def compact_hash(obj: Any) -> str:
    """The runtime's own hash encoding (`canonical` in both scripts): compact sorted JSON, not RFC 8785."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def write_json(path: Path, doc: Any) -> Path:
    """train_whisper.py / eval_asr.py `write_json`: indent 2, sorted keys, final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Base model, cuts and hashes (README "What gatedtrain needs" items 1-3; MM-17, MM-20, MM-21)
# ---------------------------------------------------------------------------

def build_base(d: Path, tag: str = "") -> Path:
    """A plain whisper-base.en directory with the real base's 11 file names; the weights a minimal safetensors file."""
    d.mkdir(parents=True, exist_ok=True)
    for rel in BASE_FILES:
        p = d / rel
        if rel.endswith(".safetensors"):
            p.write_bytes(safetensors_bytes(f"{rel} {tag}"))
        elif rel.endswith(".json"):
            p.write_text(json.dumps({"_fixture": rel, "tag": tag}, indent=2) + "\n", encoding="utf-8")
        else:
            p.write_text(f"fixture stand-in {rel} {tag}\n", encoding="utf-8")
    return d


def models_json(base: Path) -> dict:
    """models/MODELS.json as fetch_models.py writes it (the base entry only)."""
    files = {rel: {"bytes": (base / rel).stat().st_size, "sha256": file_sha256(base / rel)} for rel in BASE_FILES}
    return {"schema": "ptc_mm.whisper_models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "models": {BASE_NAME: {"repo": REPO, "revision": REVISION, "license": "apache-2.0",
                                   "bytes": sum(f["bytes"] for f in files.values()), "files": files}}}


def base_ref_doc(base: Path, *, revision: str = REVISION) -> dict:
    """train_whisper.py `base_binding` reduced to the cut's base_ref.json keys."""
    files = {rel: file_sha256(base / rel) for rel in BASE_FILES}
    return {"repo": REPO, "revision": revision, "files": files, "files_sha256": compact_hash(files)}


def build_cut(out: Path, step: int, *, examples_consumed: int, base: Path, script_sha256: str, profile: str = "full",
              seed: int = SEED, tag: str = "", decoding: dict | None = None) -> Path:
    """`checkpoint-<step>/` as train_whisper.py save_cut writes it, `cut.json` last."""
    d = out / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(safetensors_bytes(f"whisper cut {step} {tag}"))
    write_json(d / "config.json", {"_name_or_path": "/inputs/a4/models/whisper-base.en", "model_type": "whisper",
                                   "use_cache": True})
    write_json(d / "generation_config.json", {"decoder_start_token_id": 50257, "max_length": 448})
    write_json(d / "processor_config.json", {"feature_extractor": {"feature_size": 80, "sampling_rate": 16000}})
    write_json(d / "tokenizer.json", {"model": {"type": "BPE"}, "version": "1.0"})
    write_json(d / "tokenizer_config.json", {"tokenizer_class": "WhisperTokenizer"})
    write_json(d / "decoding.json", decoding if decoding is not None else DECODING)
    write_json(d / "base_ref.json", base_ref_doc(base))
    write_json(d / "trainer_state.json", {"global_step": step, "max_steps": PROFILE["max_steps"], "log_history": []})
    files = {n: file_sha256(d / n) for n in CUT_FILES}
    write_json(d / "cut.json", {
        "schema": "gatedtrain_mm.whisper_sft_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "global_step": step, "examples_consumed": examples_consumed, "files": files, "deployable": list(DEPLOYABLE),
        "script_sha256": script_sha256, "decoder_code_sha256": DECODER_CODE_SHA256, "seed": seed, "profile": profile})
    return d


def cut_model_hash(cut: Path) -> str:
    """MM-17: T0-09 manifest hash of the cut's `deployable` files (full weights; no composition with the base)."""
    entries = [e for e in manifest_entries(cut) if e["path"] in DEPLOYABLE]
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


# ---------------------------------------------------------------------------
# Data and a training run (train_whisper.py main: samples.jsonl and train_steps.jsonl before each optimizer step)
# ---------------------------------------------------------------------------

def utt_id(i: int) -> str:
    return f"AMI_EN2001a_H0{i % 4}_FIX{i % 7:03d}_{i * 1000:07d}_{i * 1000 + 450:07d}"


def build_data(d: Path, n: int = N_TRAIN, *, texts: dict[int, str] | None = None) -> list[dict]:
    """`data/` as make_data.py writes it: train.jsonl, data_meta.json and one (stand-in) WAV per row."""
    rows = []
    for i in range(n):
        uid = utt_id(i)
        audio = d / "audio" / f"{uid}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"RIFF" + f"fixture stand-in {uid}".encode())
        text = (texts or {}).get(i, f"WE NEED TO DECIDE THE THE BUTTON LAYOUT NUMBER {i}")
        rows.append({"utt_id": uid, "text": text, "speaker_id": f"FIX{i % 7:03d}", "meeting_id": "EN2001a",
                     "audio": f"audio/{uid}.wav", "audio_sha256": file_sha256(audio), "duration": 4.5,
                     "n_words": len(text.split())})
    write_jsonl(d / "train.jsonl", rows)
    write_json(d / "data_meta.json", {"schema": "ptc_mm.whisper_train_data", "n_rows": n})
    return rows


def build_items(d: Path, n: int, *, texts: dict[int, str] | None = None) -> list[dict]:
    """`items/` as make_data.py writes it: items.jsonl (the MM-6 item set), item_set_meta.json, stand-in WAVs."""
    rows = []
    for i in range(n):
        item = f"AMI_TS3003a_H0{i % 4}_ITEM{i:03d}_{i * 2000:07d}_{i * 2000 + 600:07d}"
        audio = d / "audio" / f"{item}.wav"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"RIFF" + f"fixture stand-in {item}".encode())
        text = (texts or {}).get(i, f"SO THE REMOTE CONTROL SHOULD BE UH ROUND AND YELLOW ITEM {i}")
        rows.append({"item": item, "text": text, "reference_normalized": text.lower(), "audio": f"audio/{item}.wav",
                     "audio_sha256": file_sha256(audio), "speaker_id": f"ITM{i % 5:03d}", "meeting_id": "TS3003a",
                     "duration": 6.0, "n_words": len(text.split())})
    write_jsonl(d / "items.jsonl", rows)
    write_json(d / "item_set_meta.json", {"schema": "ptc_mm.whisper_item_set", "n_items": n})
    return rows


def settings(profile: dict | None = None, seed: int = SEED) -> dict:
    s = {**COMMON, **(profile or PROFILE)}
    s["seed"] = seed
    return s


def simulate(rows: list[dict], s: dict, *, attempt: int = 1) -> tuple[list[dict], list[dict], dict[int, int]]:
    """(samples.jsonl lines, train_steps.jsonl lines, {step: examples consumed up to it}): each epoch a permutation of
    the rows in batches of `per_device_train_batch_size`, the last batch of an epoch short (dataloader_drop_last
    false); the epoch counter starts at 0 (on_epoch_begin increments -1)."""
    ids = [r["utt_id"] for r in rows]
    bs, total = s["per_device_train_batch_size"], s["max_steps"]
    samples, steps, consumed, n = [], [], {}, 0
    step, epoch = 0, -1
    while step < total:
        epoch += 1
        order = list(ids)
        random.Random(s["seed"] * 1000 + epoch).shuffle(order)
        for b0 in range(0, len(order), bs):
            if step >= total:
                break
            step += 1
            batch = order[b0:b0 + bs]
            for pos, uid in enumerate(batch):
                samples.append({"step": step, "epoch": epoch, "micro_batch": 0, "position": pos, "utt_id": uid,
                                "row": sha256_text(uid.encode("utf-8")), "attempt": attempt,
                                "utc": "2026-11-13T19:00:00Z"})
            steps.append({"step": step, "epoch": epoch, "n_examples": len(batch), "example_ids": batch,
                          "utc": "2026-11-13T19:00:00Z"})
            n += len(batch)
            consumed[step] = n
    return samples, steps, consumed


def run_args(*, s: dict, profile: str, attempt: int, base_container: str, train_container: str, train_sha256: str,
             files_sha256: str, script_sha256: str, n_train: int) -> dict:
    """train_whisper.py run_args.json (main, :467-490)."""
    return {
        "schema": "gatedtrain_mm.whisper_sft_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "profile": profile, "dry_run": False, "attempt": attempt, "settings": s, "max_steps": s["max_steps"],
        "seed": s["seed"], "seed_derivation": "Seq2SeqTrainingArguments(seed=S, data_seed=S) ...",
        "model_name_or_path": base_container, "base_model": base_container,
        "base_ref": {"repo": REPO, "revision": REVISION, "files_sha256": files_sha256, "verified": True},
        "train_file": train_container, "train_file_sha256": train_sha256, "n_train": n_train,
        "decoding": DECODING, "decoding_sha256": compact_hash(DECODING),
        "consumption_log": "samples.jsonl", "step_log": "train_steps.jsonl", "consumption_semantics": "...",
        "training_args": {"seed": s["seed"], "data_seed": s["seed"], "save_strategy": "no"},
        "device": "cuda", "gpu": "NVIDIA GeForce RTX 4090", "torch": "2.15.0", "torch_cuda": "13.2",
        "transformers": "5.21.0", "decoder_code_sha256": DECODER_CODE_SHA256, "script_sha256": script_sha256,
        "created_at": "2026-11-13T19:00:00Z",
    }


def build_run(out: Path, *, runtime: Path, base: Path, script_sha256: str, profile: str = "full",
              s: dict | None = None) -> tuple[Path, dict[int, int]]:
    s = s or settings()
    out.mkdir(parents=True)
    rows = read_jsonl(runtime / "data" / "train.jsonl")
    samples, steps, consumed = simulate(rows, s)
    shutil.copyfile(WHISPER / "env-freeze.txt", out / "env-freeze.txt")
    write_json(out / "env-runtime.json", {"python": "3.12.16", "executable": f"{RUNTIME_CONTAINER}/.venv/bin/python",
                                          "image_cuda_version": "13.0.3", "uv_lock_sha256": "sha256:" + "4" * 64,
                                          "script_sha256": script_sha256, "decoder_code_sha256": DECODER_CODE_SHA256,
                                          "uid": 1000, "gid": 1000, "cpu_count": 32,
                                          "env_freeze_sha256": file_sha256(out / "env-freeze.txt")})
    write_json(out / "run_args.json", run_args(
        s=s, profile=profile, attempt=1, base_container=f"{RUNTIME_CONTAINER}/models/{BASE_NAME}",
        train_container=f"{RUNTIME_CONTAINER}/data/train.jsonl",
        train_sha256=file_sha256(runtime / "data" / "train.jsonl"),
        files_sha256=base_ref_doc(base)["files_sha256"], script_sha256=script_sha256, n_train=len(rows)))
    write_jsonl(out / "samples.jsonl", samples)
    write_jsonl(out / "train_steps.jsonl", steps)
    write_jsonl(out / "train_log.jsonl", [{"step": st["step"], "loss": 1.0} for st in steps if st["step"] % 10 == 0])
    write_jsonl(out / "memory.jsonl", [{"step": k, "seconds_per_step": 0.3} for k in consumed if k % 10 == 0])
    write_jsonl(out / "attempts.jsonl", [{"attempt": 1, "started_at": "2026-11-13T19:00:00Z"}])
    for step, n in consumed.items():
        if step % s["save_steps"] == 0 or step == s["max_steps"]:
            build_cut(out, step, examples_consumed=n, base=base, script_sha256=script_sha256, profile=profile,
                      seed=s["seed"])
    return out, consumed


# ---------------------------------------------------------------------------
# Evaluation runs (eval_asr.py evaluate: outputs/, scores.jsonl, summary.json, env-runtime.json, arm-wer/)
# ---------------------------------------------------------------------------

REF_WORDS = 20


def baseline_errors(i: int) -> int:
    return 2 + i % 5


def candidate_errors(i: int) -> int:
    return max(0, baseline_errors(i) - 1 - i % 2)


def build_eval(out: Path, items: list[dict], errors_fn: Callable[[int], int], *, tag: str, model: str | None,
               item_set: str | None, filled: bool = True, base_only: bool = False) -> Path:
    """An evaluation run as eval_asr.py writes it. `filled`: model_hash and item_set_hash set (as after ext-arm-fill
    or passed on the command line); otherwise null, as the evaluator writes them without --model-hash and
    --item-set-hash. The sampler hash is the evaluator's compact-JSON hash (equal to RFC 8785 for this sampler)."""
    out.mkdir(parents=True)
    scores = []
    for i, it in enumerate(items):
        e = errors_fn(i)
        rel = f"outputs/{it['item']}.json"
        write_json(out / rel, {"item": it["item"], "hypothesis": f"{tag} hypothesis {i}",
                               "hypothesis_normalized": f"{tag} hypothesis {i}",
                               "reference_normalized": it["reference_normalized"]})
        scores.append({"item": it["item"], "repeat": 0, "seed": SEEDS[0], "error": None, "truncated": 0,
                       "wer": e / REF_WORDS, "substitutions": e, "deletions": 0, "insertions": 0,
                       "ref_words": REF_WORDS, "hyp_words": REF_WORDS, "empty": 0, "errors": e,
                       "output_ref": {"path": rel, "sha256": file_sha256(out / rel)}})
    write_jsonl(out / "scores.jsonl", scores)
    arm = out / f"arm-{METRIC}"
    arm.mkdir()
    lines = []
    for row in scores:
        dst = arm / row["output_ref"]["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(out / row["output_ref"]["path"], dst)
        sub = {k: row[k] for k in ("substitutions", "deletions", "insertions", "ref_words", "hyp_words", "empty",
                                   "truncated")}
        lines.append({"item": row["item"], "repeat": 0, "seed": row["seed"], "score": row["wer"], "error": None,
                      "sub_metrics": sub, "output_ref": row["output_ref"]})
    write_jsonl(arm / "items.jsonl", lines)
    write_json(arm / "eval_meta.json", {
        "schema": META_SCHEMA, "schema_version": "1.0", "evaluator_id": EVALUATOR_ID, "adapter": ADAPTER,
        "metric": METRIC, "direction": "lower_is_better", "score_range": [0, None],
        "checkpoint": {"model_hash": model if filled else None, "base_model_hash": None},
        "normalizer_sha256": None, "env_version_sha256": None, "item_set_hash": item_set if filled else None,
        "eval_seeds": list(SEEDS), "sampler": SAMPLER, "sampler_hash": compact_hash(SAMPLER), "scorer": dict(SCORER),
        "framework": {"name": "transformers", "version": "5.21.0"}, "created_at": "2026-11-13T21:00:00Z"})
    tot = sum(r["errors"] for r in scores)
    write_json(out / "summary.json", {"base_only": base_only, "items": len(items), "errors": 0,
                                      "corpus_wer": tot / (REF_WORDS * len(items)),
                                      "wer": {"mean": tot / (REF_WORDS * len(items))}})
    write_json(out / "env-runtime.json", {"script_sha256": EVAL_ASR_SHA256, "decoder_code_sha256": DECODER_CODE_SHA256,
                                          "normalizer_json_sha256": NORMALIZER_SHA256,
                                          "model_dir": f"{RUNTIME_CONTAINER}/models/{BASE_NAME}" if base_only
                                          else "/inputs/cut", "checkpoint": None if base_only else "/inputs/cut"})
    return out


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38) on the deciding metric `wer`: 20 baseline scores, range [0, null], not binary."""
    rec = {"schema": SMOKE_SCHEMA, "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": AREA, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 9.0, "host_ram_peak_gb": 12.0, "swap_peak_gb": 0.0, "seconds_per_step": 0.3,
           "projected_gpu_h": 0.1, "gpu_log_total_h": 4.43,
           "baseline_scores": [baseline_errors(i) / REF_WORDS for i in range(20)], "error_count": 0,
           "verdict": "PASS", "reasons": []}
    rec.update(overrides)
    return rec


def guard_log(host_work: Path, label: str, *, lines: list[str] | None = None) -> Path:
    """gpu-step.sh's log: one finished attempt of 3600 s."""
    host_work.mkdir(parents=True, exist_ok=True)
    lines = lines if lines is not None else [f"2026-11-11T09:00:00Z START {label} attempt=1",
                                             f"2026-11-11T10:00:00Z END {label} attempt=1 rc=0"]
    p = host_work / "gpu-guard.log"
    p.write_text("".join(l + "\n" for l in lines), encoding="utf-8")
    return p


class WhisperScenario:
    """A whole area-4 run collected against the whisper-base.en base (module docstring)."""

    def __init__(self, root: Path, *, n_items: int = 60, reg_mutate: Callable[[dict], None] | None = None,
                 overlap: int = 0, filled: bool = True, s: dict | None = None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.runtime = root / "area4-whisper"
        self.data = self.runtime / "data"
        self.items_dir = self.runtime / "items"
        texts = {i: f"OVERLAPPING TRANSCRIPT NUMBER {i}" for i in range(overlap)}
        self.train_rows = build_data(self.data, texts=texts)
        self.item_rows = build_items(self.items_dir, n_items, texts=texts)
        self.items = [r["item"] for r in self.item_rows]
        self.base = build_base(self.runtime / "models" / BASE_NAME)
        write_json(self.runtime / "models" / "MODELS.json", models_json(self.base))
        self.script = self.runtime / "train_whisper.py"
        self.script.write_text(SCRIPT_TEXT, encoding="utf-8")
        (self.runtime / "eval_asr.py").write_text(EVAL_TEXT, encoding="utf-8")
        (self.runtime / "uv.lock").write_text("# lock stand-in\n", encoding="utf-8")
        self.interpreter = self.runtime / ".venv" / "bin" / "python"
        self.interpreter.parent.mkdir(parents=True)
        self.interpreter.write_text(INTERPRETER_TEXT, encoding="utf-8")
        (root / "trainer.yaml").write_text("# area 4 trainer config (fixture)\nbackend: hf_seq2seq\nprofile: full\n",
                                           encoding="utf-8")
        self.host_work = root / "host-work"
        self.output_root = root / "out"
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((EXT / "gpu-log.fixture.md").read_bytes())
        self.settings = s or settings()
        self.stage_out, self.consumed = build_run(self.output_root / "area4-whisper" / "full-1", runtime=self.runtime,
                                                  base=self.base, script_sha256=file_sha256(self.script),
                                                  s=self.settings)
        self.cuts = sorted(self.stage_out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        self.final_cut = self.cuts[-1]
        self.base_model = manifest_hash(self.base)
        self.cand_model = cut_model_hash(self.final_cut)
        self.item_set_hash = manifest_hash(self.items_dir)
        self.reg_mutate = reg_mutate
        self.render()
        self.base_eval = build_eval(root / "eval" / "base", self.item_rows, baseline_errors, tag="base",
                                    model=self.base_model, item_set=self.item_set_hash, filled=filled, base_only=True)
        self.cand_eval = build_eval(root / "eval" / "cand", self.item_rows, candidate_errors, tag="cand",
                                    model=self.cand_model, item_set=self.item_set_hash, filled=filled)
        self.baseline = self.base_eval / f"arm-{METRIC}"
        self.candidate = self.cand_eval / f"arm-{METRIC}"
        self.launch = write_canonical(root / "launch.json", self.launch_record())
        self.label = self.launch_record()["label"]
        self.smoke = write_canonical(root / "smoke.json", smoke_record(base_model=self.base_model,
                                                                         item_set=self.item_set_hash))
        guard_log(self.host_work, self.label)

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: str(self.root), ENV_INTERPRETER: f"{RUNTIME_CONTAINER}/.venv/bin/python",
                ENV_INTERPRETER_TREES: str(self.interpreter.parent)}

    @property
    def paths(self) -> dict[str, Path]:
        return {"spec": self.root / "spec.yaml", "registration": self.root / "registration.yaml"}

    def render(self, reg_mutate: Callable[[dict], None] | None = None,
               spec_mutate: Callable[[dict], None] | None = None) -> None:
        text = (WHISPER / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", self.item_set_hash).replace("{sampler}", json.dumps(SAMPLER))
        text = text.replace("{evaluator_hash}", sha_of({"evaluator": EVALUATOR_ID, "metric": METRIC,
                                                         "direction": "lower_is_better", "scorer": SCORER}))
        mutate = reg_mutate or self.reg_mutate
        if mutate is not None:
            reg = yaml.safe_load(text)
            mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = (WHISPER / "spec.template.yaml").read_text(encoding="utf-8")
        spec = spec.replace("{registration_sha256}", file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        spec = spec.replace("{base}", str(self.base)).replace("{data}", str(self.data))
        spec = spec.replace("{items}", str(self.items_dir))
        if spec_mutate is not None:
            doc = yaml.safe_load(spec)
            spec_mutate(doc)
            spec = yaml.safe_dump(doc, sort_keys=False)
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    def launch_record(self, **overrides) -> dict[str, Any]:
        """ext-launch's record of the full run (README "Launch request facts", full; gpus true)."""
        rec = {"schema": RECORD_SCHEMA, "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": AREA, "profile": "full", "label": LABEL, "host_work": str(self.host_work),
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID,
                           "venv_lock_sha256": file_sha256(self.runtime / "uv.lock")},
               "framework": {"name": "transformers", "version": "5.21.0", "commit": None},
               "training_seed": SEED, "gpus": True,
               "argv": [f"{RUNTIME_CONTAINER}/.venv/bin/python", f"{RUNTIME_CONTAINER}/train_whisper.py", "--profile",
                        "full", "--out", "/out", "--seed", str(SEED)],
               "mounts": [{"host": str(self.runtime), "container": RUNTIME_CONTAINER, "mode": "ro"},
                          {"host": str(self.stage_out), "container": "/out", "mode": "rw"}],
               "tmpfs": ["/tmp"], "network": "none", "network_reason": None, "user": run_user(),
               "env_names": list(ENV_NAMES), "gpu_est_h": 0.1, "gpu_used_h": 4.43,
               "output_dir": str(self.stage_out), "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": f"{RUNTIME_CONTAINER}/.venv/bin/python",
               "interpreter_sha256": file_sha256(self.interpreter),
               "script_path": f"{RUNTIME_CONTAINER}/train_whisper.py", "script_sha256": file_sha256(self.script),
               "host_config_sha256": host_config_sha256(self.host_work, self.output_root, [self.root],
                                                         interpreter=f"{RUNTIME_CONTAINER}/.venv/bin/python",
                                                         interpreter_trees=[self.interpreter.parent])}
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def rewrite_launch(self, **overrides) -> None:
        rec = self.launch_record(**overrides)
        self.launch.write_bytes(canonical_bytes(rec))
        self.label = rec["label"]
        guard_log(self.host_work, self.label)

    def edit_json(self, path: Path, fn: Callable[[dict], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        write_json(path, doc)

    def edit_logs(self, fn: Callable[[list[dict], list[dict]], None]) -> None:
        samples = read_jsonl(self.stage_out / "samples.jsonl")
        steps = read_jsonl(self.stage_out / "train_steps.jsonl")
        fn(samples, steps)
        write_jsonl(self.stage_out / "samples.jsonl", samples)
        write_jsonl(self.stage_out / "train_steps.jsonl", steps)

    def reseal(self, cut: Path) -> None:
        """Rewrite cut.json's file hashes after a cut file was edited (the cut stays complete)."""
        self.edit_json(cut / "cut.json", lambda d: d.update(files={n: file_sha256(cut / n) for n in CUT_FILES}))

    def collect(self, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, smoke: Path | None = None, timeout: int = 300):
        args = ["ext-collect", str(self.paths["spec"]), "--adapter", ADAPTER, "--stage-out", str(self.stage_out),
                "--launch-record", str(launch or self.launch), "--smoke-record", str(smoke or self.smoke),
                "--eval-baseline", str(baseline or self.baseline), "--eval-candidate", str(candidate or self.candidate),
                "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())

    def fill(self, arm: Path, checkpoint: Path, *, base: Path | None = None):
        args = ["ext-arm-fill", "--adapter", ADAPTER, "--checkpoint", str(checkpoint), "--items", str(self.items_dir)]
        if base is not None:
            args += ["--base", str(base)]
        return run_cli(args + [str(arm)], cwd=self.root, env=self.env())

    def request(self, out: Path, **overrides) -> dict[str, Any]:
        """An ext-launch request for the full training run into `out` (README "Launch request facts")."""
        rec = self.launch_record()
        req = {"area": AREA, "label": LABEL, "host_work": rec["host_work"],
               "runtime": {"kind": "venv_in_container", "image": IMAGE_ID, "venv_lock": str(self.runtime / "uv.lock")},
               "framework": rec["framework"], "training_seed": SEED, "gpus": True, "network": "none",
               "network_reason": None, "user": run_user(),
               "env": {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home", "XDG_CACHE_HOME": "/tmp/cache",
                       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"},
               "mounts": [rec["mounts"][0], {"host": str(out), "container": "/out", "mode": "rw"}],
               "tmpfs": ["/tmp"], "output_dir": str(out), "gpu_est_h": 0.03, "argv": list(rec["argv"])}
        req.update(overrides)
        return req

    def launch_request(self, req: dict, name: str, *, profile: str = "smoke", smoke: Path | None = None,
                       extra: list[str] = ()):
        path = self.root / f"{name}.request.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        out = self.root / f"{name}.launch.json"
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(path), "--gpu-log",
                str(self.host_work / "gpu-log.md"), "--out-record", str(out)]
        if smoke is not None:
            args += ["--smoke-record", str(smoke)]
        return run_cli(args + list(extra), cwd=self.root, env=self.env()), out


def decoder_block_sha256(script: Path) -> str:
    """train_whisper.py / eval_asr.py `decoder_code_sha256`: SHA-256 of the text from `# BEGIN SHARED DECODER` up to
    `# END SHARED DECODER`."""
    text = script.read_text(encoding="utf-8")
    return sha256_text(text[text.index("# BEGIN SHARED DECODER"):text.index("# END SHARED DECODER")].encode("utf-8"))


def steps_per_epoch(n: int, bs: int) -> int:
    return math.ceil(n / bs)
