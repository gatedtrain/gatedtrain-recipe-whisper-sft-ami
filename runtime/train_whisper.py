#!/usr/bin/env python
"""Supervised fine-tuning of Whisper base.en on AMI meeting speech with the Hugging Face transformers
Seq2SeqTrainer, prototype area "audio" (row 5 of the multimodal prototype scope; prototype, unpromotable).

The trainer is transformers' own `Seq2SeqTrainer` (full fine-tune, AdamW, linear schedule, token cross-entropy on
the transcript). This launcher adds, without editing transformers source (MM-27):
  - an observed consumption log: `compute_loss` takes the example ids each micro-batch carries, and a callback writes
    them for the optimizer step in `on_pre_optimizer_step`, before `optimizer.step()` (transformers trainer.py:1914);
  - cuts written atomically with a completion marker: Trainer checkpointing is off (`save_strategy="no"`); a
    callback in `on_step_end` (after the optimizer step and `global_step += 1`, trainer.py:1927-1929) writes
    `.checkpoint-<step>.partial/` and renames it to `checkpoint-<step>/` after `cut.json`, written last;
  - recorded seeds and their derivation, the base and data bindings, an env freeze before torch is imported.

Outputs in --out (the run's only read-write mount):
  env-freeze.txt     `name==version` lines of the runtime, written before torch is imported (MM-33)
  env-runtime.json   interpreter, image CUDA version, lock and script SHA-256 (EXT-83), decoder code SHA-256
  run_args.json      every setting, TrainingArguments, seed and derivation, base binding (repo, revision, file
                     hashes), data binding (train_file and its SHA-256)
  samples.jsonl      observed consumption log: one line per (optimizer step, training example), written before the step
  train_steps.jsonl  one line per optimizer step: the example ids whose gradients it applied, written before the step
  train_log.jsonl    Trainer logs (loss, grad norm, learning rate) per logging step
  memory.jsonl       per cut: seconds, seconds per step, peak device memory, peak RSS
  attempts.jsonl     one line per start and finish of this command (gpu-step.sh reruns after a preemption)
  checkpoint-<S>/    the cut after optimizer step S; `cut.json` written last

Usage (inside the container; README.md has the request shape):
  python train_whisper.py --profile smoke|full --out /out [--base DIR] [--data DIR] [--seed S]
  python train_whisper.py --profile cpu-sanity --device cpu --out DIR     # host CPU check, tiny settings
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_NAME = "whisper-base.en"
DEFAULT_BASE = HERE / "models" / BASE_NAME
DEFAULT_DATA = HERE / "data"
MODELS_JSON = HERE / "models" / "MODELS.json"

COMMON = {
    # Optimization (Whisper fine-tuning recipes: AdamW, linear decay after warm-up, lr 1e-5 for small models;
    # README.md "Training settings"). No dropout (Whisper config: dropout, attention and activation dropout 0) and no
    # SpecAugment (config apply_spec_augment false): the data order is the only randomness.
    "learning_rate": 1.0e-5,
    "lr_scheduler_type": "linear",
    "warmup_steps": 30,
    "weight_decay": 0.0,
    "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1.0e-8,
    "max_grad_norm": 1.0,
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 1,
    "precision": "bf16",               # autocast bf16, fp32 master weights; no loss scaler, so no skipped step
    "gradient_checkpointing": False,
    "dataloader_num_workers": 4,
    "logging_steps": 10,
    # transformers enable_full_determinism (torch.use_deterministic_algorithms, cuDNN deterministic, CUBLAS workspace):
    # on for the CPU profiles, where two runs must give the same cut hash; off on the GPU (README.md "Reproducibility").
    "full_determinism": False,
    "label_text": "AMI text lower-cased, one leading space (Whisper's transcript tokens start with a space)",
}
PROFILES = {
    # smoke: 60 steps (1,920 examples, 0.64 epoch), cuts 30 and 60; measures seconds per step and peak memory.
    "smoke": {"max_steps": 60, "save_steps": 30},
    # full: 300 steps = 9,600 examples = 3.2 epochs of the 3,000-row shard, cuts 100, 200, 300 (README.md "Profiles").
    "full": {"max_steps": 300, "save_steps": 100},
    # host CPU sanity check only: fp32, 4 examples per step, 4 steps, cuts 2 and 4, no workers.
    "cpu-sanity": {"max_steps": 4, "save_steps": 2, "per_device_train_batch_size": 4, "precision": "fp32",
                   "dataloader_num_workers": 0, "warmup_steps": 1, "logging_steps": 1, "full_determinism": True},
    # host CPU pilot for the power plan (README.md "Power"): fp32, 60 steps of 16 examples.
    "cpu-pilot": {"max_steps": 60, "save_steps": 60, "per_device_train_batch_size": 16, "precision": "fp32",
                  "dataloader_num_workers": 0, "warmup_steps": 6, "logging_steps": 5, "full_determinism": True},
}
CUT_MARKER = "cut.json"
WEIGHTS = "model.safetensors"
# Files a cut holds besides cut.json; DEPLOYABLE are the model components (MM-17): weights, model and generation
# config, processor (feature extractor and tokenizer), the deployment decoder. base_ref.json and trainer_state.json
# are references and resume-only information.
DEPLOYABLE = ("model.safetensors", "config.json", "generation_config.json", "processor_config.json",
              "tokenizer.json", "tokenizer_config.json", "decoding.json")
CUT_FILES = DEPLOYABLE + ("base_ref.json", "trainer_state.json")


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
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
# Pre-step (MM-33)
# ---------------------------------------------------------------------------

def write_env_freeze(out: Path) -> dict:
    pkgs = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pkgs[name.lower().replace("_", "-")] = dist.version
    lines = [f"{k}=={pkgs[k]}" for k in sorted(pkgs)]
    (out / "env-freeze.txt").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    lock = HERE / "uv.lock"
    runtime = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "image_cuda_version": os.environ.get("CUDA_VERSION"),
        "uv_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "decoder_code_sha256": decoder_code_sha256(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "cpu_count": os.cpu_count(),
        "env_freeze_sha256": sha256_file(out / "env-freeze.txt"),
    }
    write_json(out / "env-runtime.json", runtime)
    return runtime


def prepare_out(out: Path) -> int:
    """As areas 2 and 3: a non-empty --out is a rerun after a preemption; earlier contents move to
    `preempted-attempt-<k>/` (never deleted). Returns the attempt number."""
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if not p.name.startswith("preempted-attempt-")]
    previous = sorted(p for p in out.iterdir() if p.name.startswith("preempted-attempt-"))
    attempt = len(previous) + 1
    if existing:
        attempt = len(previous) + 2
        dest = out / f"preempted-attempt-{attempt - 1}"
        dest.mkdir()
        for p in existing:
            p.rename(dest / p.name)
        if (dest / "attempts.jsonl").exists():
            shutil.copyfile(dest / "attempts.jsonl", out / "attempts.jsonl")
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "started_at": utc_now(), "argv": sys.argv})
    return attempt


# ---------------------------------------------------------------------------
# Bindings
# ---------------------------------------------------------------------------

def base_binding(base: Path, verify: bool) -> dict:
    """Base model reference (MM-18, MM-21): repo, 40-hex revision, the SHA-256 of every file, from
    models/MODELS.json; with `verify`, every file is re-hashed and must match (0.3 GB)."""
    record = json.loads(MODELS_JSON.read_text(encoding="utf-8"))["models"][BASE_NAME]
    files = record["files"]
    for p in sorted(base.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the base directory must be plain (MM-20)")
    on_disk = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())
    if on_disk != sorted(files):
        raise SystemExit(f"{base} does not hold exactly the files of {MODELS_JSON} [{BASE_NAME}]")
    if verify:
        for rel, want in files.items():
            if sha256_file(base / rel) != want["sha256"]:
                raise SystemExit(f"{base / rel} differs from its SHA-256 in {MODELS_JSON}")
    rev = record["revision"]
    if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
        raise SystemExit(f"base revision {rev!r} is not a 40-hex commit (MM-18)")
    return {"repo": record["repo"], "revision": rev, "local_dir": str(base),
            "files": {k: v["sha256"] for k, v in files.items()},
            "files_sha256": sha256_bytes(canonical({k: v["sha256"] for k, v in files.items()})),
            "verified": verify}


def read_train(data_dir: Path) -> tuple[list[dict], Path]:
    """The training shard; every row's audio file must exist and match its recorded SHA-256 (data binding)."""
    path = data_dir / "train.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    ids = [r["utt_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: utterance ids are not unique")
    for r in rows:
        if sha256_file(data_dir / r["audio"]) != r["audio_sha256"]:
            raise SystemExit(f"{data_dir / r['audio']} differs from its SHA-256 in {path}")
    return rows, path


# ---------------------------------------------------------------------------
# Cuts
# ---------------------------------------------------------------------------

def canonicalize_safetensors_header(path: Path) -> None:
    """safetensors serializes its header from a hash map, so two saves of identical tensors can differ in key order
    (area 3, measured). Rewrite the header as JSON with sorted keys, compact separators and space padding to 8 bytes
    (the format allows trailing spaces); tensor bytes and offsets are unchanged."""
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    text = json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + raw[8 + n:])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    os.chmod(path, 0o644)


def save_cut(model, processor, state, out: Path, step: int, examples_consumed: int, base_ref: dict,
             extra: dict) -> Path:
    """checkpoint-<step>/ via .checkpoint-<step>.partial/ and an atomic rename: weights and configs
    (`save_pretrained`, safetensors), processor (feature extractor and tokenizer), decoding.json (the deployment
    decoder), base_ref.json, trainer_state.json (TrainerState.save_to_json, resume-only information), then cut.json
    last with the SHA-256 of every other file."""
    cut = out / f"checkpoint-{step}"
    if cut.exists():
        return cut
    tmp = out / f".checkpoint-{step}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    # The Trainer turns the KV cache off while training; the deployable config keeps the base's `use_cache: true`.
    use_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        model.save_pretrained(str(tmp), safe_serialization=True)
    finally:
        model.config.use_cache = use_cache
    processor.save_pretrained(str(tmp))
    canonicalize_safetensors_header(tmp / WEIGHTS)
    write_json(tmp / "decoding.json", decoding_doc())
    write_json(tmp / "base_ref.json", {k: base_ref[k] for k in ("repo", "revision", "files", "files_sha256")})
    state.save_to_json(str(tmp / "trainer_state.json"))
    present = sorted(p.name for p in tmp.iterdir())
    if present != sorted(CUT_FILES):
        raise RuntimeError(f"cut {step} holds {present}, not {sorted(CUT_FILES)}")
    files = {name: sha256_file(tmp / name) for name in CUT_FILES}
    write_json(tmp / CUT_MARKER, {
        "schema": "gatedtrain_mm.whisper_sft_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "global_step": step, "examples_consumed": examples_consumed, "files": files,
        "deployable": list(DEPLOYABLE), **extra,      # no wall-clock field: the cut hash is reproducible
    })
    for p in tmp.iterdir():
        fd = os.open(p, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    os.replace(tmp, cut)
    return cut


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, help="override the profile's step count (the last step is a cut)")
    p.add_argument("--no-verify-base", action="store_true", help="skip re-hashing the base files (host checks)")
    return p.parse_args(argv)


def resolve_settings(args) -> dict:
    s = {**COMMON, **PROFILES[args.profile]}
    if args.max_steps is not None:
        s["max_steps"] = args.max_steps
    s["seed"] = args.seed
    return s


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    attempt = prepare_out(out)
    runtime = write_env_freeze(out)                 # before torch is imported (MM-33)
    s = resolve_settings(args)
    t_start = time.time()
    base = args.base.resolve()
    data_dir = args.data.resolve()
    base_ref = base_binding(base, verify=not args.no_verify_base)
    rows, train_file = read_train(data_dir)

    import numpy as np
    import torch
    import transformers
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback,
                              WhisperForConditionalGeneration, WhisperProcessor)

    device = torch.device(args.device)
    processor = WhisperProcessor.from_pretrained(str(base), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(base), torch_dtype=torch.float32,
                                                            local_files_only=True)
    tok = processor.tokenizer
    decoder_start = model.config.decoder_start_token_id

    class Utterances(torch.utils.data.Dataset):
        """One row per training utterance: log-mel features, label ids, and the example id (utt_id)."""

        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            audio = load_audio(data_dir / r["audio"], r["audio_sha256"])
            feats = processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
            labels = tok(" " + r["text"].lower()).input_ids
            return {"input_features": feats["input_features"][0], "labels": labels, "example_id": r["utt_id"]}

    def collate(batch):
        feats = torch.as_tensor(np.stack([b["input_features"] for b in batch]))
        width = max(len(b["labels"]) for b in batch)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        for k, b in enumerate(batch):
            labels[k, :len(b["labels"])] = torch.as_tensor(b["labels"])
        # The tokenizer prefixes <|startoftranscript|><|notimestamps|>; the model prepends the decoder start token
        # itself when it shifts the labels right, so the leading start token is dropped here (as the transformers
        # Whisper fine-tuning collator does).
        if bool((labels[:, 0] == decoder_start).all()):
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels, "example_ids": [b["example_id"] for b in batch]}

    pending: list[list[str]] = []                   # example ids of the micro-batches since the last optimizer step

    class LoggingTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            ids = inputs.pop("example_ids", None)
            if model.training and ids is not None:
                pending.append(list(ids))
            return super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    hf_out = out / "hf-trainer"
    targs = Seq2SeqTrainingArguments(
        output_dir=str(hf_out), max_steps=s["max_steps"], per_device_train_batch_size=s["per_device_train_batch_size"],
        gradient_accumulation_steps=s["gradient_accumulation_steps"], learning_rate=s["learning_rate"],
        lr_scheduler_type=s["lr_scheduler_type"], warmup_steps=s["warmup_steps"], weight_decay=s["weight_decay"],
        adam_beta1=s["adam_beta1"], adam_beta2=s["adam_beta2"], adam_epsilon=s["adam_epsilon"],
        max_grad_norm=s["max_grad_norm"], bf16=(s["precision"] == "bf16"), fp16=False,
        gradient_checkpointing=s["gradient_checkpointing"], seed=s["seed"], data_seed=s["seed"],
        full_determinism=s["full_determinism"],
        save_strategy="no", eval_strategy="no", logging_steps=s["logging_steps"], report_to="none",
        remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=s["dataloader_num_workers"],
        dataloader_drop_last=False, disable_tqdm=True, use_cpu=(device.type == "cpu"), predict_with_generate=False,
        optim="adamw_torch")
    decoding = decoding_doc()
    run_args = {
        "schema": "gatedtrain_mm.whisper_sft_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "profile": args.profile, "dry_run": False, "attempt": attempt, "settings": s, "max_steps": s["max_steps"],
        "seed": s["seed"],
        "seed_derivation": ("Seq2SeqTrainingArguments(seed=S, data_seed=S): transformers set_seed(S) when the Trainer "
                            "is built (trainer.py:419); the training order is accelerate's "
                            "seedable sampler seeded with data_seed (trainer.py:836, DataLoaderConfiguration.data_seed) "
                            "and the epoch; dropout and SpecAugment are off, so the order is the only randomness. The "
                            "order actually used is the observed log, not a reconstruction."),
        "model_name_or_path": str(base), "base_model": str(base),
        "base_ref": {k: base_ref[k] for k in ("repo", "revision", "files_sha256", "verified")},
        "train_file": str(train_file), "train_file_sha256": sha256_file(train_file), "n_train": len(rows),
        "decoding": decoding, "decoding_sha256": sha256_bytes(canonical(decoding)),
        "consumption_log": "samples.jsonl", "step_log": "train_steps.jsonl",
        "consumption_semantics": ("samples.jsonl: one line per (optimizer step, training example) whose loss entered "
                                  "that step's gradient, written in on_pre_optimizer_step before optimizer.step(); "
                                  "train_steps.jsonl: one line per optimizer step with its example ids. bf16 autocast "
                                  "has no loss scaler, so every logged step applied its update."),
        "training_args": json.loads(targs.to_json_string()),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
        "decoder_code_sha256": runtime["decoder_code_sha256"], "script_sha256": runtime["script_sha256"],
        "created_at": utc_now(),
    }
    write_json(out / "run_args.json", run_args)

    counters = {"epoch": -1, "consumed": 0, "t_last": time.time(), "step_last": 0}

    class ConsumptionAndCuts(TrainerCallback):
        def on_epoch_begin(self, targs_, state, control, **kw):
            counters["epoch"] += 1

        def on_pre_optimizer_step(self, targs_, state, control, **kw):
            step = state.global_step + 1               # the step this optimizer.step() completes
            ids = [i for mb in pending for i in mb]
            now = utc_now()
            for mb_index, mb in enumerate(pending):
                for pos, utt in enumerate(mb):
                    append_jsonl(out / "samples.jsonl", {
                        "step": step, "epoch": counters["epoch"], "micro_batch": mb_index, "position": pos,
                        "utt_id": utt, "row": sha256_bytes(utt.encode("utf-8")), "attempt": attempt, "utc": now})
            append_jsonl(out / "train_steps.jsonl", {"step": step, "epoch": counters["epoch"], "n_examples": len(ids),
                                                     "example_ids": ids, "utc": now})
            counters["consumed"] += len(ids)
            pending.clear()

        def on_log(self, targs_, state, control, logs=None, **kw):
            append_jsonl(out / "train_log.jsonl", {"step": state.global_step, **(logs or {})})

        def on_step_end(self, targs_, state, control, model=None, **kw):
            step = state.global_step
            if step % s["save_steps"] and step != s["max_steps"]:
                return
            save_cut(model, processor, state, out, step, counters["consumed"], base_ref,
                     {"script_sha256": runtime["script_sha256"], "decoder_code_sha256": runtime["decoder_code_sha256"],
                      "seed": s["seed"], "profile": args.profile})
            now = time.time()
            mem = {"step": step, "seconds_since_last_cut": round(now - counters["t_last"], 2),
                   "seconds_per_step": round((now - counters["t_last"]) / max(1, step - counters["step_last"]), 3),
                   "examples_consumed": counters["consumed"], "epoch": counters["epoch"],
                   "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                   "utc": utc_now()}
            if device.type == "cuda":
                mem["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
                mem["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
            append_jsonl(out / "memory.jsonl", mem)
            print(json.dumps(mem, sort_keys=True), flush=True)
            counters["t_last"], counters["step_last"] = now, step

    trainer = LoggingTrainer(model=model, args=targs, train_dataset=Utterances(), data_collator=collate,
                             processing_class=processor.feature_extractor, callbacks=[ConsumptionAndCuts()])
    trainer.train()
    if pending:
        raise RuntimeError(f"{sum(len(mb) for mb in pending)} examples were loaded after the last optimizer step")
    if hf_out.exists() and not any(hf_out.iterdir()):
        hf_out.rmdir()
    append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "finished_at": utc_now(), "max_steps": s["max_steps"],
                                          "global_step": trainer.state.global_step,
                                          "examples_consumed": counters["consumed"],
                                          "wall_seconds": round(time.time() - t_start, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
