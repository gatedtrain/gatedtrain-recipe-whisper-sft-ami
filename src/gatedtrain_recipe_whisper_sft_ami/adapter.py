"""HF Seq2SeqTrainer full-parameter SFT adapter (area "audio": Whisper base.en on AMI IHM with transformers 5.21.0;
MULTIMODAL-SPEC.md section 6 row 5 and the maintainers' area 4 decisions (not published); tests/README.md
section 11.21,
EXT-138 to EXT-153).

MM-12: a pure-Python class that imports no framework, torch or pickle loader. It reads JSON, JSONL, the text
`env-freeze.txt` and the safetensors header (standard library `struct` and `json`), and hashes everything else.

Run layout, read from the area runtime's code `runtime/train_whisper.py` and
`eval_asr.py` (and its README.md "Trainer", "Cut layout", "Outputs of `train_whisper.py`", "What gatedtrain needs";
read 2026-11-13; where the two differ the code is followed, docs/findings.md):

- `save_cut` (train_whisper.py:325-364), called in `on_step_end` at every `save_steps` and at `max_steps` (:516-522),
  writes `checkpoint-<S>/` (S the optimizer step) into `.checkpoint-<S>.partial/` and renames it: `model.safetensors`
  (fp32, header canonicalized), `config.json`, `generation_config.json`, `processor_config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `decoding.json` (the deployment decoder), `base_ref.json` (repo, revision, per-file SHA-256
  of the base, `files_sha256`), `trainer_state.json` (resume only), then `cut.json` last: `global_step`,
  `examples_consumed`, `files` (SHA-256 of the nine), `deployable` (`DEPLOYABLE`, :92-93), `script_sha256`,
  `decoder_code_sha256`, `seed`, `profile`. Trainer checkpointing is off; the Trainer's `hf-trainer/` output directory
  is removed when empty and is never a cut;
- `samples.jsonl` (:503-507): one line per (optimizer step, example) in `on_pre_optimizer_step`, before the step:
  `{step, epoch, micro_batch, position, utt_id, row = sha256(utt_id), attempt, utc}`; `train_steps.jsonl` (:508-509):
  `{step, epoch, n_examples, example_ids, utc}`. The epoch counter starts at 0 (`on_epoch_begin` increments -1); with
  `gradient_accumulation_steps` 1 and `dataloader_drop_last` false an epoch of N rows at batch B is ceil(N / B) steps,
  each of B examples except the last (N - (ceil(N / B) - 1) B), every row once. Nothing is filtered;
- `run_args.json` (:467-491): `profile`, `dry_run`, `attempt`, `settings`, `max_steps`, `seed`, `model_name_or_path` =
  `base_model` (the container path of the base directory), `base_ref` (`repo`, `revision`, `files_sha256`,
  `verified`), `train_file`, `train_file_sha256`, `n_train`, `decoding`, `decoding_sha256` (compact JSON),
  `decoder_code_sha256`, `script_sha256`; `env-runtime.json` (:218-240) `script_sha256` (EXT-83) and
  `decoder_code_sha256`; `env-freeze.txt` `name==version` lines.

Identity (MM-17): full weights, no LoRA. The spec's base is the plain whisper-base.en directory; its model hash is the
manifest hash of all its files (the baseline arm evaluates the base). A cut's model hash is the manifest hash of its
`deployable` files; no composition with the base (MM-18 does not apply).
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from gatedtrain.canonical import entries_hash, file_sha256, manifest_entries, manifest_hash, sha256_bytes, text_lines
from gatedtrain.errors import Problem
from gatedtrain.ext.registry import EVALUATOR_ROWS, TRAINER_ROWS
from gatedtrain.ext.sdk import (compact_sha256, is_int as _int, plain_file as _plain_file, read_cut_marker, read_json as _read_json,
                   read_jsonl, read_run_args, runtime_argv_problems, runtime_script_problems, safetensors_problem,
                   sha_or_none as _sha_or_none, single_shard as _shard)

NAME = "hf_seq2seq"
LIBRARY = ("transformers Seq2SeqTrainer with the area runtime's cut writer (train_whisper.py save_cut, Trainer "
           "checkpointing off, cut.json written last)")
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")        # train_whisper.py save_cut; `.checkpoint-<S>.partial` never
CUT_MARKER = "cut.json"
WEIGHTS = "model.safetensors"
COMPONENTS = tuple(TRAINER_ROWS[NAME]["components"])          # train_whisper.py DEPLOYABLE (EXT-140)
BASE_REF = "base_ref.json"
DECODING = "decoding.json"
CUT_FILES = COMPONENTS + (BASE_REF, "trainer_state.json")      # train_whisper.py CUT_FILES
FRAMEWORK_PACKAGE = "transformers"
PINS = tuple(TRAINER_ROWS[NAME]["pins"])
FREEZE_FILE = "env-freeze.txt"
RUNTIME_FILE = "env-runtime.json"
SAMPLES = "samples.jsonl"
STEPS = "train_steps.jsonl"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
# The plain whisper-base.en directory (README "Data", MODELS.json: 11 files); these make it a model directory.
BASE_REQUIRED = ("config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json")
# The SHARED DECODER block of train_whisper.py and eval_asr.py (byte-identical; README "Deployment decoder"): the
# evaluator's decoder code, which every cut's `decoder_code_sha256` must name (README item 5; EXT-145, EXT-153).
DECODER_CODE_SHA256 = "sha256:478dd30eae25241a8e63b4437a89f1f8a064a1fccb77990cdef1a4fc6088cc5f"
# eval_asr.py `sampler` = the deployment decoder plus these evaluation keys (EXT-145).
EVAL_ONLY_KEYS = ("batch_size", "dtype", "order")
TRAIN_SCRIPT = "train_whisper.py"
EVAL_SCRIPT = "eval_asr.py"
DEFAULT_BASE_REL = "models/whisper-base.en"                    # both scripts: DEFAULT_BASE = HERE / this
PROFILES = ("smoke", "full")
# argparse options of the two scripts (parse_args), for the launch-side argv check (EXT-150): option -> takes a value.
# argparse accepts a unique prefix of an option (allow_abbrev), so a token is resolved the same way.
TRAIN_OPTIONS = {"--profile": True, "--out": True, "--base": True, "--data": True, "--seed": True, "--device": True,
                 "--max-steps": True, "--no-verify-base": False}
EVAL_OPTIONS = {"--checkpoint": True, "--base-only": False, "--base": True, "--items": True, "--out": True,
                "--eval-seeds": True, "--model-hash": True, "--base-model-hash": True, "--item-set-hash": True,
                "--limit": True, "--batch-size": True, "--device": True, "--dtype": True}

CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    "rule": ("a checkpoint-<S> directory is a cut iff its cut.json, written last after an atomic rename from "
             ".checkpoint-<S>.partial, parses with global_step = S and a positive integer examples_consumed, a "
             "deployable list of the seven model components (weights, model and generation config, processor config, "
             "tokenizer files, decoding.json), and files listing exactly the other files of the directory (those seven, "
             "base_ref.json and trainer_state.json), each a plain file with its SHA-256; model.safetensors is whole; "
             ".partial directories and the Trainer's hf-trainer/ are never cuts; the cut's last step included is S"),
    "complete_dir_pattern": CHECKPOINT_DIR_RE.pattern,
    "sources": ["runtime/train_whisper.py:325-364 save_cut (cut.json last, "
                "os.replace)",
                "runtime/train_whisper.py:92-94 DEPLOYABLE, CUT_FILES",
                "runtime/train_whisper.py:516-522 on_step_end (save_steps, max_steps)",
                "runtime/README.md \"Cut layout\""],
    "verified": True,                     # the layout was read from the CPU sanity run's cuts (README, sanity-evidence)
}
SEED_DERIVATION = ("Seq2SeqTrainingArguments(seed=S, data_seed=S) (train_whisper.py): transformers set_seed(S) when the "
                   "Trainer is built and accelerate's seedable sampler seeded with data_seed and the epoch; dropout and "
                   "SpecAugment are off, so the training order is the only randomness, and the order used is the "
                   "observed log (samples.jsonl, train_steps.jsonl), never a reconstruction. The training seed is "
                   "run_args.json `seed` (= settings.seed), which must equal the launch record's training_seed and "
                   "every cut's cut.json `seed`. Evaluation is greedy (K = 1; eval seed 11 is recorded, nothing is "
                   "drawn); 05 section 14 H(run_seed, step, data_ref, sample_index) is not claimed.")


class HfSeq2SeqAdapter:
    """The MM-12 interface for HF Seq2SeqTrainer full-parameter SFT runs of the area runtime (EXT-138 to EXT-153)."""

    name = NAME
    trainer_id = TRAINER_ROWS[NAME]["uses"][0]
    evaluator_ids = tuple(TRAINER_ROWS[NAME]["evaluators"])
    framework_name = FRAMEWORK_PACKAGE
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(TRAINER_ROWS[NAME]["spec_binding"])
    # ext-collect message (EXT-143).
    consumption_fix = "collect the run's own samples.jsonl and train_steps.jsonl as the run wrote them (MM-23, MM-27)"

    def __init__(self) -> None:
        self._base_cache: dict[str, list[dict]] = {}

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    # --- the base directory and cuts (MM-17, MM-20; EXT-139, EXT-140) ---

    @staticmethod
    def is_cut(d: Path) -> bool:
        """Whether `d` is (meant to be) a cut: it holds cut.json. The plain base directory never does."""
        return (Path(d) / CUT_MARKER).exists() or (Path(d) / CUT_MARKER).is_symlink()

    @staticmethod
    def base_dir_problem(d: Path) -> str | None:
        """None when `d` is a plain whisper-base.en directory: its configs (model, generation, feature extractor,
        tokenizer) and whole safetensors weights."""
        for rel in BASE_REQUIRED:
            if not _plain_file(Path(d) / rel):
                return f"{rel} is missing (not a Whisper model directory)"
        return safetensors_problem(Path(d) / WEIGHTS)

    def _cut_files_problem(self, d: Path) -> str | None:
        """Review 9 M1 (EXT-154): a file that cannot be read makes the directory incomplete, never an exception."""
        try:
            return self._cut_files_problem_unguarded(d)
        except OSError as exc:
            return (f"{Path(exc.filename).name if exc.filename else Path(d).name} cannot be read "
                    f"({exc.__class__.__name__}: {exc.strerror or exc})")

    def _cut_files_problem_unguarded(self, d: Path) -> str | None:
        doc, why = read_cut_marker(d)
        if doc is None:
            return why
        files = doc["files"]
        if sorted(files) != sorted(CUT_FILES):
            return f"{CUT_MARKER} lists {sorted(files)}, not {sorted(CUT_FILES)}"
        present = sorted(p.name for p in Path(d).iterdir())
        if present != sorted([*CUT_FILES, CUT_MARKER]):
            extra = sorted(set(present) - set(CUT_FILES) - {CUT_MARKER})
            return f"the directory holds {extra or present}, which {CUT_MARKER} does not list"
        for name in CUT_FILES:
            if not _plain_file(Path(d) / name):
                return f"{name} is not a plain file"
            if file_sha256(Path(d) / name) != files[name]:
                return f"{name} is not the SHA-256 {CUT_MARKER} records"
        if not (isinstance(doc.get("deployable"), list) and all(isinstance(x, str) for x in doc["deployable"])
                and sorted(doc["deployable"]) == sorted(COMPONENTS)):          # review 11 L1
            return f"{CUT_MARKER} deployable is {doc.get('deployable')!r}, not {list(COMPONENTS)}"
        for key in ("global_step", "examples_consumed"):
            if not (_int(doc.get(key)) and doc[key] >= 1):
                return f"{CUT_MARKER} {key} is {doc.get(key)!r}, not a positive integer"
        return safetensors_problem(Path(d) / WEIGHTS)

    def model_dir_problem(self, d: Path) -> str | None:
        """None when `d` is a complete cut (it holds cut.json) or a plain whisper-base.en directory."""
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        if self.is_cut(d):
            return self._cut_files_problem(d)
        return self.base_dir_problem(d)

    def cut_problem(self, d: Path, step: int) -> str | None:
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        bad = self._cut_files_problem(d)
        if bad:
            return bad
        doc, _ = read_cut_marker(d)
        if doc.get("global_step") != step:
            return f"{CUT_MARKER} global_step is {doc.get('global_step')!r}, not {step}"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        """Every top-level `checkpoint-<S>` directory, complete or not, in step order; `.checkpoint-<S>.partial` and
        `hf-trainer/` are never one."""
        stage_out = Path(stage_out)
        if not stage_out.is_dir():
            return []
        out = []
        for p in stage_out.iterdir():
            m = CHECKPOINT_DIR_RE.match(p.name)
            if m and p.is_dir() and not p.is_symlink():
                out.append((int(m.group(1)), p))
        return sorted(out)

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        """(complete [(global_step, path)] in step order, incomplete [path]): a cut's last step included is its
        optimizer step (05 section 12.4), the ledger's `step` (EXT-139)."""
        complete, incomplete = [], []
        for step, d in self.checkpoint_dirs(stage_out):
            if self.cut_problem(d, step) is None:
                complete.append((step, d))
            else:
                incomplete.append(d)
        return complete, incomplete

    def _base_entries(self, base: Path) -> list[dict]:
        """The base directory's manifest entries, walked once per adapter instance (290 MB on the real base)."""
        key = str(Path(base).resolve())
        if key not in self._base_cache:
            self._base_cache[key] = manifest_entries(Path(base))
        return self._base_cache[key]

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """EXT-140 (MM-17). The base alone: `model_hash` = `cut_hash` = its manifest hash (every file). A cut:
        `model_hash` over its deployable files, `cut_hash` over the whole cut; full weights, so `base` is not used and
        `base_ref` is null."""
        is_cut = self.is_cut(cut)
        if entries is None and not is_cut:                # the base is walked once per adapter instance
            entries = self._base_entries(Path(cut))
        if entries is None:
            entries = manifest_entries(Path(cut))
        if not is_cut:
            self._base_cache.setdefault(str(Path(cut).resolve()), entries)
            h = entries_hash(entries)
            return {"model_hash": h, "cut_hash": h, "components": [e["path"] for e in entries], "base_ref": None}
        comps = [e for e in entries if e["path"] in COMPONENTS]
        return {"model_hash": entries_hash(comps), "cut_hash": entries_hash(entries),
                "components": [e["path"] for e in comps], "base_ref": None}

    # --- consumption (MM-23, MM-27; EXT-143) ---

    def _ledger_plan(self, stage_out: Path, data: Path, cuts: list) -> tuple[list[dict], list[str]]:
        """(the samples.jsonl lines of the steps up to the last cut, in file order) and the problems that refuse the log
        (empty when it is consistent). Every field train_whisper.py derives is re-derived (README item 6)."""
        args, why = read_run_args(stage_out)
        if args is None:
            return [], [why]
        if not cuts:
            return [], ["no complete cut"]
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        bs, ga, attempt = s.get("per_device_train_batch_size"), s.get("gradient_accumulation_steps"), args.get("attempt")
        if not (_int(bs) and bs > 0):
            return [], [f"run_args.json settings.per_device_train_batch_size is {bs!r}, not a positive integer"]
        if ga != 1:
            return [], [f"run_args.json settings.gradient_accumulation_steps is {ga!r}: the per-step rule (one "
                        "micro-batch per optimizer step) holds for 1 only, the runtime's setting"]
        shard = _shard(data)
        if shard is None:
            return [], [f"the spec's training data {data} holds {len(list(Path(data).glob('*.jsonl')))} shards, not the "
                        "one shard the run read"]
        rows, why = read_jsonl(Path(data) / shard)
        if rows is None:
            return [], [f"the training shard: {why}"]
        ids = [r.get("utt_id") for r in rows]
        if not ids or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
            return [], [f"the training shard {shard} has no unique non-empty string utt_id per row"]
        id_set, n = set(ids), len(ids)
        spe = math.ceil(n / bs)                       # steps per epoch (dataloader_drop_last false)
        rest = n - (spe - 1) * bs                     # examples of an epoch's last step
        marks = []
        for _, d in cuts:
            doc, why = read_cut_marker(d)
            if doc is None:
                return [], [f"{Path(d).name}: {why}"]
            marks.append((doc.get("global_step"), doc.get("examples_consumed"), Path(d).name))
        last = max(m[0] for m in marks)
        samples, why = read_jsonl(Path(stage_out) / SAMPLES)
        if samples is None:
            return [], [why]
        steps, why = read_jsonl(Path(stage_out) / STEPS)
        if steps is None:
            return [], [why]
        out: list[str] = []

        def bad(msg: str) -> bool:
            out.append(msg)
            return len(out) >= 8

        # train_steps.jsonl (train_whisper.py:508-509)
        nums = [st.get("step") for st in steps]
        if not all(_int(k) for k in nums) or nums != list(range(1, len(steps) + 1)):     # review 10 L3: true == 1
            return [], [f"{STEPS} steps {nums[:10]}{' ...' if len(nums) > 10 else ''} are not consecutive from 1"]
        per_step: dict[int, list[str]] = {}
        seen: dict[int, set[str]] = {}
        for st in steps:
            k, e, cnt, xs = st["step"], st.get("epoch"), st.get("n_examples"), st.get("example_ids")
            if not (_int(e) and isinstance(xs, list) and all(isinstance(x, str) for x in xs)):
                if bad(f"{STEPS} step {k} has no integer epoch and list of string example_ids"):
                    return [], out
                continue
            per_step[k] = xs
            want_e = (k - 1) // spe
            if e != want_e:
                if bad(f"{STEPS} step {k} is in epoch {e}, not {want_e} ({spe} steps per epoch: {n} rows at batch "
                       f"{bs})"):
                    return [], out
            want_n = rest if k % spe == 0 else bs
            if len(xs) != want_n:
                if bad(f"{STEPS} step {k} applies {len(xs)} examples, not {want_n} (per_device_train_batch_size {bs}; "
                       f"the last step of each epoch carries the rest, {rest})"):
                    return [], out
            if cnt != len(xs):
                if bad(f"{STEPS} step {k} n_examples {cnt!r} is not the number of its example_ids ({len(xs)})"):
                    return [], out
            for x in xs:
                if x not in id_set:
                    if bad(f"{STEPS} step {k} applies {x[:60]!r}, which is not a utt_id of the training shard {shard}"):
                        return [], out
                elif x in seen.setdefault(e if _int(e) else -1, set()):
                    if bad(f"{STEPS} step {k} applies {x[:60]!r} twice in epoch {e} (an epoch reads every row once)"):
                        return [], out
                else:
                    seen[e].add(x)
        if out:
            return [], out
        # samples.jsonl (train_whisper.py:503-507): the lines of a step are its example_ids in order
        by_step: dict[int, list[tuple[int, dict]]] = {}
        for n_line, x in enumerate(samples, start=1):
            k = x.get("step")
            if not (_int(k) and k in per_step):
                if bad(f"{SAMPLES} line {n_line}: step {k!r} is not a step {STEPS} logs"):
                    return [], out
                continue
            by_step.setdefault(k, []).append((n_line, x))
        for k, xs in per_step.items():
            lines = by_step.get(k, [])
            if [x.get("utt_id") for _, x in lines] != xs:
                if bad(f"{SAMPLES} has {len(lines)} lines for step {k} whose utt_ids are not the step's {STEPS} "
                       "example_ids in order"):
                    return [], out
                continue
            for pos, (n_line, x) in enumerate(lines):
                where = f"{SAMPLES} line {n_line} (step {k})"
                if x.get("epoch") != (k - 1) // spe:
                    bad(f"{where}: epoch {x.get('epoch')!r}, not {(k - 1) // spe}")
                if x.get("micro_batch") != 0:
                    bad(f"{where}: micro_batch {x.get('micro_batch')!r} is not 0 (one micro-batch per step)")
                if x.get("position") != pos:
                    bad(f"{where}: position {x.get('position')!r} is not {pos}, its place in the step's example_ids")
                if x.get("row") != sha256_bytes(x["utt_id"].encode("utf-8")):
                    bad(f"{where}: row is not sha256(utt_id)")
                if x.get("attempt") != attempt:
                    bad(f"{where}: attempt {x.get('attempt')!r} is not run_args.json attempt {attempt!r}")
                if len(out) >= 8:
                    return [], out
        if out:
            return [], out
        # EXT-158 (review 9 L2): train_whisper.py stops at max_steps and saves its last cut there, so a longer (or
        # shorter) log is another run's or an edited one.
        max_steps = args.get("max_steps")
        if not (_int(max_steps) and len(steps) == max_steps):
            bad(f"{STEPS} logs {len(steps)} steps, not run_args.json max_steps {max_steps!r} (the Trainer stops at "
                "max_steps)")
        # MM-27: each cut's examples_consumed equals the log up to its global step
        for gs, consumed, name in marks:
            if gs > len(steps):
                bad(f"{name}: global_step {gs} is beyond the {len(steps)} steps {STEPS} logs")
                continue
            want = sum(len(per_step[k]) for k in range(1, gs + 1))
            if consumed != want:
                bad(f"{name} cut.json examples_consumed is {consumed!r}, but the steps up to {gs} applied {want} "
                    "examples")
        if out:
            return [], out
        return [x for x in samples if x["step"] <= last], []

    def arm_problems(self, arm: Path, records: list[dict], item_rows: list[dict]) -> list[str]:
        """EXT-155 (review 9 L1): every output an arm references (`outputs/<item>.json`, eval_asr.py) names its item
        and the registered row's `reference_normalized`. eval_asr.py writes the reference of the `--items` it read, so an
        evaluation of another item set (edited transcripts) shows here; edited audio does not (the outputs carry no
        audio hash), which the launch-side `--items` rule covers (EXT-156)."""
        ref_of = {r.get("item"): r.get("reference_normalized") for r in item_rows}
        out: list[str] = []
        for rec in records:
            ref, item = rec.get("output_ref"), rec.get("item")
            if not (isinstance(ref, dict) and isinstance(ref.get("path"), str)) or item not in ref_of:
                continue                                  # no output, or an unknown item (the gate reports it)
            doc, why = _read_json(Path(arm) / ref["path"])
            if not isinstance(doc, dict):
                out.append(f"output {ref['path']}: {why or 'not a JSON object'}")
            elif doc.get("item") != item:
                out.append(f"output {ref['path']} names item {doc.get('item')!r}, not {item!r}")
            elif doc.get("reference_normalized") != ref_of[item]:
                out.append(f"output {ref['path']} reference_normalized is not the registered item's: the evaluation "
                           "read another item set than the registered one (EXT-155)")
            if len(out) >= 8:
                break
        return out

    def consumption_problems(self, stage_out: Path, data: Path, cuts: list) -> list[str]:
        """EXT-143: why the observed logs cannot give the per-example consumption up to the last cut."""
        return self._ledger_plan(stage_out, data, cuts)[1]

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `row`, derivation `observed` (MM-23 with sample log): one ledger record per samples.jsonl line of
        the steps up to the last cut; `data_ref.row` = the line's `row` = sha256(utt_id), shard = the spec's one shard,
        `sample_id` = H(run, attempt, epoch, step, utt_id) (NUL-joined text); `rewards`, `seed_inputs` and
        `sampler_config_hash` null (supervised: no reward, no per-example randomness, no sampler)."""
        from gatedtrain.ledger import WRITER_ID, check_precision

        plan, bad = self._ledger_plan(stage_out, data, cuts)
        if bad:                                          # collect refuses these before writing
            raise ValueError("; ".join(bad))
        check_precision("row", "observed")
        shard = _shard(data)
        snapshot = manifest_hash(data)
        rows = []
        for x in plan:
            key = "\x00".join(str(v) for v in (run_id, attempt_id, x["epoch"], x["step"], x["utt_id"]))
            rows.append({
                "run_id": run_id, "attempt_id": attempt_id, "step": x["step"],
                "sample_id": "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "data_ref": {"snapshot": snapshot, "shard": shard, "row": x["row"], "mixture": None, "component": None},
                "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None, "gen_version": None,
                "reward_epoch": None, "rewards": None, "sampler_config_hash": None, "seed_inputs": None, "chunk": None,
                "precision": "row", "derivation": "observed", "writer_id": WRITER_ID,
            })
        last = max(step for step, _ in cuts)
        return {"precision": "row", "rows": rows, "derivation": "observed",
                "reason": (f"{SAMPLES} logs every example of each optimizer step before the step and {STEPS} the step's "
                           f"example ids (MM-23, MM-27): {len(rows)} records of steps 1 .. {last}, data_ref.row = "
                           f"sha256(utt_id) in shard {shard}, derivation observed; no example is filtered")}

    # --- pins and checks (MM-21, MM-29, MM-33; EXT-141, EXT-142, EXT-145) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """`transformers==<v>` in env-freeze.txt equals the launch record's framework; the freeze pins accelerate and
        torch (MM-33). Raises ValueError with the reason."""
        freeze = Path(stage_out) / FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        versions: dict[str, str] = {}
        for line in text_lines(freeze.read_text(encoding="utf-8")):
            line = line.strip()
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                versions[name.strip().lower().replace("_", "-")] = version.strip()
        missing = [p for p in PINS if p not in versions]
        if missing:
            raise ValueError(f"{freeze} has no `<name>==<version>` line for {', '.join(missing)} (pins of {NAME})")
        version = versions[FRAMEWORK_PACKAGE]
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if fw.get("name") != FRAMEWORK_PACKAGE or fw.get("version") != version:
            raise ValueError(f"{freeze} pins {FRAMEWORK_PACKAGE}=={version}, but the launch record claims "
                             f"framework {fw.get('name')!r} {fw.get('version')!r}")
        return {"name": FRAMEWORK_PACKAGE, "version": version, "freeze_sha256": file_sha256(freeze)}

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        """EXT-83: env-runtime.json `script_sha256` equals the record's (train_whisper.py writes it as area 1 does)."""
        return runtime_script_problems(stage_out, launch)

    BASE_FIX = ("train from the spec's whisper-base.en directory inside a read-only mount (train_whisper.py --base, "
                "default <area4-whisper>/models/whisper-base.en, re-hashed at start), and record that mount in the "
                "launch record")
    DATA_FIX = ("train on the spec's data directory inside a read-only mount (train_whisper.py --data, its train.jsonl "
                "recorded in run_args.json train_file, train_file_sha256 and n_train)")

    def argv_problems(self, argv: Any, *, profile: str | None = None) -> list[str]:
        """EXT-150, EXT-156 (review 9 L1), ext-launch and --recheck. argv[1] is train_whisper.py or eval_asr.py of the
        runtime directory (the pinned interpreter's). eval_asr.py loads `--base` (with --base-only) and `--items`
        without hashing them, so an evaluation may use only the defaults `<runtime>/models/whisper-base.en` and
        `<runtime>/items` inside the read-only runtime mount (as ddpo_sd15, EXT-133); `--limit` is refused under
        `--profile full`. train_whisper.py: `--no-verify-base` is refused (collect refuses a run that did not re-hash
        its base, EXT-141) and `--profile` is smoke or full (the CPU profiles are host checks, never collected).
        Options are resolved as argparse resolves them (a unique prefix counts)."""
        out, script, parsed = runtime_argv_problems(argv, profile=profile,
                                                    options_of={TRAIN_SCRIPT: TRAIN_OPTIONS, EVAL_SCRIPT: EVAL_OPTIONS},
                                                    eval_script=EVAL_SCRIPT, default_base_rel=DEFAULT_BASE_REL,
                                                    rule="EXT-150, EXT-156")
        if script == TRAIN_SCRIPT:
            for opt, value, tok in parsed:
                if opt == "--no-verify-base":
                    out.append(f"{TRAIN_SCRIPT} {tok} skips re-hashing the base, and ext-collect refuses such a run "
                               "(run_args.json base_ref.verified false, EXT-141)")
                if opt == "--profile" and value not in PROFILES:
                    out.append(f"{TRAIN_SCRIPT} --profile {value!r} is not smoke or full: the CPU profiles are host "
                               "checks and are never collected (EXT-150)")
        return out

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record") -> list[Problem]:
        """`ext-launch --spec` (optional for this adapter): the spec's base directory and its data directory lie inside
        read-only mounts (EXT-141, EXT-142)."""
        from gatedtrain.ext.launch import _ro_table, _within

        table = _ro_table(mounts)
        out = []
        if not any(_within(str(Path(base).resolve()), host) for _, host in table):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's base directory {base}", self.BASE_FIX))
        if not any(_within(str(Path(data).resolve()), host) for _, host in table):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's training data {data}", self.DATA_FIX))
        return out

    def _base_files(self, base: Path) -> dict[str, str]:
        """{relative path: "sha256:<hex>"} of the base directory, base_ref.json's `files` shape."""
        return {e["path"]: "sha256:" + e["sha256"] for e in self._base_entries(base)}

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data", **_: Any) -> list[Problem]:
        """EXT-141 base binding, EXT-142 data binding, EXT-145 seed, profile, last cut, script and decoder."""
        from gatedtrain.ext.launch import container_file

        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent
        out: list[Problem] = []
        args, why = read_run_args(stage_out)
        if args is None:
            return [Problem("EXT-BASE", base_field, f"{why}, so the run cannot be bound to the spec's base (EXT-141)",
                            self.BASE_FIX)]
        mounts = launch.get("mounts")
        cuts = [d for _, d in self.find_cuts(stage_out)[0]] or [Path(last_cut)]
        # EXT-141 (MM-21): the run loaded the spec's base through a read-only mount and re-hashed it against
        # MODELS.json; every cut's base_ref.json names that base by a 40-hex revision and its files' SHA-256.
        got, mp = args.get("base_model"), args.get("model_name_or_path")
        if mp != got:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json model_name_or_path {mp!r} is not its base_model "
                               f"{got!r}", self.BASE_FIX))
        hit = container_file(f"{got.rstrip('/')}/config.json", mounts) if isinstance(got, str) and got else None
        if hit is None or Path(hit[1]).parent.resolve() != Path(base).resolve():
            out.append(Problem("EXT-BASE", base_field, f"run_args.json base_model {got!r} is not, through a read-only "
                               f"mount, the spec's base directory {base} (MM-21)", self.BASE_FIX))
        rb = args.get("base_ref") if isinstance(args.get("base_ref"), dict) else {}
        if rb.get("verified") is not True:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json base_ref.verified is {rb.get('verified')!r}: the "
                               "run did not re-hash the base files against MODELS.json (--no-verify-base)",
                               "train without --no-verify-base"))
        try:
            files = self._base_files(base)
        except (OSError, ValueError):
            files = None                                  # collect reports an unreadable base (EXT-BASE)
        for d in cuts:
            ref, why = _read_json(Path(d) / BASE_REF)
            if not isinstance(ref, dict):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{BASE_REF}: {why or 'not an object'}",
                                   self.BASE_FIX))
                continue
            rev = ref.get("revision")
            if not (isinstance(rev, str) and REVISION_RE.fullmatch(rev)):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{BASE_REF} revision {rev!r} is not a 40-hex "
                                   "commit (a branch name is refused, MM-18)", self.BASE_FIX))
            if files is not None and ref.get("files") != files:
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{BASE_REF} files are not the SHA-256 of the "
                                   f"spec's base directory {base} (the run trained another base, MM-21)",
                                   self.BASE_FIX))
            elif not (isinstance(ref.get("files"), dict) and ref.get("files_sha256") == compact_sha256(ref["files"])):
                out.append(Problem("EXT-BASE", base_field, f"{Path(d).name}/{BASE_REF} files_sha256 is not the hash of "
                                   "its files (compact sorted JSON, train_whisper.py base_binding)", self.BASE_FIX))
            if (rb.get("repo"), rb.get("revision"), rb.get("files_sha256")) != (ref.get("repo"), rev,
                                                                               ref.get("files_sha256")):
                out.append(Problem("EXT-BASE", base_field, f"run_args.json base_ref is not {Path(d).name}/{BASE_REF} "
                                   "(repo, revision, files_sha256)", self.BASE_FIX))
        # EXT-142: the shard the run read is the spec's, through a read-only mount, with its hash and row count.
        if data is not None:
            shard = _shard(data)
            path = args.get("train_file")
            if shard is None:
                out.append(Problem("EXT-DATA", data_field, f"the spec's training data {data} holds "
                                   f"{len(list(Path(data).glob('*.jsonl')))} shards, not the one shard the run read",
                                   "point the training dataset at the run's data directory (one train.jsonl)"))
            else:
                hit = container_file(path, mounts) if isinstance(path, str) else None
                if hit is None or Path(hit[1]).resolve() != (Path(data) / shard).resolve():
                    out.append(Problem("EXT-DATA", data_field, f"run_args.json train_file {path!r} is not, through a "
                                       f"read-only mount, the spec's shard {Path(data) / shard} (EXT-142)",
                                       self.DATA_FIX))
                elif args.get("train_file_sha256") != _sha_or_none(Path(data) / shard):
                    out.append(Problem("EXT-DATA", data_field, f"run_args.json train_file_sha256 "
                                       f"{args.get('train_file_sha256')!r} is not the SHA-256 of {Path(data) / shard} "
                                       "(EXT-142)", self.DATA_FIX))
                rows, _ = read_jsonl(Path(data) / shard)
                if rows is not None and args.get("n_train") != len(rows):
                    out.append(Problem("EXT-DATA", data_field, f"run_args.json n_train {args.get('n_train')!r} is not "
                                       f"the {len(rows)} rows of {Path(data) / shard} (EXT-142)", self.DATA_FIX))
        # EXT-145: seed, profile, script, decoder, and the run finished its steps.
        claimed, field = launch.get("training_seed"), "--launch-record.training_seed"
        seed = args.get("seed")
        if not (_int(seed) and seed == claimed):
            out.append(Problem("EXT-LAUNCH", field, f"the launch record claims training_seed {claimed!r}, but "
                               f"run_args.json `seed` is {seed!r} (MM-29)", "launch with the seed the record names"))
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        if s.get("seed") != seed:
            out.append(Problem("EXT-LAUNCH", field, f"run_args.json settings.seed {s.get('seed')!r} is not its seed "
                               f"{seed!r} (the TrainingArguments seed and data_seed, MM-29)",
                               "collect the run as train_whisper.py wrote it"))
        if args.get("profile") != launch.get("profile"):
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"run_args.json `profile` is "
                               f"{args.get('profile')!r}, the launch record's {launch.get('profile')!r}",
                               "collect the run launched with this record's profile"))
        if args.get("dry_run") is not False:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json dry_run is {args.get('dry_run')!r}, not "
                               "false", "collect a real training run"))
        if args.get("script_sha256") != launch.get("script_sha256"):
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 "
                               f"{args.get('script_sha256')!r} is not the launch record's "
                               f"{launch.get('script_sha256')!r} (EXT-83)",
                               "collect the run launched with this record's training script"))
        decoding = args.get("decoding")
        if not (isinstance(decoding, dict) and args.get("decoding_sha256") == compact_sha256(decoding)):
            out.append(Problem("EXT-SAMPLER", "--stage-out", f"run_args.json decoding_sha256 "
                               f"{args.get('decoding_sha256')!r} is not the hash of its decoding (compact sorted JSON)",
                               "collect the run as train_whisper.py wrote it"))
        rt, _ = _read_json(stage_out / RUNTIME_FILE)
        code = rt.get("decoder_code_sha256") if isinstance(rt, dict) else None
        for d in cuts:
            doc, _ = read_cut_marker(d)
            doc = doc or {}
            if doc.get("seed") != claimed:
                out.append(Problem("EXT-LAUNCH", field, f"{Path(d).name}/cut.json seed is {doc.get('seed')!r}, not the "
                                   f"launch record's training_seed {claimed!r} (MM-29)",
                                   "collect the cuts of the run launched with this record"))
            if doc.get("script_sha256") != launch.get("script_sha256"):
                out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"{Path(d).name}/cut.json "
                                   f"script_sha256 is {doc.get('script_sha256')!r}, not the launch record's "
                                   f"{launch.get('script_sha256')!r} (EXT-83)",
                                   "collect the cuts of the run launched with this record's training script"))
            if doc.get("profile") != launch.get("profile"):
                out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"{Path(d).name}/cut.json profile is "
                                   f"{doc.get('profile')!r}, the launch record's {launch.get('profile')!r}",
                                   "collect the run launched with this record's profile"))
            dc = doc.get("decoder_code_sha256")
            if not (isinstance(code, str) and dc == code == args.get("decoder_code_sha256")):
                out.append(Problem("EXT-SAMPLER", "--stage-out", f"{Path(d).name}/cut.json decoder_code_sha256 {dc!r}, "
                                   f"{RUNTIME_FILE} {code!r} and run_args.json {args.get('decoder_code_sha256')!r} "
                                   "differ: the cut was not written by this run's decoder code",
                                   "collect the cuts of this run as it wrote them"))
            elif dc != DECODER_CODE_SHA256:
                out.append(Problem("EXT-SAMPLER", "--stage-out", f"{Path(d).name}/cut.json decoder_code_sha256 {dc} is "
                                   f"not the evaluator's decoder code {DECODER_CODE_SHA256} (eval_asr.py SHARED DECODER; "
                                   "README item 5)", "train and evaluate with the same SHARED DECODER block"))
            dec, _ = _read_json(Path(d) / DECODING)
            if dec != decoding:
                out.append(Problem("EXT-SAMPLER", "--stage-out", f"{Path(d).name}/{DECODING} is not run_args.json "
                                   "`decoding`", "collect the cuts of this run as it wrote them"))
        m = CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        if not (_int(args.get("max_steps")) and args["max_steps"] == last):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is step {last}, but run_args.json max_steps "
                               f"is {args.get('max_steps')!r}: the run did not save its last step",
                               "collect a run that finished and saved its last step"))
        if s.get("max_steps") != args.get("max_steps"):
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json settings.max_steps {s.get('max_steps')!r} "
                               f"is not its max_steps {args.get('max_steps')!r}",
                               "collect the run as train_whisper.py wrote it"))
        return out

    def sampler_problems(self, cuts: list[Path], sampler: Any) -> list[Problem]:
        """EXT-145: every cut's deployment decoder (decoding.json, a model component) is the registered sampler without
        its evaluation-only keys (batch size, dtype, order)."""
        from gatedtrain.ext import arm

        want = ({k: v for k, v in sampler.items() if k not in EVAL_ONLY_KEYS} if isinstance(sampler, dict) else None)
        out = []
        for d in cuts:
            doc, _ = _read_json(Path(d) / DECODING)
            if want is None or not arm.same_json(doc, want):
                out.append(Problem("EXT-SAMPLER", "registration.sampler", f"{Path(d).name}/{DECODING} is not the "
                                   "registered sampler's decoder: the cut deploys another decoder than the gate "
                                   "evaluates", "register eval_asr.py's sampler for the decoder the cuts carry"))
        return out
