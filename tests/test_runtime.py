"""The runtime scripts (runtime/train_whisper.py, runtime/eval_asr.py) on CPU without torch, transformers or
soundfile: stand-in modules replace them, so what runs is the scripts' own code (settings, the shared deployment
decoder, the label text and collator, the consumption logs, the cut writer, the WER scorer and the arm writer), and
the outputs are read with the adapter and gatedtrain's arm reader as ext-collect reads them.

Why: these scripts run only in the training and evaluation containers (or for minutes on a host CPU with the real
models), so a fault in their pure parts was found only after the run. The stand-in Seq2SeqTrainer follows what
runtime/README.md "Trainer" relies on: per epoch a seeded permutation of the rows, micro-batches of
per_device_train_batch_size (the last one short, dataloader_drop_last false), every micro-batch through
compute_loss, then on_pre_optimizer_step, global_step += 1, on_step_end, stop at max_steps. Its compute_loss refuses
inputs the Whisper model would refuse (an `example_ids` key) and labels that still start with the decoder start token.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import io
import json
import re
import shutil
import struct
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np

from gatedtrain.canonical import file_sha256, sha256_bytes
from gatedtrain.ext.arm import read_lines, read_meta
from gatedtrain_recipe_whisper_sft_ami import adapter as adapter_mod
from gatedtrain_recipe_whisper_sft_ami.adapter import HfSeq2SeqAdapter

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
FIX = Path(__file__).resolve().parent / "fixtures" / "whisper"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"whisper_runtime_{name}", RUNTIME / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tw = load("train_whisper")
ea = load("eval_asr")

SOT, NO_TS, EOT = 50257, 50362, 50256          # <|startoftranscript|>, <|notimestamps|>, <|endoftext|>


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def simple_norm(text: str) -> str:
    """Stand-in for Whisper's EnglishTextNormalizer: lower case, punctuation to spaces, single spaces."""
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())


# ---------------------------------------------------------------------------
# Stand-in modules
# ---------------------------------------------------------------------------

class Tensor(np.ndarray):
    """numpy array with the two torch methods the scripts call."""

    def to(self, *args):
        return self


def as_tensor(x, dtype=None):
    return np.asarray(x, dtype=dtype).view(Tensor)


def full(shape, value, dtype=None):
    return np.full(shape, value, dtype=dtype).view(Tensor)


class FakeTokenizer:
    eos_token_id = EOT

    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        return types.SimpleNamespace(input_ids=[SOT, NO_TS] + [ord(c) for c in text] + [EOT])

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens
        return "".join(chr(i) for i in ids if i < EOT)


class FakeFeatureExtractor:
    def __call__(self, audio, sampling_rate, return_tensors):
        assert sampling_rate == 16000
        n = len(audio) if isinstance(audio, list) else 1
        return {"input_features": np.zeros((n, 2, 3), dtype=np.float32).view(Tensor)}


class FakeProcessor:
    def __init__(self):
        self.tokenizer, self.feature_extractor = FakeTokenizer(), FakeFeatureExtractor()

    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert local_files_only
        return cls()

    def save_pretrained(self, d):
        for name in ("processor_config.json", "tokenizer.json", "tokenizer_config.json"):
            Path(d, name).write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")


def safetensors_bytes(values: list[float]) -> bytes:
    data = struct.pack(f"<{len(values)}f", *values)
    header = json.dumps({"proj.weight": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(data)]},
                         "__metadata__": {"format": "pt"}}).encode()
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + data


class FakeModel:
    """WhisperForConditionalGeneration: generate() returns the next scripted token sequences (one per audio)."""

    scripts: deque = deque()

    def __init__(self):
        self.config = types.SimpleNamespace(use_cache=False, decoder_start_token_id=SOT)
        self.training, self.updates = True, 0

    @classmethod
    def from_pretrained(cls, path, torch_dtype=None, local_files_only=False):
        assert local_files_only
        return cls()

    def to(self, device):
        return self

    def eval(self):
        self.training = False
        return self

    def save_pretrained(self, d, safe_serialization):
        assert safe_serialization
        Path(d, "model.safetensors").write_bytes(safetensors_bytes([float(self.updates), 1.0]))
        Path(d, "config.json").write_text(json.dumps({"use_cache": self.config.use_cache}) + "\n")
        Path(d, "generation_config.json").write_text("{}\n")

    def generate(self, input_features, num_beams, do_sample, max_new_tokens, return_timestamps):
        assert (num_beams, do_sample, max_new_tokens, return_timestamps) == (1, False, 200, False)
        seqs = [FakeModel.scripts.popleft() for _ in range(input_features.shape[0])]
        if any(s == "FAIL" for s in seqs):
            raise RuntimeError("CUDA error: device-side assert")
        width = max(len(s) for s in seqs)
        return np.array([s + [EOT] * (width - len(s)) for s in seqs])


class FakeTrainingArguments:
    def __init__(self, **kw):
        self.kw = kw
        self.__dict__.update(kw)

    def to_json_string(self):
        return json.dumps(self.kw)


class FakeTrainerCallback:
    pass


class FakeSeq2SeqTrainer:
    def __init__(self, *, model, args, train_dataset, data_collator, processing_class, callbacks):
        self.model, self.args, self.dataset, self.collate, self.callbacks = (model, args, train_dataset,
                                                                             data_collator, callbacks)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        assert set(inputs) == {"input_features", "labels"}, sorted(inputs)
        labels = np.asarray(inputs["labels"])
        assert not (labels[:, 0] == SOT).any(), "labels still start with the decoder start token"
        self.batches.append(labels)
        return 0.0

    def _event(self, name, **kw):
        for cb in self.callbacks:
            getattr(cb, name)(self.args, self.state, None, **kw)

    def train(self):
        a = self.args
        rng = np.random.default_rng(a.data_seed)
        self.batches = []
        self.state = types.SimpleNamespace(global_step=0, save_to_json=lambda p: Path(p).write_text(
            json.dumps({"global_step": self.state.global_step}) + "\n"))
        bs = a.per_device_train_batch_size
        while True:
            self._event("on_epoch_begin")
            order = [int(i) for i in rng.permutation(len(self.dataset))]
            for b0 in range(0, len(order), bs):
                inputs = self.collate([self.dataset[i] for i in order[b0:b0 + bs]])
                self.model.training = True
                self.compute_loss(self.model, inputs)
                self._event("on_pre_optimizer_step")
                self.model.updates += 1
                self.state.global_step += 1
                if self.state.global_step % a.logging_steps == 0:
                    self._event("on_log", logs={"loss": 1.0})
                self._event("on_step_end", model=self.model)
                if self.state.global_step >= a.max_steps:
                    return


def fake_sf_read(f, dtype, always_2d):
    raw = f.read()
    return np.zeros(16000 * 10, dtype=np.float32), 16000          # 10 s: a bound of 136 text tokens


def stub_modules() -> dict:
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    torch = mod("torch", __version__="stub", float16="float16", float32="float32", long=np.int64,
                as_tensor=as_tensor, full=full, no_grad=contextlib.nullcontext, manual_seed=lambda s: None,
                device=lambda d: types.SimpleNamespace(type=str(d).split(":")[0]),
                cuda=types.SimpleNamespace(is_available=lambda: False), version=types.SimpleNamespace(cuda=None),
                utils=types.SimpleNamespace(data=types.SimpleNamespace(Dataset=object)))
    return {
        "torch": torch,
        "transformers": mod("transformers", __version__="5.21.0", Seq2SeqTrainer=FakeSeq2SeqTrainer,
                            Seq2SeqTrainingArguments=FakeTrainingArguments, TrainerCallback=FakeTrainerCallback,
                            WhisperForConditionalGeneration=FakeModel, WhisperProcessor=FakeProcessor),
        "transformers.models": mod("transformers.models"),
        "transformers.models.whisper": mod("transformers.models.whisper"),
        "transformers.models.whisper.english_normalizer": mod(
            "transformers.models.whisper.english_normalizer", EnglishTextNormalizer=lambda spelling: simple_norm),
        "soundfile": mod("soundfile", read=fake_sf_read),
    }


_real_version = importlib.metadata.version


def fake_version(name):
    return "0.13.1" if name == "soundfile" else _real_version(name)


@contextlib.contextmanager
def stubbed():
    with mock.patch.dict(sys.modules, stub_modules()), \
            mock.patch.object(importlib.metadata, "version", fake_version), \
            contextlib.redirect_stdout(io.StringIO()):
        yield


def write_audio_rows(d: Path, rows: list[dict], key: str) -> list[dict]:
    """Rows whose `audio` file exists under `d` with the recorded SHA-256 (bytes stand in for WAV)."""
    out = []
    for i, r in enumerate(rows):
        raw = f"wav stand-in {r[key]} {i}".encode()
        (d / r["audio"]).parent.mkdir(parents=True, exist_ok=True)
        (d / r["audio"]).write_bytes(raw)
        out.append(dict(r, audio_sha256=sha256_bytes(raw)))
    return out


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------

class TestPins(unittest.TestCase):
    def test_settings_are_the_readme_trainer_and_profiles(self):
        # runtime/README.md "Trainer" and "Profiles and cost".
        c = tw.COMMON
        self.assertEqual((c["learning_rate"], c["lr_scheduler_type"], c["warmup_steps"], c["weight_decay"],
                          c["per_device_train_batch_size"], c["gradient_accumulation_steps"], c["max_grad_norm"],
                          c["precision"], c["dataloader_num_workers"], c["full_determinism"]),
                         (1.0e-5, "linear", 30, 0.0, 32, 1, 1.0, "bf16", 4, False))
        cuts = {}
        for name, prof in tw.PROFILES.items():
            s = tw.resolve_settings(tw.parse_args(["--profile", name, "--out", "/o"]))
            cuts[name] = ([k for k in range(1, s["max_steps"] + 1) if k % s["save_steps"] == 0 or k == s["max_steps"]],
                          s["per_device_train_batch_size"])
            self.assertEqual(s["seed"], 4000)
            self.assertEqual(s["full_determinism"], name.startswith("cpu-"), name)
        self.assertEqual(cuts, {"smoke": ([30, 60], 32), "full": ([100, 200, 300], 32), "cpu-sanity": ([2, 4], 4),
                                "cpu-pilot": ([60], 16)})

    def test_shared_decoder_is_identical_and_registered(self):
        self.assertEqual(tw.decoder_code_sha256(), ea.decoder_code_sha256())
        self.assertEqual(tw.decoder_code_sha256(), adapter_mod.DECODER_CODE_SHA256)
        self.assertEqual(tw.decoding_doc(), ea.decoding_doc())
        self.assertEqual(ea.decoding_doc(), json.loads((FIX / "decoding.json").read_text(encoding="utf-8")))

    def test_cut_files_are_the_adapters(self):
        self.assertEqual(tw.CUT_FILES, adapter_mod.CUT_FILES)
        self.assertEqual(sorted(tw.DEPLOYABLE), sorted(adapter_mod.COMPONENTS))

    def test_max_text_tokens(self):
        self.assertEqual(ea.max_text_tokens(np.zeros(16000)), 28)              # 16 + 12
        self.assertEqual(ea.max_text_tokens(np.zeros(16001)), 29)              # ceil
        self.assertEqual(ea.max_text_tokens(np.zeros(0)), 16)
        self.assertEqual(ea.max_text_tokens(np.zeros(16000 * 30)), 200)        # capped by max_new_tokens

    def test_committed_items(self):
        rows = ea.read_items(RUNTIME / "items")
        meta = json.loads((RUNTIME / "data" / "data_meta.json").read_text(encoding="utf-8"))
        train_speakers = set(meta["split"]["train_speakers"])
        self.assertEqual(len(rows), meta["split"]["n_items"])
        for r in rows:
            ref = r["reference_normalized"]
            self.assertEqual(ref, " ".join(ref.split()), r["item"])
            self.assertEqual(r["n_words"], len(ref.split()))
            self.assertGreaterEqual(r["n_words"], 8)
            self.assertLessEqual(r["duration"], 30.0)
            self.assertEqual(r["audio"], f"audio/{r['item']}.wav")
            self.assertNotIn(r["speaker_id"], train_speakers)
            self.assertEqual(ref, ref.lower())


# ---------------------------------------------------------------------------
# WER scorer
# ---------------------------------------------------------------------------

class TestScorer(unittest.TestCase):
    def test_edit_counts(self):
        def c(r, h):
            d = ea.edit_counts(r.split(), h.split())
            return d["substitutions"], d["deletions"], d["insertions"], d["errors"]
        self.assertEqual(c("a b c", "a b c"), (0, 0, 0, 0))
        self.assertEqual(c("a b c", "a x c"), (1, 0, 0, 1))
        self.assertEqual(c("a b c", "a c"), (0, 1, 0, 1))
        self.assertEqual(c("a b c", "a b c d"), (0, 0, 1, 1))
        self.assertEqual(c("a b c", ""), (0, 3, 0, 3))
        self.assertEqual(c("a", "x y z"), (1, 0, 2, 3))
        self.assertEqual(c("a b", "b a"), (2, 0, 0, 2))               # substitution preferred over D + I
        self.assertEqual(c("the cat sat", "cat sat on"), (0, 1, 1, 2))

    def test_score(self):
        s = ea.score(simple_norm, "HELLO, World there", "hello world")
        self.assertEqual((s["wer"], s["ref_words"], s["hyp_words"], s["empty"], s["deletions"]), (1 / 3, 3, 2, 0, 1))
        self.assertEqual(s["reference_normalized"], "hello world there")
        empty = ea.score(simple_norm, "a b", "  ")
        self.assertEqual((empty["wer"], empty["empty"]), (1.0, 1))
        self.assertEqual(ea.score(simple_norm, "a b", "a b c d e")["wer"], 1.5)    # a hallucination exceeds 1
        with self.assertRaisesRegex(ValueError, "normalized reference is empty"):
            ea.score(simple_norm, "...", "a")


class TestTranscribe(unittest.TestCase):
    def test_text_tokens_are_cut_to_the_utterance_bound(self):
        with stubbed():
            proc = FakeProcessor()
            audios = [np.zeros(16000), np.zeros(16000), np.zeros(8000)]
            long = [ord("a")] * 40
            FakeModel.scripts = deque([[SOT, NO_TS] + [ord(c) for c in " hi"] + [EOT], [SOT] + long + [EOT],
                                       [ord("o"), ord("k"), EOT, EOT]])        # generate pads with EOT
            texts, cut = ea.transcribe(FakeModel(), proc, audios, "cpu", "float32")
        self.assertEqual(texts, [" hi", "a" * 28, "ok"])           # specials dropped, 28 = 16 + 12 tokens
        self.assertEqual(cut, [False, True, False])


# ---------------------------------------------------------------------------
# train_whisper.py end to end with the stand-ins; logs and cuts read by the adapter
# ---------------------------------------------------------------------------

class TestTrainRun(unittest.TestCase):
    """cpu-sanity profile (4 steps of 4, cuts 2 and 4) on 6 training rows: steps 1-2 are epoch 0 (4 + 2 rows),
    steps 3-4 epoch 1."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.base = root / "base"
        record = json.loads((RUNTIME / "models" / "MODELS.json").read_text())["models"]["whisper-base.en"]
        for rel in record["files"]:
            (cls.base / rel).parent.mkdir(parents=True, exist_ok=True)
            (cls.base / rel).write_text("stand-in\n")
        cls.data = root / "data"
        sample = read_jsonl(FIX / "train.sample.jsonl")
        rows = [dict(r, utt_id=f"{r['utt_id']}_{k}", audio=f"audio/{r['utt_id']}_{k}.wav") for k in range(2)
                for r in sample]
        rows = write_audio_rows(cls.data, rows, "utt_id")
        (cls.data / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        cls.rows = rows
        cls.out = root / "out"
        trainers = []
        orig_init = FakeSeq2SeqTrainer.__init__

        def init(self, **kw):
            orig_init(self, **kw)
            trainers.append(self)
        with stubbed(), mock.patch.object(FakeSeq2SeqTrainer, "__init__", init):
            tw.main(["--profile", "cpu-sanity", "--out", str(cls.out), "--base", str(cls.base), "--data",
                     str(cls.data), "--device", "cpu", "--no-verify-base"])
        cls.trainer = trainers[0]
        cls.adapter = HfSeq2SeqAdapter()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_labels_are_the_lower_cased_text_after_one_space_padded_with_minus_100(self):
        texts = {" " + r["text"].lower() for r in self.rows}
        first = self.trainer.batches[0]
        self.assertEqual(first.shape[0], 4)
        decoded = set()
        for row in first:
            ids = [int(t) for t in row if t != -100]
            self.assertEqual(ids[0], NO_TS)                               # start token dropped, notimestamps kept
            self.assertEqual(ids[-1], EOT)
            decoded.add("".join(chr(t) for t in ids[1:-1]))
            self.assertTrue(all(t == -100 for t in row[len(ids):]))      # right padding only
        self.assertLessEqual(decoded, texts)
        self.assertTrue(any(t == -100 for t in first.flatten()))

    def test_cuts_are_complete_for_the_adapter(self):
        complete, incomplete = self.adapter.find_cuts(self.out)
        self.assertEqual(([s for s, _ in complete], incomplete), ([2, 4], []))
        marker = json.loads((self.out / "checkpoint-4" / "cut.json").read_text())
        self.assertEqual((marker["global_step"], marker["examples_consumed"]), (4, 12))
        self.assertEqual(json.loads((self.out / "checkpoint-2" / "config.json").read_text()), {"use_cache": True})
        self.assertEqual(json.loads((self.out / "checkpoint-2" / "decoding.json").read_text()), tw.decoding_doc())
        self.assertEqual(marker["decoder_code_sha256"], ea.decoder_code_sha256())
        self.assertFalse((self.out / "hf-trainer").exists())

    def test_consumption_logs_pass_the_adapter(self):
        cuts, _ = self.adapter.find_cuts(self.out)
        self.assertEqual(self.adapter.consumption_problems(self.out, self.data, cuts), [])
        rows = self.adapter.consumption(self.out, self.data, cuts, run_id="r")["rows"]
        self.assertEqual(len(rows), 12)
        steps = read_jsonl(self.out / "train_steps.jsonl")
        self.assertEqual([(s["step"], s["epoch"], s["n_examples"]) for s in steps],
                         [(1, 0, 4), (2, 0, 2), (3, 1, 4), (4, 1, 2)])
        ids = {r["utt_id"] for r in self.rows}
        for k in (0, 1):
            self.assertEqual(set(steps[2 * k]["example_ids"] + steps[2 * k + 1]["example_ids"]), ids)

    def test_evaluator_accepts_the_cut(self):
        record = json.loads((RUNTIME / "models" / "MODELS.json").read_text())["models"]["whisper-base.en"]
        base_ref = json.loads((self.out / "checkpoint-4" / "base_ref.json").read_text())
        self.assertEqual((base_ref["revision"], base_ref["files"]),
                         (record["revision"], {k: v["sha256"] for k, v in record["files"].items()}))
        self.assertEqual(ea.check_cut(self.out / "checkpoint-4")["global_step"], 4)

    def test_run_args(self):
        args = json.loads((self.out / "run_args.json").read_text())
        self.assertEqual((args["seed"], args["max_steps"], args["train_file_sha256"], args["n_train"]),
                         (4000, 4, file_sha256(self.data / "train.jsonl"), 6))
        ta = args["training_args"]
        self.assertEqual((ta["seed"], ta["data_seed"], ta["save_strategy"], ta["bf16"], ta["use_cpu"],
                          ta["remove_unused_columns"], ta["label_names"], ta["dataloader_drop_last"]),
                         (4000, 4000, "no", False, True, False, ["labels"], False))
        attempts = read_jsonl(self.out / "attempts.jsonl")
        self.assertEqual((attempts[-1]["global_step"], attempts[-1]["examples_consumed"]), (4, 12))

    def test_max_steps_override_ends_with_a_cut(self):
        out = Path(self._tmp.name) / "out-3"
        with stubbed():
            tw.main(["--profile", "cpu-sanity", "--out", str(out), "--base", str(self.base), "--data", str(self.data),
                     "--device", "cpu", "--no-verify-base", "--max-steps", "3"])
        cuts, incomplete = self.adapter.find_cuts(out)
        self.assertEqual(([s for s, _ in cuts], incomplete), ([2, 3], []))
        self.assertEqual(self.adapter.consumption_problems(out, self.data, cuts), [])


class TestTrainHelpers(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_safetensors_header_is_canonical_and_tensors_unchanged(self):
        p = self.tmp / "model.safetensors"
        lengths = set()
        for k in range(1, 9):                         # canonical headers of eight different lengths
            raw = safetensors_bytes([float(v) for v in range(k)])
            p.write_bytes(raw)
            tw.canonicalize_safetensors_header(p)
            new = p.read_bytes()
            n = struct.unpack("<Q", new[:8])[0]
            self.assertEqual(n % 8, 0)
            header = new[8:8 + n].rstrip(b" ")
            lengths.add(len(header) % 8)
            self.assertEqual(header, json.dumps(json.loads(header), sort_keys=True, separators=(",", ":")).encode())
            self.assertEqual(new[8 + n:], raw[-4 * k:])
            self.assertIsNone(adapter_mod.safetensors_problem(p))
        self.assertGreater(len(lengths), 1)

    def test_read_train_and_base_binding_refusals(self):
        rows = write_audio_rows(self.tmp, [{"utt_id": "u1", "audio": "audio/u1.wav"},
                                           {"utt_id": "u2", "audio": "audio/u2.wav"}], "utt_id")
        (self.tmp / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertEqual(len(tw.read_train(self.tmp)[0]), 2)
        (self.tmp / "audio" / "u2.wav").write_bytes(b"other")
        with self.assertRaisesRegex(SystemExit, "differs from its SHA-256"):
            tw.read_train(self.tmp)
        (self.tmp / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [rows[0], rows[0]]))
        with self.assertRaisesRegex(SystemExit, "not unique"):
            tw.read_train(self.tmp)
        base = self.tmp / "base"
        base.mkdir()
        (base / "config.json").write_text("{}")
        with self.assertRaisesRegex(SystemExit, "does not hold exactly the files"):
            tw.base_binding(base, verify=False)


# ---------------------------------------------------------------------------
# eval_asr.py: the arm
# ---------------------------------------------------------------------------

class TestEvaluateArm(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        items = self.tmp / "items"
        rows = read_jsonl(FIX / "items.sample.jsonl")
        rows = write_audio_rows(items, rows, "item")
        for r in rows:
            r["reference_normalized"] = simple_norm(r["text"])
        rows[2]["reference_normalized"] += " edited"                 # not the scorer's normalization
        (items / "items.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.items, self.rows = items, rows
        norm_json = self.tmp / "normalizer.json"
        norm_json.write_text("{}\n")
        p = mock.patch.object(ea, "NORMALIZER_JSON", norm_json)
        p.start()
        self.addCleanup(p.stop)

    def evaluate(self, scripts, *extra):
        FakeModel.scripts = deque(scripts)
        out = self.tmp / "eval"
        with stubbed():
            ea.main(["--base-only", "--base", str(self.tmp), "--items", str(self.items), "--out", str(out),
                     "--device", "cpu", "--batch-size", "2", "--model-hash", "sha256:" + "a" * 64,
                     "--base-model-hash", "sha256:" + "a" * 64, "--item-set-hash", "sha256:" + "b" * 64, *extra])
        return out

    def tokens(self, text):
        return [SOT, NO_TS] + [ord(c) for c in text] + [EOT]

    def check_arm(self, arm):
        adapter = HfSeq2SeqAdapter()
        row = adapter.metric(ea.EVALUATOR_ID)
        meta, bad = read_meta(arm)
        self.assertEqual(bad, [])
        self.assertEqual((meta["metric"], meta["direction"], meta["score_range"]),
                         (row["metric"], row["direction"], list(row["score_range"])))
        records, bad = read_lines(arm, seeds=meta["eval_seeds"], score_range=row["score_range"],
                                  error_classes=row["error_classes"])
        self.assertEqual(bad, [])
        self.assertEqual(adapter.arm_problems(arm, records, self.rows), [])
        return meta, records

    def test_arm(self):
        r0, r1 = self.rows[0]["text"], self.rows[1]["text"]
        hyp1 = " ".join(r1.split()[:-1])                              # one deletion
        out = self.evaluate([self.tokens(" " + r0.lower()), self.tokens(" " + hyp1), self.tokens(" whatever")])
        meta, records = self.check_arm(out / "arm-wer")
        self.assertEqual(meta["eval_seeds"], [11])
        self.assertEqual([r["score"] for r in records[:2]], [0.0, 1 / len(r1.split())])
        self.assertEqual(records[1]["sub_metrics"]["deletions"], 1)
        self.assertEqual(records[2]["error"]["class"], "scorer_error")    # reference_normalized differs
        out_doc = json.loads((out / "arm-wer" / records[0]["output_ref"]["path"]).read_text())
        self.assertEqual(out_doc["reference_normalized"], self.rows[0]["reference_normalized"])
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual((summary["errors"], summary["base_only"]), (1, True))

    def test_decode_failure_is_an_error_per_item_of_the_batch(self):
        out = self.evaluate([self.tokens(" x"), "FAIL", self.tokens(" y")])
        meta, records = self.check_arm(out / "arm-wer")
        self.assertEqual([r["error"]["class"] if r["error"] else None for r in records],
                         ["decode_error", "decode_error", "scorer_error"])

    def test_refusals(self):
        with self.assertRaisesRegex(SystemExit, "exactly one evaluation seed"):
            self.evaluate([], "--eval-seeds", "1,2")


# ---------------------------------------------------------------------------
# make_data.py: split, filters and disjointness on tiny parquet shards
# ---------------------------------------------------------------------------

class TestMakeData(unittest.TestCase):
    """Train shard: speakers A and B (one row too short, one too long). Test shard: speakers C and D, one row by A
    (a shared speaker), one row whose text is a training text, one row under 8 normalized words."""

    WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet".upper().split()

    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.md = load("make_data")
        ihm = root / "data-source" / "ami-ihm" / "ihm"
        ihm.mkdir(parents=True)

        def row(uid, spk, text, seconds=3.0, meeting="EN2001a"):
            return {"audio_id": uid, "text": text, "speaker_id": spk, "meeting_id": meeting, "microphone_id": "H00",
                    "begin_time": 1.0, "end_time": 1.0 + seconds, "audio": {"bytes": f"wav {uid}".encode()}}
        w = self.WORDS
        self.shared_text = " ".join(w[:9])
        train = [row(f"T{k}", "A" if k % 2 else "B", " ".join(w[k % 3:k % 3 + 5]) + f" {k}") for k in range(6)]
        train += [row("Tshort", "A", "YEAH OKAY MM-HMM"), row("Tlong", "B", " ".join(w), seconds=31.0),
                  row("Tshared", "A", self.shared_text)]
        test = [row(f"E{k}", "C" if k % 2 else "D", " ".join(w[:8]) + f" -- N{k}", meeting="EN2002a")
                for k in range(5)]                     # `--` is a raw word the normalizer removes
        test += [row("Espk", "A", " ".join(w[:9]) + " X", meeting="EN2002a"),
                 row("Eover", "C", self.shared_text, meeting="EN2002a"),
                 row("Efew", "D", "ONE TWO THREE", meeting="EN2002a")]
        train.reverse()                                # the pools are sorted by utt_id, not read in file order
        test.reverse()
        files = {}
        for name, rows in (("ihm/train-00000-of-00001.parquet", train), ("ihm/test-00000-of-00001.parquet", test)):
            pq.write_table(pa.Table.from_pylist(rows), root / "data-source" / "ami-ihm" / name)
            files[name] = {"sha256": file_sha256(root / "data-source" / "ami-ihm" / name)}
        (root / "data-source" / "DATA.json").write_text(json.dumps({"datasets": {"ami-ihm": {"files": files}}}))
        (root / "models" / "whisper-base.en").mkdir(parents=True)
        (root / "models" / "whisper-base.en" / "normalizer.json").write_text("{}\n")
        md = self.md
        patches = {"HERE": root, "SOURCE": root / "data-source", "AMI_DIR": root / "data-source" / "ami-ihm",
                   "DATA": root / "data", "ITEMS": root / "items",
                   "GENERATED": (root / "data" / "audio", root / "data" / "train.jsonl", root / "items" / "audio"),
                   "COMMITTED": (root / "data" / "data_meta.json", root / "items" / "items.jsonl",
                                 root / "items" / "item_set_meta.json"),
                   "NORMALIZER_JSON": root / "models" / "whisper-base.en" / "normalizer.json"}
        for k, v in patches.items():
            p = mock.patch.object(md, k, v)
            p.start()
            self.addCleanup(p.stop)
        self.root = root

    def run_md(self):
        mods = stub_modules()
        mods["soundfile"].info = lambda f: types.SimpleNamespace(samplerate=16000, channels=1)
        with mock.patch.dict(sys.modules, mods), contextlib.redirect_stdout(io.StringIO()):
            return self.md.main(["--n-train", "100", "--n-items", "100", "--seed", "1"])   # every row that passes

    def test_split_filters_and_disjointness(self):
        self.assertEqual(self.run_md(), 0)
        train = read_jsonl(self.root / "data" / "train.jsonl")
        items = read_jsonl(self.root / "items" / "items.jsonl")
        self.assertEqual(sorted(r["utt_id"] for r in train), ["T0", "T1", "T2", "T3", "T4", "T5", "Tshared"])
        self.assertEqual([r["item"] for r in items], ["E0", "E1", "E2", "E3", "E4"])
        self.assertFalse({r["utt_id"] for r in train} & {"Tshort", "Tlong"})
        self.assertFalse({r["item"] for r in items} & {"Espk", "Eover", "Efew"})
        meta = json.loads((self.root / "items" / "item_set_meta.json").read_text())["split"]
        self.assertEqual((meta["item_pool"]["dropped_shared_speaker"], meta["item_pool"]["dropped_text_overlap"],
                          meta["exact_overlap_raw"], meta["shared_speakers"]), (1, 1, 0, []))
        for r in train + items:
            path = (self.root / "data" if "utt_id" in r else self.root / "items") / r["audio"]
            self.assertEqual(file_sha256(path), r["audio_sha256"])
        for r in items:
            self.assertEqual(r["reference_normalized"], simple_norm(r["text"]))
            self.assertEqual(r["n_words"], len(r["reference_normalized"].split()))
        self.assertEqual(train, sorted(train, key=lambda r: r["utt_id"]))

    def test_reproduces_the_committed_files_or_refuses(self):
        self.run_md()
        with self.assertRaisesRegex(SystemExit, "remove the generated data first"):
            self.run_md()
        for d in ("data/audio", "items/audio"):
            shutil.rmtree(self.root / d)
        (self.root / "data" / "train.jsonl").unlink()
        self.assertEqual(self.run_md(), 0)                               # same seed: byte for byte
        for d in ("data/audio", "items/audio"):
            shutil.rmtree(self.root / d)
        (self.root / "data" / "train.jsonl").unlink()
        items = self.root / "items" / "items.jsonl"
        items.write_text(items.read_text() + "\n")
        with self.assertRaisesRegex(SystemExit, "regenerated files differ from the committed ones: items/items.jsonl"):
            self.run_md()


if __name__ == "__main__":
    unittest.main()
