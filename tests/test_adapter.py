"""External-run adapters, slice 4: area "audio" (Whisper base.en full-parameter SFT on AMI with the HF Seq2SeqTrainer;
tests/README.md section 11.21, EXT-138 to EXT-153).

Sources: MULTIMODAL-SPEC.md MM-1 to MM-39 (section 6 row 5 and its deviations, MM-11, MM-17, MM-20, MM-21, MM-23,
MM-27, MM-29, MM-33, MM-38), design revision 16: 05 section 6.1 (tiers; "Shared labelers", R16-1: held-out references
of the training labels' annotation process leave a programmatic scorer at its tier, and the registration and the report
state the shared convention), section 11 (lineage), sections 11.2, 12.1, 12.2 (an observed `row` record);
the maintainers' area 4 decisions (not published; runtime/README.md has the proposals); the runtime `runtime/`
(README "What gatedtrain needs", "Launch request facts"; `train_whisper.py`, `eval_asr.py`; read 2026-11-13; the code is
followed where the two differ). Helpers: `tests/_whisper.py`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _whisper import (ADAPTER, AREA4, EVIDENCE, BASE_FILES, BASE_NAME, CUT_FILES, DECODER_CODE_SHA256, DECODING, DEPLOYABLE,
                      ENV_NAMES, EVAL_ASR, EVAL_ASR_SHA256, EVAL_ONLY_KEYS, EVALUATOR_ID, METRIC, N_TRAIN, REPO,
                      REVISION, RUNTIME_CONTAINER, SAMPLER, SCORER, TRAIN_WHISPER, TRAINER_ID, WHISPER,
                      WhisperScenario, build_base, build_cut, compact_hash, cut_model_hash, decoder_block_sha256,
                      read_jsonl, sha256_text, smoke_record, write_json, write_jsonl)
from gatedtrain.testing.ext import sha_of, write_canonical
from gatedtrain.testing.support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, manifest_hash, parse_errors, parse_run_dir, run_cli


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def adapter():
    return registry().adapter(ADAPTER)


class WhisperCase(unittest.TestCase):
    """One scenario per test that changes it (`fresh`)."""

    N_ITEMS = 12

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-whisper-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def fresh(self, **kw) -> WhisperScenario:
        type(self).n += 1
        kw.setdefault("n_items", self.N_ITEMS)
        return WhisperScenario(self.tmp / f"sc-{type(self).n}", **kw)

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def passes(self, sc: WhisperScenario, **kw) -> Path:
        proc = sc.collect(self.runs(), **kw)
        self.assertIn(proc.returncode, (EXIT_OK, 1), describe(proc))
        return parse_run_dir(proc.stdout)

    def refused(self, rule: str, field_prefix: str | None = None, *, sc: WhisperScenario, text: str | None = None,
                **kw):
        runs = self.runs()
        proc = sc.collect(runs, **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix}... {text!r}; got {errs}\n{describe(proc)}")
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a refusal writes no run directory")
        return hits


# ---------------------------------------------------------------------------
# EXT-138: registry rows
# ---------------------------------------------------------------------------

class TestRegistryRows(unittest.TestCase):

    def test_trainer_row(self):
        row = registry().TRAINER_ROWS[ADAPTER]
        self.assertEqual(list(row["uses"]), [TRAINER_ID])
        self.assertEqual(list(row["evaluators"]), [EVALUATOR_ID])
        self.assertEqual(row["consumption"], {"target": "row", "fallback": None})
        self.assertEqual(set(row["pins"]), {"transformers", "accelerate", "torch"})
        self.assertEqual(row["framework"], "transformers")
        self.assertEqual(tuple(row["components"]), DEPLOYABLE)
        self.assertFalse(row["spec_binding"])
        self.assertEqual(row["gpu_h_estimate"], 0.15)

    def test_evaluator_row(self):
        row = registry().EVALUATOR_ROWS[EVALUATOR_ID]
        self.assertEqual((row["metric"], row["direction"], tuple(row["score_range"]), row["tier"]),
                         (METRIC, "lower_is_better", (0, None), "T3"))
        self.assertFalse(row["binary"])
        self.assertEqual(tuple(row["error_classes"]), ("decode_error", "scorer_error"))
        self.assertEqual(row["payload_text_field"], "text")
        self.assertEqual(tuple(row["item_payload"]), ("text", "reference_normalized", "audio", "audio_sha256"))
        self.assertFalse(row["normalizer"] or row["env_version"])
        self.assertEqual(set(row["sampler_keys"]), set(SAMPLER))
        self.assertEqual(row["scorer"], SCORER)
        self.assertEqual(row["samples"], "500, 1")
        self.assertIsNone(row.get("construct_aligned"))

    def test_shared_annotation_declared(self):
        """R16-1: the references share the training labels' annotation process; the manifest (registry row) says so
        and states what a pass shows."""
        sa = registry().EVALUATOR_ROWS[EVALUATOR_ID]["shared_annotation"]
        self.assertIn("AMI", sa["convention"])
        self.assertIn("R16-1", sa["statement"])
        self.assertIn("convention", sa["statement"])
        self.assertIn("not a general", sa["statement"])

    def test_disclosures_table(self):
        """One mechanism for the registration statements of 05 section 6.1: each names its registration field, the
        registry row key that requires it and the row's fact the report prints."""
        d = {x["field"]: x for x in registry().DISCLOSURES}
        self.assertEqual(set(d), {"construct_alignment", "shared_annotation"})
        self.assertEqual((d["construct_alignment"]["row_key"], d["construct_alignment"]["fact"]),
                         ("construct_aligned", "reward"))
        self.assertEqual((d["shared_annotation"]["row_key"], d["shared_annotation"]["fact"]),
                         ("shared_annotation", "convention"))
        from gatedtrain.registration import OPTIONAL_KEYS
        self.assertIn("shared_annotation", OPTIONAL_KEYS)

    def test_identity_hash(self):
        self.assertEqual(registry().evaluator_identity_hash(EVALUATOR_ID),
                         sha_of({"evaluator": EVALUATOR_ID, "metric": METRIC, "direction": "lower_is_better",
                                 "scorer": SCORER}))

    def test_real_item_and_training_rows(self):
        """Three real rows of items.jsonl and train.jsonl: the payload fields are strings; both carry `text`."""
        from gatedtrain.ext.collect import read_item_set
        row = registry().EVALUATOR_ROWS[EVALUATOR_ID]
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-whisper-items-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copyfile(WHISPER / "items.sample.jsonl", tmp / "items.jsonl")
        rows, bad = read_item_set(tmp, tuple(row["item_payload"]))
        self.assertEqual((len(rows), bad), (3, []))
        for t in read_jsonl(WHISPER / "train.sample.jsonl"):
            self.assertIsInstance(t["text"], str)
            self.assertIsInstance(t["utt_id"], str)


# ---------------------------------------------------------------------------
# EXT-139: cuts and completeness (MM-20)
# ---------------------------------------------------------------------------

class TestCuts(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-whisper-cuts-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = build_base(self.tmp / "base")
        self.out = self.tmp / "run"
        for s in (10, 20, 30):
            build_cut(self.out, s, examples_consumed=s * 8, base=self.base, script_sha256="sha256:" + "1" * 64)

    def found(self):
        complete, incomplete = adapter().find_cuts(self.out)
        return [s for s, _ in complete], sorted(Path(p).name for p in incomplete)

    def test_complete_by_step(self):
        self.assertEqual(self.found(), ([10, 20, 30], []))

    def test_partial_and_trainer_directories_ignored(self):
        """`.checkpoint-<S>.partial/` is never a cut; neither is the Trainer's `hf-trainer/` output directory (empty
        or not)."""
        shutil.copytree(self.out / "checkpoint-30", self.out / ".checkpoint-40.partial")
        (self.out / "hf-trainer" / "checkpoint-5").mkdir(parents=True)
        self.assertEqual(self.found(), ([10, 20, 30], []))

    def test_incomplete_cuts(self):
        def edit(d, fn):
            doc = json.loads((d / "cut.json").read_text()); fn(doc)
            (d / "cut.json").write_text(json.dumps(doc))
        def no_marker(d): (d / "cut.json").unlink()
        def changed(d): (d / "config.json").write_text("{}")
        def wrong_step(d): edit(d, lambda doc: doc.update(global_step=21))
        def extra(d): (d / "optimizer.pt").write_bytes(b"x")
        def missing(d): (d / "trainer_state.json").unlink()
        def unlisted(d): edit(d, lambda doc: doc["files"].pop("decoding.json"))
        def deployable(d): edit(d, lambda doc: doc.update(deployable=["model.safetensors", "config.json"]))
        def no_examples(d): edit(d, lambda doc: doc.pop("examples_consumed"))
        def zero_examples(d): edit(d, lambda doc: doc.update(examples_consumed=0))
        def truncated(d):
            (d / "model.safetensors").write_bytes((d / "model.safetensors").read_bytes()[:-4])
            edit(d, lambda doc: doc["files"].update({"model.safetensors": file_sha256(d / "model.safetensors")}))
        def linked(d):
            (d / "tokenizer.json").unlink()
            (d / "tokenizer.json").symlink_to(self.base / "tokenizer.json")
            edit(d, lambda doc: doc["files"].update({"tokenizer.json": file_sha256(d / "tokenizer.json")}))
        def marker_garbage(d): (d / "cut.json").write_text("{not json")
        for fn in (no_marker, changed, wrong_step, extra, missing, unlisted, deployable, no_examples, zero_examples,
                   truncated, linked, marker_garbage):
            with self.subTest(case=fn.__name__):
                shutil.rmtree(self.out / "checkpoint-20")
                build_cut(self.out, 20, examples_consumed=160, base=self.base, script_sha256="sha256:" + "1" * 64)
                fn(self.out / "checkpoint-20")
                self.assertEqual(self.found(), ([10, 30], ["checkpoint-20"]))

    def test_model_directories(self):
        """The plain whisper-base.en directory and a complete cut are model directories; a truncated weight, a base
        without its feature-extractor config and an empty directory are not."""
        a = adapter()
        self.assertIsNone(a.model_dir_problem(self.base))
        self.assertIsNone(a.model_dir_problem(self.out / "checkpoint-30"))
        (self.tmp / "empty").mkdir()
        self.assertIsNotNone(a.model_dir_problem(self.tmp / "empty"))
        other = build_base(self.tmp / "no-preprocessor")
        (other / "preprocessor_config.json").unlink()
        self.assertIsNotNone(a.model_dir_problem(other))
        w = self.base / "model.safetensors"
        w.write_bytes(w.read_bytes()[:-3])
        self.assertIsNotNone(a.model_dir_problem(self.base))


# ---------------------------------------------------------------------------
# EXT-140: identity (MM-17)
# ---------------------------------------------------------------------------

class TestIdentity(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-whisper-id-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.base = build_base(self.tmp / "base")
        self.cut = build_cut(self.tmp / "run", 10, examples_consumed=80, base=self.base,
                             script_sha256="sha256:" + "1" * 64)

    def test_base_alone(self):
        """The baseline arm evaluates the base: its model hash is the manifest hash of all 11 files."""
        ident = adapter().identity(self.base)
        self.assertEqual(ident["model_hash"], manifest_hash(self.base))
        self.assertEqual(ident["cut_hash"], manifest_hash(self.base))
        self.assertEqual(sorted(ident["components"]), sorted(BASE_FILES))
        self.assertIsNone(ident["base_ref"])

    def test_full_weight_cut(self):
        """A cut: model hash over cut.json `deployable` (7 files), no composition with the base (full weights)."""
        ident = adapter().identity(self.cut)
        self.assertEqual(ident["model_hash"], cut_model_hash(self.cut))
        self.assertEqual(ident["cut_hash"], manifest_hash(self.cut))
        self.assertEqual(sorted(ident["components"]), sorted(DEPLOYABLE))
        self.assertIsNone(ident["base_ref"])
        self.assertNotIn("adapter_hash", ident)
        other = build_base(self.tmp / "other", tag="other")
        self.assertEqual(adapter().identity(self.cut, other)["model_hash"], ident["model_hash"])

    def test_components(self):
        before = adapter().identity(self.cut)["model_hash"]
        for name in ("trainer_state.json", "base_ref.json"):
            with self.subTest(changed=name):
                (self.cut / name).write_text('{"other": 1}\n')
                self.assertEqual(adapter().identity(self.cut)["model_hash"], before)
        (self.cut / "decoding.json").write_text('{"num_beams": 5}\n')     # the deployment decoder is a component
        self.assertNotEqual(adapter().identity(self.cut)["model_hash"], before)

    def test_base_walked_once(self):
        """The base (290 MB) is walked once per adapter instance: ext-arm-fill hashes it as the checkpoint and again
        as `--base`."""
        from unittest import mock
        import gatedtrain_recipe_whisper_sft_ami.adapter as mod
        a = adapter()
        with mock.patch.object(mod, "manifest_entries", side_effect=mod.manifest_entries) as walk:
            a.identity(self.base)
            a.identity(self.base, self.base)
            a.identity(self.cut, self.base)
        walked = [Path(c.args[0]).resolve() for c in walk.call_args_list]
        self.assertEqual(walked.count(self.base.resolve()), 1, walked)


# ---------------------------------------------------------------------------
# EXT-148: the whole run collected (and EXT-140, EXT-143 outputs)
# ---------------------------------------------------------------------------

class TestCollectPass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-whisper-pass-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = WhisperScenario(cls.tmp / "sc")
        cls.proc = cls.sc.collect(cls.tmp / "runs")
        cls.run_dir = parse_run_dir(cls.proc.stdout) if cls.proc.returncode == EXIT_OK else None

    def setUp(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))

    def report(self) -> dict:
        return json.loads((self.run_dir / "report" / "gate_report.json").read_text(encoding="utf-8"))

    def text(self) -> str:
        return (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")

    def test_pass_and_hashes(self):
        rep = self.report()
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual((rep["adapter"], rep["independence_tier"], rep["direction"]),
                         (ADAPTER, "T3", "lower_is_better"))
        self.assertEqual(rep["baseline_model_hash"], self.sc.base_model)
        self.assertEqual(rep["candidate_model_hash"], self.sc.cand_model)
        self.assertEqual(rep["contamination"], {"exact_overlap": 0, "allowed": False})
        self.assertEqual(rep["consumption_precision"], "row")
        self.assertEqual((rep["n_items"], rep["samples_per_item"]), (60, 1))
        self.assertGreater(rep["delta_hat"], 0)                          # oriented: lower WER is better (MM-8)

    def test_shared_annotation_reported(self):
        """R16-1: the report states the shared annotation convention: the registration's statement, and the
        manifest's beside it when they differ."""
        rep = self.report()
        sa = rep["shared_annotation"]
        reg = yaml.safe_load(self.sc.paths["registration"].read_text(encoding="utf-8"))
        row = registry().EVALUATOR_ROWS[EVALUATOR_ID]["shared_annotation"]
        self.assertEqual(sa["statement"], reg["shared_annotation"])
        self.assertEqual(sa["manifest_statement"], row["statement"])
        self.assertEqual(sa["convention"], row["convention"])
        text = self.text()
        self.assertIn("shared_annotation: The references of wer are held-out AMI IHM test transcripts", text)
        self.assertIn("shared_annotation_manifest: ", text)
        self.assertNotIn("construct_alignment", rep)

    def test_gpu_accounting(self):
        rep = self.report()
        self.assertEqual((rep["gpu_seconds"], rep["wall_seconds"]), (3600, 3600))

    def test_ledger_rows(self):
        """EXT-143: one observed `row` record per samples.jsonl line of the steps up to the last cut; data_ref.row =
        the line's row = sha256(utt_id) in shard train.jsonl of the data snapshot; step = the optimizer step;
        sample_id = H(run, attempt, epoch, step, utt_id); no seed inputs, rewards or sampler (SFT)."""
        import pyarrow.parquet as pq
        rows = pq.read_table(self.run_dir / "ledger" / "consumption.parquet").to_pylist()
        samples = read_jsonl(self.sc.stage_out / "samples.jsonl")
        self.assertEqual(len(rows), len(samples))
        self.assertEqual(len(rows), 2 * N_TRAIN + 4 * 8)                 # 30 steps: two epochs of 100, then 4 x 8
        snapshot = manifest_hash(self.sc.data)
        for r, s in zip(rows, samples):
            self.assertEqual((r["precision"], r["derivation"], r["outcome"]), ("row", "observed", "consumed"))
            self.assertEqual((r["data_ref"]["snapshot"], r["data_ref"]["shard"]), (snapshot, "train.jsonl"))
            self.assertEqual(r["data_ref"]["row"], sha256_text(s["utt_id"].encode()))
            self.assertEqual(r["step"], s["step"])
            self.assertIsNone(r["seed_inputs"])
            self.assertIsNone(r["rewards"])
            self.assertIsNone(r["chunk"])
            key = "\x00".join(str(x) for x in (r["run_id"], r["attempt_id"], s["epoch"], s["step"], s["utt_id"]))
            self.assertEqual(r["sample_id"], "sha256:" + hashlib.sha256(key.encode()).hexdigest())
        self.assertEqual(len({r["sample_id"] for r in rows}), len(rows))

    def test_lineage(self):
        """Full weights (MM-17): the training stage `used` the base (manifest hash = model hash) and generated each
        cut at its optimizer step with the hash of its deployable files; no `lora_compose` activity."""
        edges = json.loads((self.run_dir / "lineage" / "edges.json").read_text(encoding="utf-8"))
        train = [a for a in edges["activities"] if a["kind"] == "training_stage"][0]
        model = [u for u in train["used"] if u["role"] == "model"][0]
        self.assertEqual((model["hash"], model["model_hash"]), (self.sc.base_model, self.sc.base_model))
        self.assertEqual([g["last_step_included"] for g in train["generated"]], [10, 20, 30])
        self.assertEqual([g["model_hash"] for g in train["generated"]], [cut_model_hash(c) for c in self.sc.cuts])
        self.assertEqual([a for a in edges["activities"] if a["kind"] == "lora_compose"], [])

    def test_impact_on_the_shard(self):
        """A `row` path answers "consumed" (05 section 11.2); every cut is returned from step 1."""
        proc = run_cli(["impact", str(self.sc.data / "train.jsonl"), "--run-dir", str(self.run_dir), "--json"],
                       cwd=self.sc.root)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        rows = json.loads(proc.stdout)
        rows = rows["results"] if isinstance(rows, dict) else rows
        self.assertEqual([(r["last_step_included"], r["first_step"], r["precision"], r["answer"]) for r in rows],
                         [(10, 1, "row", "consumed"), (20, 1, "row", "consumed"), (30, 1, "row", "consumed")])

    def test_outputs_copied(self):
        line = read_jsonl(self.run_dir / "eval" / "candidate" / "items.jsonl")[0]
        self.assertEqual(file_sha256(self.run_dir / "eval" / "candidate" / line["output_ref"]["path"]),
                         line["output_ref"]["sha256"])


# ---------------------------------------------------------------------------
# EXT-141: base binding (MM-18, MM-21)
# ---------------------------------------------------------------------------

class TestBaseBinding(WhisperCase):

    def test_base_ref_files_differ(self):
        """Self-consistent but another base: files, files_sha256 and run_args.json agree with each other."""
        sc = self.fresh()
        def other(d):
            d["files"]["config.json"] = "sha256:" + "0" * 64
            d["files_sha256"] = compact_hash(d["files"])
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", other)
            sc.reseal(cut)
        fs = json.loads((sc.final_cut / "base_ref.json").read_text())["files_sha256"]
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(files_sha256=fs))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="not the SHA-256 of the spec's base")

    def test_files_sha256_recomputed(self):
        """files_sha256 is derived (compact JSON of `files`): a value that is not that hash refuses, even when
        run_args.json repeats it."""
        sc = self.fresh()
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", lambda d: d.update(files_sha256="sha256:" + "9" * 64))
            sc.reseal(cut)
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(files_sha256="sha256:" + "9" * 64))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="files_sha256 is not the hash")

    def test_revision_not_a_commit(self):
        sc = self.fresh()
        for cut in sc.cuts:
            sc.edit_json(cut / "base_ref.json", lambda d: d.update(revision="main"))
            sc.reseal(cut)
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(revision="main"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="40-hex")

    def test_run_args_base_ref_differs(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(repo="openai/whisper-tiny.en"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="run_args.json base_ref")

    def test_base_model_not_the_spec_base(self):
        sc = self.fresh()                                    # both keys agree, on another directory
        sc.edit_json(sc.stage_out / "run_args.json",
                     lambda a: a.update(base_model="/inputs/a4/models/other", model_name_or_path="/inputs/a4/models/other"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="is not, through a read-only mount")

    def test_model_name_or_path_differs(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(model_name_or_path="/inputs/a4/models/other"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="model_name_or_path")

    def test_base_not_verified(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["base_ref"].update(verified=False))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="verified")

    def test_cut_as_the_spec_base_refused(self):
        sc = self.fresh()
        sc.render(spec_mutate=lambda d: d["artifacts"]["base"].update(uri=f"file:{sc.cuts[0]}"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)


# ---------------------------------------------------------------------------
# EXT-142: data binding
# ---------------------------------------------------------------------------

class TestDataBinding(WhisperCase):

    def test_train_file_outside_the_mounts(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(train_file="/elsewhere/train.jsonl"))
        self.refused("EXT-DATA", "artifacts.data.uri", sc=sc, text="train_file")

    def test_train_file_hash_differs(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(train_file_sha256="sha256:" + "1" * 64))
        self.refused("EXT-DATA", "artifacts.data.uri", sc=sc, text="train_file_sha256")

    def test_n_train_recomputed(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(n_train=N_TRAIN + 1))
        self.refused("EXT-DATA", "artifacts.data.uri", sc=sc, text="n_train")

    def test_two_shards_refused(self):
        sc = self.fresh()
        shutil.copyfile(sc.data / "train.jsonl", sc.data / "train-b.jsonl")
        self.refused("EXT-DATA", "artifacts.data.uri", sc=sc, text="shard")


# ---------------------------------------------------------------------------
# EXT-143: observed consumption log (MM-23, MM-27), every structural rule recomputed
# ---------------------------------------------------------------------------

def _row(uid: str) -> str:
    return sha256_text(uid.encode("utf-8"))


class TestConsumptionLog(WhisperCase):

    def test_epoch_shape_of_the_fixture(self):
        """The fixture follows the runtime: 100 rows in batches of 8 give 13 steps per epoch, the 13th carrying 4;
        epochs counted from 0."""
        sc = self.fresh()
        steps = read_jsonl(sc.stage_out / "train_steps.jsonl")
        self.assertEqual([st["n_examples"] for st in steps[11:15]], [8, 4, 8, 8])
        self.assertEqual([st["epoch"] for st in steps[12:14]], [0, 1])
        self.assertEqual(steps[25]["n_examples"], 4)
        self.assertEqual(adapter().consumption_problems(sc.stage_out, sc.data, [(c, d) for c, d in
                                                                                 adapter().find_cuts(sc.stage_out)[0]]),
                         [])

    def refuse_log(self, fn, text):
        sc = self.fresh()
        sc.edit_logs(fn)
        return self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=text)

    @staticmethod
    def swap_id(samples, steps, step, pos, new):
        """Replace the id at (step, pos) in both logs, with its row re-derived, so only the replacement is wrong."""
        old = steps[step - 1]["example_ids"][pos]
        steps[step - 1]["example_ids"][pos] = new
        for x in samples:
            if x["step"] == step and x["utt_id"] == old:
                x.update(utt_id=new, row=_row(new))

    def test_steps_not_consecutive(self):
        def fn(samples, steps):
            for st in steps[3:]:
                st["step"] += 1
            for x in samples:
                if x["step"] >= 4:
                    x["step"] += 1
        self.refuse_log(fn, "consecutive")

    def test_step_count(self):
        """A step applies per_device_train_batch_size examples, except the last step of an epoch (the rest)."""
        def fn(samples, steps):
            gone = steps[1]["example_ids"].pop()
            steps[1]["n_examples"] -= 1
            samples[:] = [x for x in samples if not (x["step"] == 2 and x["utt_id"] == gone)]
        self.refuse_log(fn, "step 2 applies 7 examples, not 8")

    def test_epoch_end_count(self):
        """The 13th step (the end of epoch 0) carries 4 examples; one moved there from step 12 refuses at both."""
        def fn(samples, steps):
            moved = steps[11]["example_ids"].pop()
            steps[11]["n_examples"] -= 1
            steps[12]["example_ids"].append(moved)
            steps[12]["n_examples"] += 1
            for x in samples:
                if x["step"] == 12 and x["utt_id"] == moved:
                    x.update(step=13, position=4)
            samples.sort(key=lambda x: (x["step"], x["position"]))
        hits = self.refuse_log(fn, "step 13 applies 5 examples, not 4")
        self.assertTrue(hits)

    def test_samples_epoch_field(self):
        """samples.jsonl's epoch is checked on its own (train_steps.jsonl left right)."""
        def fn(samples, steps):
            for x in samples:
                if x["step"] == 14:
                    x["epoch"] = 0
        self.refuse_log(fn, "epoch 0, not 1")

    def test_epoch_field(self):
        def fn(samples, steps):
            steps[13]["epoch"] = 0
            for x in samples:
                if x["step"] == 14:
                    x["epoch"] = 0
        self.refuse_log(fn, "epoch 0, not 1")

    def test_unknown_example(self):
        self.refuse_log(lambda samples, steps: self.swap_id(samples, steps, 3, 0, "AMI_XX0000_H00_NOBODY_0000000_0000001"),
                        "not a utt_id")

    def test_example_twice_in_an_epoch(self):
        self.refuse_log(lambda samples, steps: self.swap_id(samples, steps, 2, 0, steps[0]["example_ids"][0]),
                        "twice in epoch 0")

    def test_samples_differ_from_step_ids(self):
        def fn(samples, steps):
            a, b = samples[0]["utt_id"], samples[1]["utt_id"]
            samples[0].update(utt_id=b, row=_row(b))
            samples[1].update(utt_id=a, row=_row(a))
        self.refuse_log(fn, "example_ids")

    def test_row_recomputed(self):
        self.refuse_log(lambda samples, steps: samples[5].update(row="sha256:" + "8" * 64), "sha256(utt_id)")

    def test_position_and_micro_batch(self):
        for field, value, text in (("position", 7, "position"), ("micro_batch", 1, "micro_batch")):
            with self.subTest(field=field):
                self.refuse_log(lambda samples, steps: samples[2].update({field: value}), text)

    def test_attempt(self):
        self.refuse_log(lambda samples, steps: samples[0].update(attempt=2), "attempt")

    def test_n_examples(self):
        self.refuse_log(lambda samples, steps: steps[4].update(n_examples=9), "n_examples")

    def test_step_without_samples_lines(self):
        def fn(samples, steps):
            samples[:] = [x for x in samples if x["step"] != 7]
        self.refuse_log(fn, "step 7")

    def test_gradient_accumulation(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(gradient_accumulation_steps=2))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="gradient_accumulation_steps")

    def test_cut_examples_consumed(self):
        """MM-27: a cut's examples_consumed equals the sum of the logged counts of steps up to its global step."""
        sc = self.fresh()
        cut = sc.cuts[1]
        sc.edit_json(cut / "cut.json", lambda d: d.update(examples_consumed=d["examples_consumed"] + 1))
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="examples_consumed")

    def test_cut_beyond_the_log(self):
        def fn(samples, steps):
            steps[:] = [st for st in steps if st["step"] <= 25]
            samples[:] = [x for x in samples if x["step"] <= 25]
        self.refuse_log(fn, "checkpoint-30")

    def test_missing_logs_refused(self):
        for name in ("samples.jsonl", "train_steps.jsonl"):
            with self.subTest(log=name):
                sc = self.fresh()
                (sc.stage_out / name).unlink()
                self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text=name)

    def test_records_after_the_last_cut_orphaned(self):
        """Records of steps after the given last cut are not consumption of these cuts (MM-27)."""
        sc = self.fresh()
        a = adapter()
        cuts = [(10, sc.cuts[0])]
        self.assertEqual(a.consumption_problems(sc.stage_out, sc.data, cuts), [])
        cons = a.consumption(sc.stage_out, sc.data, cuts, run_id="r")
        self.assertEqual((cons["precision"], cons["derivation"]), ("row", "observed"))
        self.assertEqual(sorted({r["step"] for r in cons["rows"]}), list(range(1, 11)))
        self.assertEqual(len(cons["rows"]), 80)


# ---------------------------------------------------------------------------
# EXT-144: exact overlap (MM-11) on `text`
# ---------------------------------------------------------------------------

class TestOverlap(WhisperCase):

    def test_overlap_refused(self):
        sc = self.fresh(overlap=2)
        self.refused("EXT-CONTAMINATION", "registration.contamination.allow_exact_overlap", sc=sc, text="2 item(s)")

    def test_allowed_overlap_reported(self):
        sc = self.fresh(overlap=2, reg_mutate=lambda r: r["contamination"].update(allow_exact_overlap=True))
        rep = json.loads((self.passes(sc) / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["contamination"], {"exact_overlap": 2, "allowed": True})

    def test_field_is_text(self):
        """The overlap key is the raw AMI `text` on both sides; an equal normalized reference is not an overlap."""
        from gatedtrain.ext.collect import exact_overlap
        sc = self.fresh()
        rows = [dict(sc.item_rows[0], reference_normalized=sc.train_rows[0]["text"].lower()),
                dict(sc.item_rows[1], text=sc.train_rows[1]["text"])]
        self.assertEqual(exact_overlap(rows, sc.data, registry().EVALUATOR_ROWS[EVALUATOR_ID]["payload_text_field"]), 1)


# ---------------------------------------------------------------------------
# EXT-145: seed, pins, profile, last cut, script, decoder
# ---------------------------------------------------------------------------

class TestRunChecks(WhisperCase):

    def test_seed(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: (a.update(seed=1), a["settings"].update(seed=1)))
        self.refused("EXT-LAUNCH", "--launch-record.training_seed", sc=sc, text="run_args.json `seed` is 1")

    def test_settings_seed(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(seed=1))
        self.refused("EXT-LAUNCH", "--launch-record.training_seed", sc=sc, text="settings.seed")

    def test_cut_seed(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(seed=1))
        self.refused("EXT-LAUNCH", "--launch-record.training_seed", sc=sc, text="cut.json")

    def test_framework_version(self):
        sc = self.fresh()
        p = sc.stage_out / "env-freeze.txt"
        p.write_text(p.read_text().replace("transformers==5.21.0", "transformers==5.20.0"))
        self.refused("EXT-FREEZE", "--stage-out", sc=sc)

    def test_pins(self):
        for drop in ("accelerate==", "torch=="):
            with self.subTest(drop=drop):
                sc = self.fresh()
                p = sc.stage_out / "env-freeze.txt"
                p.write_text("".join(l + "\n" for l in p.read_text().splitlines() if not l.startswith(drop)))
                self.refused("EXT-FREEZE", "--stage-out", sc=sc, text=drop[:-2])

    def test_profile(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(profile="smoke"))
        self.refused("EXT-PROFILE", "--launch-record.profile", sc=sc)
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "cut.json", lambda d: d.update(profile="smoke"))
        self.refused("EXT-PROFILE", "--launch-record.profile", sc=sc, text="cut.json")

    def test_last_cut_is_max_steps(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: (a.update(max_steps=40),
                                                                a["settings"].update(max_steps=40)))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="max_steps")

    def test_settings_max_steps(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a["settings"].update(max_steps=40))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="settings.max_steps")

    def test_dry_run(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(dry_run=True))
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="dry_run")

    def test_script(self):
        for where in ("env-runtime.json", "cut.json", "run_args.json"):
            with self.subTest(where=where):
                sc = self.fresh()
                path = {"env-runtime.json": sc.stage_out / "env-runtime.json", "cut.json": sc.final_cut / "cut.json",
                        "run_args.json": sc.stage_out / "run_args.json"}[where]
                sc.edit_json(path, lambda d: d.update(script_sha256="sha256:" + "2" * 64))
                self.refused("EXT-LAUNCH", "--launch-record.script_sha256", sc=sc)

    def test_decoder_code(self):
        """cut.json, env-runtime.json and run_args.json carry the same decoder code hash, which is the evaluator's
        (eval_asr.py's SHARED DECODER block)."""
        for where in ("cut", "runtime", "all"):
            with self.subTest(where=where):
                sc = self.fresh()
                other = "sha256:" + "3" * 64
                if where in ("cut", "all"):
                    for cut in sc.cuts:
                        sc.edit_json(cut / "cut.json", lambda d: d.update(decoder_code_sha256=other))
                if where in ("runtime", "all"):
                    sc.edit_json(sc.stage_out / "env-runtime.json", lambda d: d.update(decoder_code_sha256=other))
                if where == "all":
                    sc.edit_json(sc.stage_out / "run_args.json", lambda d: d.update(decoder_code_sha256=other))
                self.refused("EXT-SAMPLER", "--stage-out", sc=sc, text="decoder_code_sha256")

    def test_decoding_of_the_cut(self):
        sc = self.fresh()
        sc.edit_json(sc.final_cut / "decoding.json", lambda d: d.update(max_new_tokens=100))
        sc.reseal(sc.final_cut)
        self.refused("EXT-SAMPLER", "--stage-out", sc=sc, text="decoding.json")

    def test_decoding_sha256_recomputed(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda a: a.update(decoding_sha256="sha256:" + "5" * 64))
        self.refused("EXT-SAMPLER", "--stage-out", sc=sc, text="decoding_sha256")

    def test_registered_decoder(self):
        """Every cut's decoding.json is the registered sampler without its evaluation-only keys."""
        sc = self.fresh(reg_mutate=lambda r: r["sampler"].update(max_new_tokens=100))
        self.refused("EXT-SAMPLER", "registration.sampler", sc=sc)


# ---------------------------------------------------------------------------
# EXT-146: ext-arm-fill on eval_asr.py's arm-wer/
# ---------------------------------------------------------------------------

class TestArmFill(WhisperCase):

    def meta(self, arm: Path) -> dict:
        return json.loads((arm / "eval_meta.json").read_text(encoding="utf-8"))

    def test_baseline_arm(self):
        """The base: model_hash = its manifest hash, item_set_hash filled; the evaluator's compact-JSON sampler hash
        equals the RFC 8785 hash for this sampler (integers and strings only), so it is neither filled nor
        converted; base_model_hash stays null (no --base, full weights)."""
        sc = self.fresh(filled=False)
        arm = sc.baseline
        self.assertEqual(compact_hash(SAMPLER), sha_of(SAMPLER))
        items_before = (arm / "items.jsonl").read_bytes()
        proc = sc.fill(arm, sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        meta = self.meta(arm)
        self.assertEqual(meta["checkpoint"], {"model_hash": sc.base_model, "base_model_hash": None})
        self.assertEqual((meta["item_set_hash"], meta["sampler_hash"]), (sc.item_set_hash, sha_of(SAMPLER)))
        self.assertNotIn("converted", proc.stdout)
        self.assertEqual((arm / "items.jsonl").read_bytes(), items_before)
        once = (arm / "eval_meta.json").read_bytes()
        proc = sc.fill(arm, sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual((arm / "eval_meta.json").read_bytes(), once)

    def test_candidate_arm(self):
        """A full-weight cut needs no --base; with --base, base_model_hash is filled with the base's model hash."""
        sc = self.fresh(filled=False)
        proc = sc.fill(sc.candidate, sc.final_cut)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(self.meta(sc.candidate)["checkpoint"], {"model_hash": sc.cand_model, "base_model_hash": None})
        sc = self.fresh(filled=False)
        proc = sc.fill(sc.candidate, sc.final_cut, base=sc.base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(self.meta(sc.candidate)["checkpoint"], {"model_hash": sc.cand_model,
                                                                 "base_model_hash": sc.base_model})

    def test_other_cut_refused(self):
        sc = self.fresh()
        before = (sc.candidate / "eval_meta.json").read_bytes()
        proc = sc.fill(sc.candidate, sc.cuts[0])
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("ARM.checkpoint.model_hash", [e["field"] for e in parse_errors(proc.stderr)])
        self.assertEqual((sc.candidate / "eval_meta.json").read_bytes(), before)

    def test_fill_then_collect(self):
        sc = self.fresh(filled=False)
        for arm, ck in ((sc.baseline, sc.base), (sc.candidate, sc.final_cut)):
            proc = sc.fill(arm, ck)
            self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.passes(sc)


# ---------------------------------------------------------------------------
# EXT-147: registration (sampler keys, shared annotation, the per-item mean)
# ---------------------------------------------------------------------------

class TestRegistration(unittest.TestCase):

    @staticmethod
    def reg() -> dict:
        text = (WHISPER / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", "sha256:" + "a" * 64).replace("{sampler}", json.dumps(SAMPLER))
        return yaml.safe_load(text.replace("{evaluator_hash}", "sha256:" + "b" * 64))

    def fields(self, reg: dict, evaluator: str = EVALUATOR_ID) -> list[str]:
        from gatedtrain.registration import check_contents
        return [p.field for p in check_contents(reg, evaluator_id=evaluator)]

    def test_template_accepted(self):
        self.assertEqual(self.fields(self.reg()), [])

    def test_sampler_keys(self):
        reg = self.reg()
        reg["sampler"]["temperature"] = 0.0
        del reg["sampler"]["num_beams"]
        f = self.fields(reg)
        self.assertIn("registration.sampler.temperature", f)
        self.assertIn("registration.sampler.num_beams", f)

    def test_shared_annotation_required(self):
        reg = self.reg()
        del reg["shared_annotation"]
        self.assertIn("registration.shared_annotation", self.fields(reg))
        reg["shared_annotation"] = " \n"
        self.assertIn("registration.shared_annotation", self.fields(reg))

    def test_construct_alignment_refused_for_wer(self):
        reg = self.reg()
        reg["construct_alignment"] = "wer measures what the training loss measures."
        self.assertIn("registration.construct_alignment", self.fields(reg))

    def test_corpus_wer_not_a_deciding_estimator(self):
        """Corpus WER is descriptive only (decision; MM-GAP-13): the evaluator declares the per-item mean alone, so a
        ratio-of-sums clause refuses."""
        from gatedtrain.items import EVALUATORS
        self.assertEqual(tuple(EVALUATORS[EVALUATOR_ID]["estimators"]), ("mean",))
        reg = self.reg()
        reg["clause"]["estimator"] = "ratio_of_sums"
        self.assertTrue([f for f in self.fields(reg) if f.startswith("registration.clause.estimator")],
                        self.fields(reg))

    def test_collect_refuses_without_it(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-whisper-reg-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        sc = WhisperScenario(tmp / "sc", n_items=12, reg_mutate=lambda r: r.pop("shared_annotation"))
        proc = sc.collect(tmp / "runs")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("registration.shared_annotation", [e["field"] for e in parse_errors(proc.stderr)])

    def test_same_statement_printed_once(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-whisper-stmt-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        row = registry().EVALUATOR_ROWS[EVALUATOR_ID]["shared_annotation"]["statement"]
        sc = WhisperScenario(tmp / "sc", n_items=12, reg_mutate=lambda r: r.update(shared_annotation=row))
        proc = sc.collect(tmp / "runs")
        self.assertIn(proc.returncode, (EXIT_OK, 1), describe(proc))
        text = (parse_run_dir(proc.stdout) / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn(f"shared_annotation: {row}\n", text)
        self.assertNotIn("shared_annotation_manifest: ", text)


# ---------------------------------------------------------------------------
# EXT-149: host rows and launch; EXT-151: smoke on `wer`
# ---------------------------------------------------------------------------

class TestLaunch(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-whisper-launch-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.sc = WhisperScenario(self.tmp / "sc", n_items=12)
        self.out = self.sc.output_root / "area4-whisper" / "full-2"
        self.out.mkdir()

    def refused(self, proc, out, rule, field):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue([e for e in errs if e["rule"] == rule and e["field"] == field], f"{rule} {field}: {errs}")
        self.assertFalse(out.exists())

    def test_gpu_request_accepted(self):
        proc, out = self.sc.launch_request(self.sc.request(self.out), "ok")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("--runtime=nvidia", proc.stdout)
        self.assertIn("GATEDTRAIN_GPU_STEP_EST_H=0.03 ", proc.stdout)
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((rec["adapter"], rec["framework"]["name"]), (ADAPTER, "transformers"))

    def test_other_env_name_refused(self):
        req = self.sc.request(self.out)
        req["env"]["HF_HOME"] = "/tmp/hf"
        proc, out = self.sc.launch_request(req, "env")
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.HF_HOME")

    def test_spec_optional_and_checked(self):
        proc, out = self.sc.launch_request(self.sc.request(self.out), "spec", extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        req = self.sc.request(self.out)
        req["mounts"] = [m for m in req["mounts"] if m["mode"] == "rw"]
        proc, out = self.sc.launch_request(req, "nospec", extra=["--spec", str(self.sc.paths["spec"])])
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        fields = [e["field"] for e in parse_errors(proc.stderr)]
        self.assertIn("artifacts.base.uri", fields)
        self.assertIn("artifacts.data.uri", fields)

    def test_registry_host_rows(self):
        reg = registry()
        from gatedtrain.ext.rowkit import python_tree
        from gatedtrain_recipe_whisper_sft_ami.rows import OUT_NAME, RUNTIME
        tree = python_tree(RUNTIME)
        self.assertEqual(reg.input_roots(ADAPTER), ((str(Path(tree).parent),) if tree else ())
                         + (RUNTIME, "{output_root}/" + OUT_NAME))
        self.assertEqual(set(reg.env_names(ADAPTER)), set(ENV_NAMES))
        self.assertEqual(reg.env_paths(ADAPTER), ("HOME", "XDG_CACHE_HOME"))
        self.assertEqual(reg.scan_skip(ADAPTER)[0], f"{RUNTIME}/.venv")
        self.assertEqual(reg.interpreter(ADAPTER), f"{RUNTIME}/.venv/bin/python")
        self.assertEqual(reg.interpreter_trees(ADAPTER), (tree,) if tree else ())

    def test_full_run_smoke_on_wer(self):
        """EXT-151 (MM-38): the smoke is judged on `wer` ([0, null]: no mean rule; the mode and sd rules apply)."""
        good = write_canonical(self.tmp / "smoke.json", smoke_record(base_model=self.sc.base_model,
                                                                     item_set=self.sc.item_set_hash))
        proc, out = self.sc.launch_request(self.sc.request(self.out), "full", profile="full", smoke=good)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("GATEDTRAIN_GPU_STEP_EST_H=0.1 ", proc.stdout)
        flat = write_canonical(self.tmp / "smoke-flat.json", smoke_record(
            base_model=self.sc.base_model, item_set=self.sc.item_set_hash, baseline_scores=[0.0] * 10 + [0.2] * 10))
        proc, out = self.sc.launch_request(self.sc.request(self.out), "flat", profile="full", smoke=flat)
        self.refused(proc, out, "EXT-SMOKE", "--smoke-record")
        self.assertIn("degenerate_mode", proc.stderr)


# ---------------------------------------------------------------------------
# EXT-150: launch-side argv checks
# ---------------------------------------------------------------------------

class TestArgv(unittest.TestCase):

    RT = "/srv/recipes/whisper-sft-ami/runtime"

    def test_eval_base_default_only(self):
        """eval_asr.py loads `--base` (with --base-only) without hashing it: only the script's default
        `<script dir>/models/whisper-base.en` is allowed."""
        a, rt = adapter(), self.RT
        ev = [f"{rt}/.venv/bin/python", f"{rt}/eval_asr.py", "--base-only", "--items", f"{rt}/items", "--out", "/out"]
        self.assertEqual(a.argv_problems(ev), [])
        self.assertEqual(a.argv_problems(ev + ["--base", f"{rt}/models/whisper-base.en"]), [])
        for bad in (ev + ["--base", "/inputs/other"], ev + ["--base=/inputs/other"],
                    ev + ["--base", f"{rt}/models/whisper-base.en/../whisper-tiny.en"], ev + ["--base"]):
            with self.subTest(argv=bad[-1]):
                self.assertTrue(a.argv_problems(bad))

    def test_training_flags(self):
        """train_whisper.py: `--no-verify-base` is refused (collect refuses its run, EXT-141), and the profile is
        smoke or full (the host CPU profiles are never collected)."""
        a, rt = adapter(), self.RT
        tr = [f"{rt}/.venv/bin/python", f"{rt}/train_whisper.py", "--profile", "full", "--out", "/out", "--seed", "4000"]
        self.assertEqual(a.argv_problems(tr), [])
        self.assertEqual(a.argv_problems(tr + ["--max-steps", "300", "--base", "/inputs/x"]), [])
        self.assertEqual(a.argv_problems(tr[:2] + ["--profile=smoke", "--out", "/out"]), [])
        for bad in (tr + ["--no-verify-base"], tr[:3] + ["cpu-sanity"] + tr[4:], tr[:2] + ["--profile=cpu-pilot"],
                    tr + ["--no-verify"], tr[:2] + ["--prof", "cpu-sanity"]):     # argparse's unique prefixes too
            with self.subTest(argv=bad):
                self.assertTrue(a.argv_problems(bad))
        ev = [f"{rt}/.venv/bin/python", f"{rt}/eval_asr.py", "--checkpoint", "/inputs/cut", "--out", "/out"]
        self.assertEqual(a.argv_problems(ev + ["--bas", "/inputs/x"]), [])   # ambiguous prefix: argparse refuses it

    def test_launch_refuses(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-whisper-argv-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        sc = WhisperScenario(tmp / "sc", n_items=12)
        out = sc.output_root / "area4-whisper" / "eval-1"
        out.mkdir()
        (sc.runtime / "models" / "other").mkdir()
        ev = [f"{RUNTIME_CONTAINER}/.venv/bin/python", f"{RUNTIME_CONTAINER}/eval_asr.py", "--base-only", "--items",
              f"{RUNTIME_CONTAINER}/items", "--out", "/out", "--eval-seeds", "11"]
        proc, rec = sc.launch_request(sc.request(out, argv=ev), "eval-ok")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        for name, argv in (("eval-other", ev + ["--base", f"{RUNTIME_CONTAINER}/models/other"]),
                           ("train-noverify", sc.request(out)["argv"] + ["--no-verify-base"])):
            with self.subTest(case=name):
                proc, rec = sc.launch_request(sc.request(out, argv=argv), name)
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                self.assertIn("--request.argv", [e["field"] for e in parse_errors(proc.stderr)])
                self.assertFalse(rec.exists())


# ---------------------------------------------------------------------------
# EXT-152: an unhashable base refuses (exit 3), never a traceback
# ---------------------------------------------------------------------------

class TestUnhashableBase(WhisperCase):

    def test_symlink_in_base(self):
        sc = self.fresh()
        (sc.base / "extra.json").symlink_to(sc.base / "config.json")
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)

    def test_missing_base(self):
        sc = self.fresh()
        sc.render(spec_mutate=lambda d: d["artifacts"]["base"].update(uri=f"file:{sc.root / 'no-such-base'}"))
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)

    def test_unreadable_base_file(self):
        sc = self.fresh()
        p = sc.base / "vocab.json"
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)
        try:
            p.read_bytes()
            self.skipTest("the file stays readable (running as root)")
        except PermissionError:
            pass
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc)


# ---------------------------------------------------------------------------
# EXT-153: facts of the real runtime (read only; skipped where the runtime is absent)
# ---------------------------------------------------------------------------

@unittest.skipUnless(EVAL_ASR.is_file() and TRAIN_WHISPER.is_file(), "the area-4 runtime is absent")
class TestRuntimeFacts(unittest.TestCase):

    def test_scorer_code_hash_is_eval_asr(self):
        self.assertEqual(file_sha256(EVAL_ASR), EVAL_ASR_SHA256)

    def test_decoder_block(self):
        from gatedtrain_recipe_whisper_sft_ami import adapter as hf_seq2seq
        self.assertEqual(decoder_block_sha256(EVAL_ASR), DECODER_CODE_SHA256)
        self.assertEqual(decoder_block_sha256(TRAIN_WHISPER), DECODER_CODE_SHA256)
        self.assertEqual(hf_seq2seq.DECODER_CODE_SHA256, DECODER_CODE_SHA256)

    def test_fixture_decoding_is_the_cut_file(self):
        """The fixture decoding.json is byte-identical to a real cut's (the cpu-sanity cut 4 records its SHA-256)."""
        cut = json.loads((EVIDENCE / "cpu-sanity-cut4.json").read_text(encoding="utf-8"))
        self.assertEqual(file_sha256(WHISPER / "decoding.json"), cut["files"]["decoding.json"])
        self.assertEqual(sorted(cut["files"]), sorted(CUT_FILES))
        self.assertEqual(cut["deployable"], list(DEPLOYABLE))
        self.assertEqual(cut["decoder_code_sha256"], DECODER_CODE_SHA256)

    def test_fixture_sampler_is_the_evaluators(self):
        meta = json.loads((EVIDENCE / "cpu-sanity-base-arm-wer" / "eval_meta.json").read_text())
        self.assertEqual(meta["sampler"], SAMPLER)
        # The evidence was written by eval_asr.py before its re-pins for the framework's renames (RESULT.md): only the
        # code hash differs from the current row.
        self.assertEqual({**meta["scorer"], "code_sha256": SCORER["code_sha256"]}, SCORER)
        self.assertEqual(meta["scorer"]["code_sha256"],
                         "sha256:6703cd48bbc4420fcbfba0dff7bbf97051c98cf861700219aff0c8a16e5f0467")
        self.assertEqual({k: v for k, v in SAMPLER.items() if k not in EVAL_ONLY_KEYS}, DECODING)

    def test_base_files(self):
        doc = json.loads((AREA4 / "models" / "MODELS.json").read_text(encoding="utf-8"))["models"][BASE_NAME]
        self.assertEqual(sorted(doc["files"]), sorted(BASE_FILES))
        self.assertEqual((doc["repo"], doc["revision"]), (REPO, REVISION))
        if not (AREA4 / "models" / BASE_NAME).is_dir():
            self.skipTest("the base is not downloaded (runtime/fetch_models.py)")
        self.assertEqual(sorted(p.name for p in (AREA4 / "models" / BASE_NAME).iterdir()), sorted(BASE_FILES))

    def test_step_log_shape(self):
        lines = read_jsonl(EVIDENCE / "cpu-sanity-train_steps.jsonl")
        self.assertEqual([set(l) for l in lines], [{"step", "epoch", "n_examples", "example_ids", "utc"}] * 4)
        self.assertEqual([(l["step"], l["epoch"], l["n_examples"]) for l in lines], [(1, 0, 4), (2, 0, 4), (3, 0, 4),
                                                                                       (4, 0, 4)])


class TestMakeDataGuard(unittest.TestCase):
    """runtime/make_data.py runs in a fresh checkout (the committed metadata and items present), refuses only when
    generated content exists, and detects a rewritten committed file."""

    def setUp(self):
        import importlib.util
        src = Path(__file__).resolve().parents[1] / "runtime"            # this checkout's committed files
        self.rt = Path(tempfile.mkdtemp(prefix="gatedtrain-whisper-make-data-"))
        self.addCleanup(shutil.rmtree, self.rt, True)
        shutil.copyfile(src / "make_data.py", self.rt / "make_data.py")
        for rel in ("data/data_meta.json", "items/items.jsonl", "items/item_set_meta.json"):
            (self.rt / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, self.rt / rel)
        spec = importlib.util.spec_from_file_location("whisper_make_data", self.rt / "make_data.py")
        self.md = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.md)

    def test_fresh_checkout_passes_the_guard(self):
        # Past the guard, the run stops at the missing download (data-source/DATA.json), not at the committed files
        # (stand-ins for the runtime venv's audio modules, which this environment need not have).
        import sys
        import types
        from unittest import mock
        stubs = {m: types.ModuleType(m) for m in ("numpy", "soundfile") if m not in sys.modules}
        with mock.patch.dict(sys.modules, stubs), self.assertRaises(FileNotFoundError) as cm:
            self.md.main([])
        self.assertIn("DATA.json", str(cm.exception))

    def test_generated_content_refuses(self):
        for rel in ("data/audio", "data/train.jsonl", "items/audio"):
            with self.subTest(rel=rel):
                p = self.rt / rel
                p.mkdir() if rel.endswith("audio") else p.write_text("{}\n")
                with self.assertRaisesRegex(SystemExit, "exists: remove the generated data first"):
                    self.md.main([])
                shutil.rmtree(p) if p.is_dir() else p.unlink()

    def test_changed_committed_file_detected(self):
        before = {t: t.read_bytes() for t in self.md.COMMITTED}
        self.assertEqual(self.md.changed_files(before), [])
        meta = self.rt / "items" / "item_set_meta.json"
        meta.write_bytes(meta.read_bytes() + b" ")
        self.assertEqual(self.md.changed_files(before), [meta])


if __name__ == "__main__":
    unittest.main()
