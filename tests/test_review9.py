"""Review round 9 of the `hf_seq2seq` adapter (EXT-154 to EXT-158), moved from gatedtrain's tests with the recipe:
unreadable files refuse (EXT-154), an arm's references are the registered ones (EXT-155), the evaluation scripts'
inputs are bound at launch (EXT-156), arms covering fewer items are MISSING (EXT-157), a step log longer than the
run refuses (EXT-158).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from gatedtrain.testing.support import (EXIT_OK, EXIT_REFUSED, describe, file_sha256, parse_errors, parse_run_dir,
                                        run_cli)

import _whisper as W

NOT_PASS = 1


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def _root_can_read() -> bool:
    return os.geteuid() == 0


class Case(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-review9-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def path(self, name: str) -> Path:
        type(self).n += 1
        return self.tmp / f"{name}-{type(self).n}"

    def unreadable(self, p: Path) -> None:
        if _root_can_read():
            self.skipTest("running as root: chmod 0 does not make a file unreadable")
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)

    def refused(self, proc, rule: str, field_prefix: str | None = None, text: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix} {text!r}; got {errs}\n{describe(proc)}")
        return hits


class TestUnreadableWhisper(Case):

    def sc(self):
        return W.WhisperScenario(self.path("w"), n_items=12)

    def test_shard(self):
        sc = self.sc()
        self.unreadable(sc.data / "train.jsonl")
        self.refused(sc.collect(self.path("runs")), "SPEC-REF", "artifacts.data.uri")

    def test_last_cut_file(self):
        sc = self.sc()
        self.unreadable(sc.final_cut / "trainer_state.json")
        self.refused(sc.collect(self.path("runs")), "EXT-STAGE", "--stage-out", "cannot be read")

    def test_middle_cut_file(self):
        """A middle cut with an unreadable file is incomplete (reported), never a traceback."""
        sc = self.sc()
        self.unreadable(sc.cuts[0] / "trainer_state.json")
        proc = sc.collect(self.path("runs"))
        self.assertIn(proc.returncode, (EXIT_OK, NOT_PASS), describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        rep = json.loads((parse_run_dir(proc.stdout) / "report" / "gate_report.json").read_text())
        self.assertEqual(rep["incomplete_cuts"], [sc.cuts[0].name])

    def test_freeze(self):
        sc = self.sc()
        self.unreadable(sc.stage_out / "env-freeze.txt")
        self.refused(sc.collect(self.path("runs")), "EXT-FREEZE", "--stage-out")

    def test_checks_on_an_unreadable_shard(self):
        """The adapter's own data check returns EXT-DATA instead of raising (collect calls it even when the data
        manifest failed)."""
        sc = self.sc()
        self.unreadable(sc.data / "train.jsonl")
        launch = json.loads(sc.launch.read_text())
        probs = registry().adapter(W.ADAPTER).checks(last_cut=sc.final_cut, base=sc.base, launch=launch,
                                                     base_field="b", stage_out=sc.stage_out, data=sc.data,
                                                     data_field="d")
        self.assertIn("EXT-DATA", [p.rule for p in probs])


class TestUnreadableOthers(Case):
    """The sweep: trl_grpo through the generic paths (freeze, cut files, stage directory); the sb3_ppo case lives in
    its recipe repository."""

    def test_unlistable_stage_directory(self):
        sc = W.WhisperScenario(self.path("w"), n_items=12)
        if _root_can_read():
            self.skipTest("running as root")
        sc.stage_out.chmod(0)
        self.addCleanup(sc.stage_out.chmod, 0o755)
        self.refused(sc.collect(self.path("runs")), "EXT-STAGE", "--stage-out")


class TestArmReferences(Case):

    def edit_output(self, sc, arm: Path, fn) -> None:
        lines = W.read_jsonl(arm / "items.jsonl")
        ref = lines[0]["output_ref"]
        doc = json.loads((arm / ref["path"]).read_text())
        fn(doc)
        (arm / ref["path"]).unlink()                                     # break the hard link: the arm's own copy
        W.write_json(arm / ref["path"], doc)
        lines[0]["output_ref"]["sha256"] = file_sha256(arm / ref["path"])
        W.write_jsonl(arm / "items.jsonl", lines)

    def test_reference_differs(self):
        """An evaluation that read another `--items` writes that set's normalized reference into its outputs."""
        sc = W.WhisperScenario(self.path("w"), n_items=12)
        self.edit_output(sc, sc.candidate, lambda d: d.update(reference_normalized="an edited reference"))
        self.refused(sc.collect(self.path("runs")), "EXT-ARM", "--eval-candidate", "reference_normalized")

    def test_output_item_differs(self):
        sc = W.WhisperScenario(self.path("w"), n_items=12)
        self.edit_output(sc, sc.baseline, lambda d: d.update(item="another item"))
        self.refused(sc.collect(self.path("runs")), "EXT-ARM", "--eval-baseline", "item")

    def test_output_unreadable(self):
        sc = W.WhisperScenario(self.path("w"), n_items=12)
        ref = W.read_jsonl(sc.candidate / "items.jsonl")[0]["output_ref"]
        p = sc.candidate / ref["path"]
        p.unlink()
        p.write_text("{not json")
        lines = W.read_jsonl(sc.candidate / "items.jsonl")
        lines[0]["output_ref"]["sha256"] = file_sha256(p)
        W.write_jsonl(sc.candidate / "items.jsonl", lines)
        self.refused(sc.collect(self.path("runs")), "EXT-ARM", "--eval-candidate")


class TestEvalArgv(unittest.TestCase):

    A4 = "/srv/recipes/whisper-sft-ami/runtime"

    def problems(self, adapter: str, argv: list, profile: str | None = None) -> list[str]:
        return registry().adapter(adapter).argv_problems(argv, profile=profile)

    def cases(self):
        return (("hf_seq2seq", self.A4, "eval_asr.py", "train_whisper.py", "11"),)

    def test_items_default_only(self):
        for adapter, rt, ev, _, seeds in self.cases():
            with self.subTest(adapter=adapter):
                base = [f"{rt}/.venv/bin/python", f"{rt}/{ev}", "--base-only", "--out", "/out", "--eval-seeds", seeds]
                self.assertEqual(self.problems(adapter, base), [])
                self.assertEqual(self.problems(adapter, base + ["--items", f"{rt}/items"]), [])
                for bad in (["--items", "/inputs/items-copy"], ["--items=/inputs/items-copy"],
                            ["--items", f"{rt}/items/../items-copy"]):
                    self.assertTrue(self.problems(adapter, base + bad), bad)

    def test_script_pinned(self):
        """argv[1] is one of the adapter's scripts in the runtime directory argv[0] (the pinned interpreter) names:
        a byte-identical copy elsewhere (another input root) brings its own base and items."""
        for adapter, rt, ev, tr, _ in self.cases():
            with self.subTest(adapter=adapter):
                py = f"{rt}/.venv/bin/python"
                self.assertEqual(self.problems(adapter, [py, f"{rt}/{tr}", "--profile", "smoke", "--out", "/out"]), [])
                for bad in (f"/srv/out/x/{ev}", f"{rt}/make_data.py", f"{rt}/sub/{ev}"):
                    self.assertTrue(self.problems(adapter, [py, bad, "--out", "/out"]), bad)

    def test_limit_refused_for_full(self):
        for adapter, rt, ev, _, seeds in self.cases():
            with self.subTest(adapter=adapter):
                argv = [f"{rt}/.venv/bin/python", f"{rt}/{ev}", "--base-only", "--out", "/out", "--limit", "20"]
                self.assertEqual(self.problems(adapter, argv, "smoke"), [])
                self.assertTrue(self.problems(adapter, argv, "full"))
                self.assertTrue(self.problems(adapter, argv[:-2] + ["--limit=20"], "full"))
                self.assertEqual(self.problems(adapter, argv[:-2], "full"), [])

    def test_ambiguous_prefix(self):
        """`--ba` and `--bas` are prefixes of --base, --base-only (and --base-model-hash): argparse refuses them as
        ambiguous, so they are not a way around the --base rule; the resolver leaves them alone."""
        argv = [f"{self.A4}/.venv/bin/python", f"{self.A4}/eval_asr.py", "--checkpoint", "/inputs/cut", "--out", "/out"]
        for p in ("--ba", "--bas", "--ite"):            # --ite: --items and --item-set-hash
            self.assertEqual(self.problems("hf_seq2seq", argv + [p, "/inputs/x"]), [])

    def test_launch_and_recheck_refuse(self):
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-review9-launch-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        sc = W.WhisperScenario(tmp / "sc", n_items=12)
        out = sc.output_root / "area4-whisper" / "eval-1"
        out.mkdir()
        rc = W.RUNTIME_CONTAINER
        ev = [f"{rc}/.venv/bin/python", f"{rc}/eval_asr.py", "--base-only", "--items", f"{rc}/items", "--out", "/out",
              "--eval-seeds", "11", "--limit", "20"]
        smoke = W.write_canonical(tmp / "smoke.json", W.smoke_record(base_model=sc.base_model,
                                                                     item_set=sc.item_set_hash))
        proc, rec = sc.launch_request(sc.request(out, argv=ev), "smoke-ok")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        proc2 = run_cli(["ext-launch", "--adapter", W.ADAPTER, "--recheck", str(rec)], cwd=sc.root, env=sc.env())
        self.assertEqual(proc2.returncode, EXIT_OK, describe(proc2))
        proc, rec = sc.launch_request(sc.request(out, argv=ev), "full-limit", profile="full", smoke=smoke)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("--request.argv", [e["field"] for e in parse_errors(proc.stderr)])
        self.assertFalse(rec.exists())


class TestCoverage(Case):

    def test_both_arms_limited(self):
        """Expected items are the registered set x K per arm; an item with no record is missing in that arm; above
        `max_unpaired` (0) the clause is MISSING, even when both arms skip the same items."""
        sc = W.WhisperScenario(self.path("w"), n_items=60)
        for arm in (sc.baseline, sc.candidate):
            lines = W.read_jsonl(arm / "items.jsonl")[:50]
            W.write_jsonl(arm / "items.jsonl", lines)
        proc = sc.collect(self.path("runs"))
        self.assertEqual(proc.returncode, NOT_PASS, describe(proc))
        rep = json.loads((parse_run_dir(proc.stdout) / "report" / "gate_report.json").read_text())
        self.assertEqual(rep["status"], "MISSING")
        self.assertIn("missing_above_bound", json.dumps(rep))


class TestLogBeyondTheRun(Case):
    def test_whisper_extra_step(self):
        """A well-formed step 31 (8 ids of epoch 2 not yet seen) after max_steps 30: train_whisper.py cannot write
        it (the Trainer stops at max_steps), so the log is another run's or edited."""
        sc = W.WhisperScenario(self.path("w"), n_items=12)
        def fn(samples, steps):
            seen = {x for st in steps if st["epoch"] == 2 for x in st["example_ids"]}
            ids = [r["utt_id"] for r in sc.train_rows if r["utt_id"] not in seen][:8]
            steps.append(dict(steps[-1], step=31, example_ids=ids, n_examples=8))
            for i, u in enumerate(ids):
                samples.append(dict(samples[-1], step=31, position=i, utt_id=u, row=W.sha256_text(u.encode())))
        sc.edit_logs(fn)
        self.refused(sc.collect(self.path("runs")), "EXT-CONSUMPTION", "--stage-out", "max_steps")


if __name__ == "__main__":
    unittest.main()
