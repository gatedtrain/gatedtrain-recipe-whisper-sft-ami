"""gatedtrain's generic rules exercised through `hf_seq2seq`, moved from gatedtrain's SD3.5 and CAD tests with the
recipe: a boolean step number refuses (review 10 L3), a mixed-type `deployable` list is an incomplete cut (L1)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import _whisper as W
from gatedtrain.testing.support import EXIT_REFUSED, describe, parse_errors


class TestCrossRules(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="whisper-cross-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_boolean_step_refused(self):
        sc = W.WhisperScenario(self.tmp / "w", n_items=12)
        sc.edit_logs(lambda samples, steps: steps[0].update(step=True))
        proc = sc.collect(self.tmp / "runs-w")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["rule"] == "EXT-CONSUMPTION"
                         and "consecutive" in e["message"]], proc.stderr)

    def test_mixed_type_deployable(self):
        from gatedtrain.ext import registry
        other = W.WhisperScenario(self.tmp / "l1")
        d = other.final_cut
        doc = json.loads((d / "cut.json").read_text())
        doc["deployable"] = [1, "x"]
        (d / "cut.json").write_text(json.dumps(doc))
        step = int(d.name.split("-")[1])
        self.assertIn("deployable", registry.adapter("hf_seq2seq").cut_problem(d, step))


if __name__ == "__main__":
    unittest.main()
