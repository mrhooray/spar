import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import modal_multi_seed


class MultiSeedTest(unittest.TestCase):
    def test_invokes_single_target_eval_for_every_seed_and_preserves_results(self):
        report, calls, snapshot = self.run_multi([73.0, 73.2, 73.1])
        self.assertTrue(report["passed"])
        self.assertEqual(report["median_training_seconds"], 73.1)
        self.assertEqual([call.args[1] for call in calls], [606, 707, 808])
        self.assertTrue(all(call.args[0] is snapshot for call in calls))
        self.assertTrue(all(call.args[3] == "sb-test123" for call in calls))
        self.assertEqual([row["training_seconds"] for row in report["runs"]], [73.0, 73.2, 73.1])

    def test_slow_run_fails_even_if_median_passes(self):
        report, _, _ = self.run_multi([73.0, 73.2, 78.0], failure=True)
        self.assertFalse(report["passed"])

    def test_failed_loss_fails_multiseed_score(self):
        report, _, _ = self.run_multi([73.0, 73.2, 73.1], losses=[3.269, 3.29, 3.271], failure=True)
        self.assertEqual(report["score"], -3.29)

    def test_duplicate_seeds_fail_before_evaluation(self):
        with patch.object(sys, "argv", ["modal_multi_seed.py", "--sandbox-id", "sb-test123", "--seeds", "606,606"]), \
                patch.object(modal_multi_seed.modal_eval, "evaluate") as evaluate, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            modal_multi_seed.main()
        evaluate.assert_not_called()

    def test_training_error_cannot_produce_a_successful_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(sys, "argv", ["modal_multi_seed.py", "--sandbox-id", "sb-test123", "--seeds", "606,707", "--output", str(directory)]), \
                    patch.object(modal_multi_seed.modal_eval, "source_snapshot", return_value={"sources": {}}), \
                    patch.object(modal_multi_seed.modal_eval, "evaluate", side_effect=ValueError("training failed")), \
                    self.assertRaisesRegex(ValueError, "training failed"):
                modal_multi_seed.main()
            self.assertFalse((directory / "result.json").exists())

    def run_multi(self, seconds, losses=None, failure=False):
        losses = losses or [3.269, 3.270, 3.271]
        measured = iter(zip(seconds, losses))
        snapshot = {"sources": {}, "commit": None, "source_sha256": {"train_gpt.py": "target"}}

        def evaluate(snapshot, seed, directory, sandbox_id):
            duration, loss = next(measured)
            result = {"seed": seed, "training_seconds": duration, "val_loss": loss, "image_id": "image",
                      "reached_target": loss <= 3.28, "beat_77_5": loss <= 3.28 and duration < 77.5}
            directory.mkdir()
            (directory / "result.json").write_text(json.dumps(result))
            return result

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(sys, "argv", ["modal_multi_seed.py", "--sandbox-id", "sb-test123", "--seeds", "606,707,808", "--output", str(directory)]), \
                    patch.object(modal_multi_seed.modal_eval, "source_snapshot", return_value=snapshot), \
                    patch.object(modal_multi_seed.modal_eval, "evaluate", side_effect=evaluate) as single, \
                    contextlib.redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaises(SystemExit) as error:
                        modal_multi_seed.main()
                    self.assertEqual(error.exception.code, 1)
                else:
                    modal_multi_seed.main()
            self.assertEqual(len(list(directory.glob("run-*/result.json"))), 3)
            return json.loads((directory / "result.json").read_text()), single.call_args_list, snapshot


if __name__ == "__main__":
    unittest.main()
