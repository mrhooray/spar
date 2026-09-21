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

    def test_mean_below_target_without_significance_fails(self):
        report, _, _ = self.run_multi([73.0, 73.2, 73.1], losses=[3.269, 3.29, 3.271], failure=True)
        self.assertLess(report["mean_val_loss"], 3.28)
        self.assertGreater(report["loss_p_value_one_sided"], 0.01)
        self.assertFalse(report["reached_target"])
        self.assertEqual(report["score"], -report["mean_val_loss"])

    def test_significant_mean_passes_despite_individual_loss_miss(self):
        losses = [3.275, 3.276, 3.277, 3.278, 3.275, 3.276, 3.277, 3.278, 3.276, 3.2801]
        report, _, _ = self.run_multi([73.0] * len(losses), losses=losses)
        self.assertTrue(report["passed"])
        self.assertTrue(report["reached_target"])
        self.assertLess(report["loss_p_value_one_sided"], 0.01)
        self.assertFalse(report["runs"][-1]["reached_target"])
        self.assertAlmostEqual(report["score"], 77.5 / 73.0)

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
        seeds = ",".join(str(606 + index * 101) for index in range(len(seconds)))
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
            with patch.object(sys, "argv", ["modal_multi_seed.py", "--sandbox-id", "sb-test123", "--seeds", seeds, "--output", str(directory)]), \
                    patch.object(modal_multi_seed.modal_eval, "source_snapshot", return_value=snapshot), \
                    patch.object(modal_multi_seed.modal_eval, "evaluate", side_effect=evaluate) as single, \
                    contextlib.redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaises(SystemExit) as error:
                        modal_multi_seed.main()
                    self.assertEqual(error.exception.code, 1)
                else:
                    modal_multi_seed.main()
            self.assertEqual(len(list(directory.glob("run-*/result.json"))), len(seconds))
            return json.loads((directory / "result.json").read_text()), single.call_args_list, snapshot


if __name__ == "__main__":
    unittest.main()
