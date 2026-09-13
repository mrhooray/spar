import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from grpclib.exceptions import StreamTerminatedError

import modal_eval
from modal_eval import collect_training


CANDIDATE = Path(os.environ.get("NANOGPT_TEST_CANDIDATE", modal_eval.HARNESS.parents[1] / "nanogpt"))


class StickyEvaluationTest(unittest.TestCase):
    def test_two_evaluations_reuse_allocation_with_fresh_processes_and_sources(self):
        sandbox = FakeSandbox()
        with tempfile.TemporaryDirectory() as temporary, self.connections(sandbox):
            for index in range(2):
                result = modal_eval.evaluate(self.snapshot(index), 42, Path(temporary) / str(index), sandbox.object_id)
                self.assertEqual(result["sandbox_id"], sandbox.object_id)
                self.assertEqual(result["training_seconds"], 73.2)
                self.assertTrue(result["reached_target"])
                self.assertEqual(sandbox.uploads["/workspace/train_gpt.py"], f"# source {index}\n")
            self.assertEqual(sandbox.training_calls, 2)
            self.assertTrue(all(env["HF_HUB_OFFLINE"] == "1" for env in sandbox.training_environments))
            self.assertEqual(sandbox.clean_calls, 2)
            self.assertEqual(sandbox.detach.call_count, 2)
            self.assertEqual(sandbox.claims, 0)

    def test_failed_training_is_not_scored_and_releases_connection(self):
        sandbox = FakeSandbox(failed=True)
        with tempfile.TemporaryDirectory() as temporary, self.connections(sandbox):
            directory = Path(temporary)
            with self.assertRaisesRegex(ValueError, "final validation"):
                modal_eval.evaluate(self.snapshot(0), 42, directory, sandbox.object_id)
            self.assertFalse((directory / "result.json").exists())
            self.assertTrue((directory / "train.stdout").exists())
            self.assertEqual(sandbox.claims, 0)
            sandbox.detach.assert_called_once()

    def test_busy_sandbox_is_not_overwritten(self):
        sandbox = FakeSandbox(busy=True)
        with tempfile.TemporaryDirectory() as temporary, self.connections(sandbox):
            with self.assertRaisesRegex(RuntimeError, "active GPU processes"):
                modal_eval.evaluate(self.snapshot(0), 42, Path(temporary), sandbox.object_id)
            self.assertEqual(sandbox.training_calls, 0)
            self.assertEqual(sandbox.clean_calls, 0)
            self.assertEqual(sandbox.uploads, {})

    def test_cleanup_failure_preserves_training_error(self):
        sandbox = FakeSandbox(failed=True)
        execute = sandbox.exec

        def exec_with_failed_cleanup(*args, **kwargs):
            if args[0] == "rmdir":
                raise OSError("sandbox stopped")
            return execute(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary, self.connections(sandbox), \
                patch.object(sandbox, "exec", side_effect=exec_with_failed_cleanup):
            with self.assertRaisesRegex(ValueError, "final validation") as raised:
                modal_eval.evaluate(self.snapshot(0), 42, Path(temporary), sandbox.object_id)
            self.assertIn("sandbox stopped", " ".join(raised.exception.__notes__))
            sandbox.detach.assert_called_once()

    def test_existing_manifest_cannot_be_overwritten(self):
        sandbox = FakeSandbox()
        with tempfile.TemporaryDirectory() as temporary, self.connections(sandbox):
            directory = Path(temporary)
            (directory / "manifest.json").write_text("{}\n")
            with self.assertRaises(FileExistsError):
                modal_eval.evaluate(self.snapshot(0), 42, directory, sandbox.object_id)
            self.assertEqual(sandbox.training_calls, 0)

    def snapshot(self, index):
        return {"sources": {name: f"# source {index}\n" for name in modal_eval.SOURCES},
                "source_sha256": {}, "commit": None}

    @contextlib.contextmanager
    def connections(self, sandbox):
        def collect(sandbox, process, path):
            path.write_text(process.stdout.read())
            return process.returncode

        with patch.object(modal_eval.modal.Sandbox, "from_id", return_value=sandbox), \
                patch.object(modal_eval.modal.Sandbox, "create", side_effect=AssertionError("must not allocate")), \
                patch.object(modal_eval, "collect_training", side_effect=collect), \
                contextlib.redirect_stderr(io.StringIO()):
            yield


class FakeSandbox:
    object_id = "sb-test123"

    def __init__(self, failed=False, busy=False):
        self.failed = failed
        self.busy = busy
        self.uploads = {}
        self.training_calls = 0
        self.training_environments = []
        self.clean_calls = 0
        self.claims = 0
        self.detach = Mock()
        self.metadata = {"id": self.object_id, "gpu": modal_eval.GPU, "image_id": "im-test",
                         "name": "abcdef12", "app_id": "ap-test", "environment": "sandbox"}
        self.filesystem = SimpleNamespace(
            read_text=lambda path: json.dumps(self.metadata),
            write_text=lambda text, path: self.uploads.update({path: text}),
        )

    def exec(self, *args, **kwargs):
        text, code = "", 0
        if args[0] == "mkdir":
            self.claims += 1
        elif args[0] == "rmdir":
            self.claims -= 1
        elif args[0] == "nvidia-smi":
            if "--query-compute-apps=pid" in args:
                text = "1234" if self.busy else ""
            else:
                text = "\n".join(f"GPU-{index}, NVIDIA H100" for index in range(8))
        elif args[0] == "bash" and "find ." in args[2]:
            self.clean_calls += 1
        elif args[0] == "bash" and "torchrun" in args[2]:
            self.training_calls += 1
            self.training_environments.append(kwargs["env"])
            code = 1 if self.failed else 0
            text = 'HARNESS_TRAINING_ENV={"torch_num_threads":1}\n'
            if not self.failed:
                text += "step:1285/1285 val_loss:3.277 train_time:73200ms\n"
        return SimpleNamespace(stdout=io.StringIO(text), stderr=io.StringIO(),
                               returncode=code, wait=lambda: code)


class MetricsTest(unittest.TestCase):
    def test_strict_target_and_time_boundary(self):
        def run(loss, milliseconds):
            return modal_eval.summarize(f"step:1285/1285 val_loss:{loss} train_time:{milliseconds}ms\n", 0)
        self.assertTrue(run("3.27999", 77499)["beat_77_5"])
        self.assertFalse(run("3.28001", 70000)["reached_target"])
        self.assertFalse(run("3.28", 77500)["beat_77_5"])
        self.assertGreater(run("3.28", 80000)["score"], run("3.281", 70000)["score"])

    def test_requires_completed_successful_training(self):
        for log, code in [("", 0), ("step:250/1285 val_loss:3.2 train_time:10000ms", 0),
                          ("step:1285/1285 val_loss:3.2 train_time:70000ms", 1),
                          ("step:1285/1285 val_loss:3.2 train_time:0ms", 0)]:
            with self.assertRaises(ValueError):
                modal_eval.summarize(log, code)


class PortabilityTest(unittest.TestCase):
    def test_relocated_harness_and_plain_source_folder_need_no_git(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            moved = root / "portable harness"
            shutil.copytree(modal_eval.HARNESS, moved,
                            ignore=shutil.ignore_patterns(".venv", "__pycache__", "results"))
            candidate = root / "plain source"
            candidate.mkdir()
            for name in modal_eval.SOURCES:
                shutil.copy2(CANDIDATE / name, candidate / name)
            result = subprocess.run(
                [sys.executable, str(moved / "modal_eval.py"), "check", "--candidate", str(candidate)],
                cwd=root, env={**os.environ, "PATH": ""}, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertIsNone(report["commit"])
            self.assertEqual(set(report["source_sha256"]), set(modal_eval.SOURCES))

    def test_protected_validation_and_clock_changes_are_rejected_without_git(self):
        original = (CANDIDATE / "train_gpt.py").read_text()
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary)
            for before, after in (
                ("val_tokens: int = 10485760", "val_tokens: int = 1048576"),
                ("training_time_ms = 0", "training_time_ms = -1000"),
            ):
                with self.subTest(before=before):
                    self.assertIn(before, original)
                    (candidate / "train_gpt.py").write_text(original.replace(before, after, 1))
                    with self.assertRaises(ValueError):
                        modal_eval.validate_candidate(candidate)


class LogCollectionTest(unittest.TestCase):
    def test_recovers_connection_and_reads_final_log_after_exit(self):
        process = Mock()
        process.poll.side_effect = [StreamTerminatedError("connection lost"), None, 0]
        sandbox = Mock()
        final = "step:1285/1285 val_loss:3.279 train_time:77000ms\n"
        sandbox.filesystem.read_text.side_effect = ["Compiling\n", "Compiling\n" + final]
        with tempfile.TemporaryDirectory() as directory, patch("modal_eval.time.sleep"):
            logfile = Path(directory) / "train.stdout"
            self.assertEqual(collect_training(sandbox, process, logfile), 0)
            self.assertTrue(logfile.read_text().endswith(final))

    def test_persistent_connection_failure_is_not_scored(self):
        process = Mock()
        process.poll.side_effect = StreamTerminatedError("connection lost")
        with patch("modal_eval.time.sleep"), self.assertRaises(StreamTerminatedError):
            collect_training(Mock(), process, Path("unused"))


class ProfileTest(unittest.TestCase):
    def test_reads_saved_timing_without_allocating_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timing = modal_eval.timing_profile(training_log(73000))
            (root / "result.json").write_text(json.dumps({"timing": timing}))
            output = io.StringIO()
            with patch.object(sys, "argv", ["modal_eval.py", "profile", "--output", str(root / "profiling")]), \
                    patch.object(modal_eval, "evaluate") as evaluate, contextlib.redirect_stdout(output):
                modal_eval.main()
            self.assertEqual(json.loads(output.getvalue())["median_step_ms"], 15)
            evaluate.assert_not_called()


def training_log(milliseconds, loss=3.27):
    return ('HARNESS_TRAINING_ENV={"torch_num_threads": 1}\n'
            "step:1/10 train_time:10ms\nstep:2/10 train_time:25ms\nstep:3/10 train_time:40ms\n"
            f"step:10/10 val_loss:{loss} train_time:{milliseconds}ms\n")


if __name__ == "__main__":
    unittest.main()
