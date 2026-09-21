import contextlib
import io
import inspect
import sys
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import modal_setup


class SandboxLifecycleTest(unittest.TestCase):
    def test_downloads_only_missing_benchmark_shards_and_reuses_them(self):
        for existing in ((), ("fineweb_val_000000.bin", "fineweb_train_000005.bin")):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for name in existing:
                    (directory / name).touch()

                def download(**kwargs):
                    for name in kwargs["allow_patterns"]:
                        (Path(kwargs["local_dir"]) / name).touch()

                snapshot_download = Mock(side_effect=download)
                module = SimpleNamespace(snapshot_download=snapshot_download)
                namespace = {}
                exec(inspect.getsource(modal_setup.download_fineweb), namespace)
                with patch.dict(sys.modules, {"huggingface_hub": module}):
                    namespace["download_fineweb"](directory)
                    namespace["download_fineweb"](directory)
                snapshot_download.assert_called_once()
                request = snapshot_download.call_args.kwargs
                self.assertEqual(request["repo_id"], "kjj0/fineweb10B-gpt2")
                self.assertEqual(request["repo_type"], "dataset")
                self.assertEqual(len(request["allow_patterns"]), 10 - len(existing))
                self.assertTrue(set(existing).isdisjoint(request["allow_patterns"]))
                self.assertEqual({path.name for path in directory.iterdir()}, {
                    "fineweb_val_000000.bin", "fineweb_train_000001.bin", "fineweb_train_000002.bin",
                    "fineweb_train_000003.bin", "fineweb_train_000004.bin", "fineweb_train_000005.bin",
                    "fineweb_train_000006.bin", "fineweb_train_000007.bin", "fineweb_train_000008.bin",
                    "fineweb_train_000009.bin",
                })

    def test_allocation_failure_stops_created_app(self):
        app = SimpleNamespace(app_id="ap-test")
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(modal_setup.modal.App, "lookup", return_value=app), \
                patch.object(modal_setup, "training_image"), \
                patch.object(modal_setup.modal.Volume, "from_name"), \
                patch.object(modal_setup.modal.Sandbox, "create", side_effect=RuntimeError("allocation failed")) as create, \
                patch.object(modal_setup, "stop") as stop, \
                self.assertRaisesRegex(RuntimeError, "allocation failed"):
            directory = Path(temporary)
            try:
                modal_setup.create(directory)
            finally:
                stop.assert_called_once_with(directory)
                self.assertEqual(create.call_args.kwargs["cloud"], "gcp")
                self.assertEqual(json.loads((directory / "app.json").read_text())["app_id"], "ap-test")

    def test_failed_offline_preflight_stops_setup_before_accepting_evaluations(self):
        def process(output="", code=0):
            return SimpleNamespace(stdout=io.StringIO(output), stderr=io.StringIO(),
                                   returncode=code, wait=lambda: code)

        sandbox = Mock(object_id="sb-test")
        sandbox.exec.side_effect = [process(), process(), process("verified online\n"),
                                   process("cache incomplete\n", 1)]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(modal_setup.modal.App, "lookup", return_value=SimpleNamespace(app_id="ap-test")), \
                patch.object(modal_setup, "training_image", return_value=SimpleNamespace(object_id="im-test")), \
                patch.object(modal_setup.modal.Volume, "from_name"), \
                patch.object(modal_setup.modal.Sandbox, "create", return_value=sandbox), \
                patch.object(modal_setup.torch_cache, "restore"), \
                patch.object(modal_setup, "stop") as stop:
            directory = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "offline preflight failed"):
                modal_setup.create(directory)
            stop.assert_called_once_with(directory)
            sandbox.filesystem.write_text.assert_not_called()
            self.assertIn("cache incomplete", (directory / "dependency-offline.txt").read_text())
            sandbox.detach.assert_called_once()

    def test_cache_save_failure_still_terminates_sandbox_and_app(self):
        sandbox = Mock()
        sandbox.poll.return_value = None
        sandbox.exec.side_effect = RuntimeError("cache unavailable")
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(modal_setup.modal.Sandbox, "from_id", return_value=sandbox), \
                patch.object(modal_setup.subprocess, "run") as command, \
                contextlib.redirect_stderr(io.StringIO()):
            directory = Path(temporary)
            (directory / "app.json").write_text(json.dumps({"app_id": "ap-test", "environment": "sandbox"}))
            (directory / "sandbox.json").write_text(json.dumps({"id": "sb-test", "archive": "/persistent/test.tar"}))
            command.return_value = SimpleNamespace(stdout="", stderr="", returncode=0, check_returncode=lambda: None)
            modal_setup.stop(directory)
            sandbox.terminate.assert_called_once_with(wait=True)
            self.assertIn("ap-test", command.call_args.args[0])
            self.assertTrue((directory / "stopped.json").exists())

    def test_stopping_an_already_stopped_app_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(modal_setup.subprocess, "run") as command, \
                contextlib.redirect_stderr(io.StringIO()):
            directory = Path(temporary)
            (directory / "app.json").write_text(json.dumps({"app_id": "ap-test", "environment": "sandbox"}))
            command.return_value = SimpleNamespace(
                stdout="", stderr="App is already stopped. (Stopped earlier.)", returncode=1,
                check_returncode=Mock(side_effect=AssertionError("already stopped is successful cleanup")),
            )
            modal_setup.stop(directory)
            command.return_value.check_returncode.assert_not_called()
            self.assertTrue((directory / "stopped.json").exists())


if __name__ == "__main__":
    unittest.main()
