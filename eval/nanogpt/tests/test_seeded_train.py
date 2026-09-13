import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import modal_eval


class SeededTrainTest(unittest.TestCase):
    def test_seeds_and_main_module_without_extra_launcher_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copy2(modal_eval.HARNESS / "seeded_train.py", root)
            (root / "torch.py").write_text(
                "from types import SimpleNamespace\n"
                "seeds = []\n"
                "manual_seed = seeds.append\n"
                "cuda = SimpleNamespace(manual_seed_all=seeds.append)\n"
                "def get_num_threads(): return 1\n"
                "def get_num_interop_threads(): return 1\n"
            )
            target = root / "train_gpt.py"
            target.write_text(
                "import json, random, sys\n"
                "from dataclasses import dataclass\n"
                "from pathlib import Path\n"
                "import numpy as np\n"
                "import torch\n"
                "@dataclass\n"
                "class Record:\n"
                "    value: int = 1\n"
                "depth = 0\n"
                "frame = sys._getframe().f_back\n"
                "while frame is not None:\n"
                "    depth += 1\n"
                "    frame = frame.f_back\n"
                "print(json.dumps({'random': random.random(), 'numpy': np.random.random(),\n"
                "    'torch_seeds': torch.seeds, 'depth': depth, 'argv': sys.argv,\n"
                "    'file': str(Path(__file__).resolve()),\n"
                "    'main_file': str(Path(sys.modules['__main__'].__file__).resolve()),\n"
                "    'record': Record().value}))\n"
            )
            runs = []
            for seed in (42, 42, 43):
                process = subprocess.run(
                    [sys.executable, str(root / "seeded_train.py")], cwd=root,
                    env={**os.environ, "NANOGPT_SEED": str(seed), "LOCAL_RANK": "0"},
                    capture_output=True, text=True, check=True,
                )
                result = json.loads(process.stdout.splitlines()[-1])
                self.assertEqual(result["torch_seeds"], [seed, seed])
                self.assertEqual(result["argv"], ["train_gpt.py"])
                self.assertEqual(result["file"], str(target))
                self.assertEqual(result["main_file"], str(target))
                self.assertEqual(result["record"], 1)
                self.assertLessEqual(result["depth"], 1)
                runs.append(result)
            self.assertEqual(runs[0], runs[1])
            self.assertNotEqual(runs[0]["random"], runs[2]["random"])
            self.assertNotEqual(runs[0]["numpy"], runs[2]["numpy"])


if __name__ == "__main__":
    unittest.main()
