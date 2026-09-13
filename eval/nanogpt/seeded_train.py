import json
import os
import random
import sys

import numpy as np
import torch


assert torch.get_num_threads() == 1, "Training must use one CPU thread per rank"
if os.environ.get("LOCAL_RANK", "0") == "0":
    print("HARNESS_TRAINING_ENV=" + json.dumps({
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }), flush=True)
seed = int(os.environ["NANOGPT_SEED"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
sys.argv = ["train_gpt.py"]
__file__ = os.path.abspath("train_gpt.py")
with open(__file__) as source:
    code = compile(source.read(), __file__, "exec")
exec(code, globals())
