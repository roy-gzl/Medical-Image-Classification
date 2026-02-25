import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    # Set random seeds for Python/NumPy/PyTorch.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # If strict determinism is required, uncomment these lines.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_json(path: Path, obj: dict):
    # Write dict as UTF-8 JSON with indentation.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
