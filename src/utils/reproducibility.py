"""
Scientific Reproducibility and Seed Management.
Ensures deterministic runs across Python, NumPy, and PyTorch.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cuda: bool = False) -> None:
    """Sets random seeds across all libraries for scientific reproducibility.

    Args:
        seed: Integer random seed (default 42).
        deterministic_cuda: If True, forces cuDNN determinism (may reduce speed).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

