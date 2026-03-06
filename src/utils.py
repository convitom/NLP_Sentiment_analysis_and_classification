"""
src/utils.py
Shared utility helpers for GoEmotions Emotion Classification.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from transformers import get_cosine_schedule_with_warmup
import yaml


###############################################################################
#  Config
###############################################################################

def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config and return as a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


###############################################################################
#  Reproducibility
###############################################################################

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


###############################################################################
#  Optimizers & Schedulers
###############################################################################

def get_optimizer(model: nn.Module, cfg: dict) -> optim.Optimizer:
    """
    Build optimizer.  Uses separate param groups:
      - Transformer backbone  : lr from config
      - Classifier head       : lr * 10  (faster convergence for new weights)
    """
    train_cfg = cfg["training"]
    name = train_cfg.get("optimizer", "adamw").lower()
    lr   = float(train_cfg.get("lr",           2e-5))
    wd   = float(train_cfg.get("weight_decay", 0.01))

    # Separate backbone vs head parameters
    backbone_params   = []
    classifier_params = []
    for pname, param in model.named_parameters():
        if "classifier" in pname:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params,   "lr": lr,       "weight_decay": wd},
        {"params": classifier_params, "lr": lr * 10,  "weight_decay": wd},
    ]

    if name == "adamw":
        return optim.AdamW(param_groups)
    elif name == "adam":
        return optim.Adam(param_groups)
    elif name == "sgd":
        return optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: '{name}'. Choose from adamw | adam | sgd.")


def get_scheduler(
    optimizer: optim.Optimizer,
    cfg: dict,
    num_training_steps: int,
):
    """
    Build LR scheduler.

    Args:
        optimizer:           The optimizer.
        cfg:                 Config dict.
        num_training_steps:  Total number of training steps (epochs * steps_per_epoch).

    Returns:
        Scheduler or None.
    """
    train_cfg    = cfg["training"]
    name         = train_cfg.get("scheduler", "cosine_warmup").lower()
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    num_warmup   = int(num_training_steps * warmup_ratio)

    if name == "cosine_warmup":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup,
            num_training_steps=num_training_steps,
        )
    elif name == "cosine":
        epochs = int(train_cfg.get("epochs", 10))
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        step_size = int(train_cfg.get("step_size", 3))
        gamma     = float(train_cfg.get("gamma",     0.1))
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: '{name}'.")


###############################################################################
#  Running average meter
###############################################################################

class AverageMeter:
    """Tracks the running average of a scalar."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0

    def __repr__(self):
        return f"AverageMeter({self.name}): avg={self.avg:.4f}"


###############################################################################
#  Result directory
###############################################################################

def get_result_dir(cfg: dict) -> str:
    """Return (and create if necessary) results/<model_name>/."""
    base       = cfg["results"].get("base_dir", "results/")
    model_name = cfg["model"]["name"]
    result_dir = os.path.join(base, model_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


###############################################################################
#  Multi-label metric helpers
###############################################################################

def apply_threshold(
    probs: np.ndarray,
    threshold: float = 0.5,
    neutral_label_idx: Optional[int] = None,
    num_emotions: int = 27,
) -> np.ndarray:
    """
    Convert probability matrix to binary predictions with Neutral fallback.

    Args:
        probs:             (N, 27) sigmoid probabilities.
        threshold:         Decision threshold.
        neutral_label_idx: If provided, samples where no emotion exceeds threshold
                           are assigned Neutral (as a separate indicator).
        num_emotions:      Number of emotion columns (27).

    Returns:
        preds: (N, 27) binary array — 1 if prob > threshold, else 0.
               Neutral is NOT added as a column; callers check row-sum == 0.
    """
    preds = (probs >= threshold).astype(np.int32)
    return preds


def is_neutral(preds: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of shape (N,): True where sample has no predicted emotion.
    These samples are considered Neutral.
    """
    return preds.sum(axis=1) == 0
