"""
src/dataloader.py
GoEmotions dataset — PyTorch DataLoader for multi-label emotion classification.

Expected layout under data_dir:
    data/
    ├── train/          # HuggingFace Arrow dataset split (folder)
    ├── validation/     # or val/
    ├── test/
    └── dataset_info.json

Labels:
    28 total classes (indices 0-27). Index 27 = Neutral.
    We train on 27 emotion classes only (indices 0-26).
    Neutral is inferred at inference time: if no emotion exceeds threshold.

Label order (matches GoEmotions on HuggingFace):
    0:admiration  1:amusement   2:anger        3:annoyance    4:approval
    5:caring      6:confusion   7:curiosity    8:desire       9:disappointment
    10:disapproval 11:disgust   12:embarrassment 13:excitement 14:fear
    15:gratitude  16:grief      17:joy         18:love        19:nervousness
    20:optimism   21:pride      22:realization 23:relief      24:remorse
    25:sadness    26:surprise   [27:neutral  <- excluded from training labels]
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset, WeightedRandomSampler
from transformers import AutoTokenizer


#==============================================================================
#  Backbone registry  (single source of truth for pretrained IDs)
#==============================================================================

BACKBONE_REGISTRY: Dict[str, Dict[str, str]] = {
    #  Key        HuggingFace pretrained ID                    Notes
    # ---------   -----------------------------------------    -----------------------
    "bert":    {"pretrained": "google-bert/bert-base-uncased"},    # Original BERT, 110M params
    "roberta": {"pretrained": "FacebookAI/roberta-base"},           # Robust BERT, 125M params
    "deberta": {"pretrained": "microsoft/deberta-v3-base"},         # Best on GLUE/SQuAD, 184M params
    "electra": {"pretrained": "google/electra-base-discriminator"}, # Efficient & fast, 110M params
}


#==============================================================================
#  Label metadata
#==============================================================================

EMOTION_NAMES: List[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise",
]   # 27 emotion classes; index == position in multi-hot vector

NEUTRAL_LABEL_IDX = 27   # original GoEmotions index for "neutral"
NUM_EMOTIONS      = 27   # number of emotion classes used for training


#==============================================================================
#  PyTorch Dataset
#==============================================================================

class GoEmotionsDataset(TorchDataset):
    """
    Wraps a HuggingFace Dataset split and returns tokenised inputs
    + multi-hot label vectors of length 27 (emotions only).

    Args:
        hf_dataset:    A HuggingFace ``Dataset`` object (one split).
        tokenizer:     Pretrained tokenizer.
        max_length:    Maximum token length.
        neutral_idx:   Original label index of Neutral (default 27).
    """

    def __init__(
        self,
        hf_dataset: Dataset,
        tokenizer: ElectraTokenizerFast,
        max_length: int = 128,
        neutral_idx: int = NEUTRAL_LABEL_IDX,
    ) -> None:
        self.dataset    = hf_dataset
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.neutral_idx = neutral_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        text   = sample["text"]

        # Tokenise
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Multi-hot encoding — 27 emotions only (drop neutral)
        raw_labels  = sample["labels"]          # list of ints from 0-27
        multi_hot   = torch.zeros(NUM_EMOTIONS, dtype=torch.float32)
        for lbl in raw_labels:
            if lbl != self.neutral_idx:         # skip neutral
                multi_hot[lbl] = 1.0

        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (max_length,)
            "labels":         multi_hot,                          # (27,)
        }


#==============================================================================
#  Weighted sampler for multi-label imbalance
#==============================================================================

def build_weighted_sampler(dataset: GoEmotionsDataset) -> WeightedRandomSampler:
    """
    Build a ``WeightedRandomSampler`` so that samples containing rare-emotion
    labels are drawn more frequently.

    Weight of sample i  =  mean of (1 / label_frequency) over its positive labels.
    Pure-neutral samples (all-zero multi-hot) get the minimum weight.

    Args:
        dataset: A ``GoEmotionsDataset`` instance.

    Returns:
        ``WeightedRandomSampler`` that produces len(dataset) samples per epoch.
    """
    n = len(dataset)

    # Accumulate label frequencies
    label_counts = np.zeros(NUM_EMOTIONS, dtype=np.float64)
    all_labels   = []
    for i in range(n):
        lbl = dataset[i]["labels"].numpy()     # (27,)
        all_labels.append(lbl)
        label_counts += lbl

    # Inverse frequency per label (add 1 to avoid div-by-zero)
    inv_freq = 1.0 / (label_counts + 1.0)

    # Sample weight = mean inv_freq over positive labels
    sample_weights = np.zeros(n, dtype=np.float64)
    for i, lbl in enumerate(all_labels):
        pos_mask = lbl > 0
        if pos_mask.any():
            sample_weights[i] = inv_freq[pos_mask].mean()
        else:
            sample_weights[i] = inv_freq.min()   # neutral-only samples get low weight

    sample_weights = torch.from_numpy(sample_weights).float()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n,
        replacement=True,
    )


#==============================================================================
#  Compute pos_weight for BCE loss
#==============================================================================

def compute_pos_weight(dataset: GoEmotionsDataset, device: torch.device) -> torch.Tensor:
    """
    Compute per-class positive weight for ``BCEWithLogitsLoss``.

        pos_weight[c] = (N - n_pos[c]) / n_pos[c]

    where N = total samples, n_pos[c] = samples with label c == 1.

    Args:
        dataset: Training ``GoEmotionsDataset``.
        device:  Target device.

    Returns:
        Tensor of shape (27,) on ``device``.
    """
    n            = len(dataset)
    label_counts = np.zeros(NUM_EMOTIONS, dtype=np.float64)

    for i in range(n):
        label_counts += dataset[i]["labels"].numpy()

    # Clamp to avoid zero division if a class never appears
    label_counts = np.clip(label_counts, 1, None)
    pos_weight   = (n - label_counts) / label_counts
    return torch.tensor(pos_weight, dtype=torch.float32, device=device)


#==============================================================================
#  Arrow file loader  — supports both individual .arrow files and disk folders
#==============================================================================

def _load_splits(data_dir: str) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load train / validation / test splits from data_dir.

    Supports two layouts automatically:

    Layout A — individual .arrow files (e.g. downloaded from HuggingFace):
        data/
        ├── go_emotions-train.arrow
        ├── go_emotions-validation.arrow
        └── go_emotions-test.arrow

    Layout B — load_from_disk folders (e.g. saved with dataset.save_to_disk()):
        data/
        ├── train/
        │   ├── dataset.arrow
        │   └── dataset_info.json
        ├── validation/
        └── test/

    Detection logic:
        1. Search for any *.arrow file in data_dir whose name contains
           "train", "validation"/"val", or "test".
        2. If found for all 3 splits → use Dataset.from_file() (Layout A).
        3. Otherwise → fall back to load_from_disk() on sub-folders (Layout B).

    Args:
        data_dir: Root directory containing the dataset files.

    Returns:
        (hf_train, hf_val, hf_test) — three HuggingFace Dataset objects.

    Raises:
        FileNotFoundError: If neither layout is detected for any split.
    """
    import glob

    arrow_files = glob.glob(os.path.join(data_dir, "*.arrow"))

    def _find_arrow(keyword_variants: List[str]) -> Optional[str]:
        """Return first .arrow file whose name contains any of the keywords."""
        for path in arrow_files:
            fname = os.path.basename(path).lower()
            if any(kw in fname for kw in keyword_variants):
                return path
        return None

    train_arrow = _find_arrow(["train"])
    val_arrow   = _find_arrow(["validation", "val"])
    test_arrow  = _find_arrow(["test"])

    # ── Layout A: individual .arrow files ────────────────────────────────────
    if train_arrow and val_arrow and test_arrow:
        print(f"[DataLoader] Loading individual .arrow files from '{data_dir}'")
        print(f"  train : {os.path.basename(train_arrow)}")
        print(f"  val   : {os.path.basename(val_arrow)}")
        print(f"  test  : {os.path.basename(test_arrow)}")
        return (
            Dataset.from_file(train_arrow),
            Dataset.from_file(val_arrow),
            Dataset.from_file(test_arrow),
        )

    # ── Layout B: load_from_disk sub-folders ─────────────────────────────────
    from datasets import load_from_disk

    val_dir = os.path.join(data_dir, "validation")
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(data_dir, "val")

    missing = []
    for split, path in [("train", os.path.join(data_dir, "train")),
                         ("val",   val_dir),
                         ("test",  os.path.join(data_dir, "test"))]:
        if not os.path.isdir(path):
            missing.append(split)

    if missing:
        raise FileNotFoundError(
            f"Could not find Arrow data for splits: {missing}.\n"
            f"  Checked data_dir='{data_dir}' for:\n"
            f"    • Individual .arrow files (e.g. go_emotions-train.arrow)\n"
            f"    • Sub-folders named train/ / validation/ / test/"
        )

    print(f"[DataLoader] Loading from disk folders in '{data_dir}'")
    return (
        load_from_disk(os.path.join(data_dir, "train")),
        load_from_disk(val_dir),
        load_from_disk(os.path.join(data_dir, "test")),
    )


#==============================================================================
#  Main factory
#==============================================================================

def get_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Build train / val / test DataLoaders from the pre-split GoEmotions Arrow files.

    Args:
        cfg: Parsed config dict.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)

        info_dict contains:
            'emotion_names'  : list of 27 emotion label names
            'pos_weight'     : Tensor(27) for weighted BCE (computed on train set)
            'label_counts'   : dict {emotion_name: count} from train set
    """
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]

    data_dir    = data_cfg["data_dir"]
    max_length  = int(data_cfg.get("max_length",      128))
    neutral_idx = int(data_cfg.get("neutral_label_idx", NEUTRAL_LABEL_IDX))
    batch_size  = int(train_cfg.get("batch_size",      32))

    # ── Resolve backbone name → pretrained ID ───────────────────────────────
    model_name = cfg["model"]["name"].lower()
    if model_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Choose from: {' | '.join(BACKBONE_REGISTRY.keys())}"
        )
    pretrained = BACKBONE_REGISTRY[model_name]["pretrained"]

    # ── Tokeniser (AutoTokenizer handles all 4 backbones transparently) ──────
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    # ── Load Arrow splits ────────────────────────────────────────────────────
    hf_train, hf_val, hf_test = _load_splits(data_dir)

    # ── Wrap into GoEmotionsDataset ─────────────────────────────────────────
    train_ds = GoEmotionsDataset(hf_train, tokenizer, max_length, neutral_idx)
    val_ds   = GoEmotionsDataset(hf_val,   tokenizer, max_length, neutral_idx)
    test_ds  = GoEmotionsDataset(hf_test,  tokenizer, max_length, neutral_idx)

    # ── Sampler (train only) ─────────────────────────────────────────────────
    train_sampler = build_weighted_sampler(train_ds)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Label frequency stats from train set ─────────────────────────────────
    label_counts_arr = np.zeros(NUM_EMOTIONS, dtype=np.int64)
    for i in range(len(train_ds)):
        label_counts_arr += train_ds[i]["labels"].numpy().astype(np.int64)
    label_counts_dict = {EMOTION_NAMES[i]: int(label_counts_arr[i]) for i in range(NUM_EMOTIONS)}

    # ── pos_weight for BCE losses ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use config override if provided, else auto-compute
    raw_pw = train_cfg.get("pos_weight", None)
    if raw_pw is not None:
        pos_weight = torch.tensor(raw_pw, dtype=torch.float32, device=device)
    else:
        pos_weight = compute_pos_weight(train_ds, device)

    print(f"[DataLoader] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[DataLoader] Max token length : {max_length}")
    print(f"[DataLoader] Label counts (train) : {label_counts_dict}")

    info = {
        "emotion_names": EMOTION_NAMES,
        "pos_weight":    pos_weight,
        "label_counts":  label_counts_dict,
    }
    return train_loader, val_loader, test_loader, info
