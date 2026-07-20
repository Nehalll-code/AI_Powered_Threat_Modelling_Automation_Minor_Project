"""
swin_branch.py — Swin Transformer Feature-Extraction Branch ("Brain 2")
========================================================================

Part of a hybrid image-forgery-detection system (Brain 1 = EfficientNet CNN,
Brain 2 = Swin Transformer, Brain 3 = FFT/DCT frequency branch — fused later
via cross-attention). This module implements **Brain 2 only**.

* **SwinBranch** — a Swin-Tiny backbone (pretrained, ImageNet-21k) whose
  classifier head is replaced by a learned 768→512 projection, producing
  a [B, 512] feature vector per image for downstream cross-attention fusion.
* **build_raw_records / apply_caps / split_dataframe / build_dataframe_loaders**
  — an exact mirror of the CNN branch's data-prep cell: scans CASIA v2
  (Au/Tp) and DeFACTO (copy-move + every ``splicing_*_img/img`` split)
  into one dataframe, applies the same per-``(source, label)`` caps, does
  the same two-step stratified 75/15/10 split (``SEED=42``, stratified on
  ``source + label``), and returns dataframe-backed ``DataLoader``s using
  a class-balanced ``WeightedRandomSampler`` built from the *training*
  labels only. Given the same paths/caps/seed, the CNN and Swin branches
  (and later the FFT/DCT branch) train/validate/test on identical images.
* **Training utilities** — layer-wise learning-rate decay (LLRD), OneCycleLR
  scheduler, mixed-precision training loop with AUC-based checkpointing.
* **Evaluation & single-image inference** helpers.

Dataset sources used in this version
-------------------------------------
* CASIA v2  : https://www.kaggle.com/datasets/dk9892/casia-v2
* DeFACTO   : https://www.kaggle.com/defactodataset/datasets
              (defactocopymove + defactosplicing, matching the CNN branch)

The exact Kaggle mount paths are module-level constants (``CASIA_AU``,
``CASIA_TP``, ``DEFACTO_CM``, ``DEFACTO_SP``) — keep these identical to
whatever the CNN branch uses.

Usage
-----
>>> from swin_branch import SwinBranch, build_dataframe_loaders, train_swin_branch
>>> train_loader, val_loader, test_loader, (df_train, df_val, df_test) = build_dataframe_loaders()

Compatible with Python ≥ 3.9, PyTorch ≥ 2.0, timm ≥ 0.9.
"""

from __future__ import annotations

# ── stdlib ───────────────────────────────────────────────────────────────
import os
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple

# ── third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── torchvision ──────────────────────────────────────────────────────────
from torchvision import transforms
from PIL import Image, ImageFile

# Some CASIA / DEFACTO images are slightly truncated; don't hard-crash on them.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── timm ─────────────────────────────────────────────────────────────────
import timm

# ── sklearn ──────────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ── global reproducibility seed ──────────────────────────────────────────
# Used for the per-(source, label) cap sampling AND the stratified
# 75/15/10 split, so the Swin branch trains/validates/tests on EXACTLY the
# same images as the CNN branch (and later the FFT/DCT branch), given the
# same dataset paths and caps.
SEED: int = 42

# ═════════════════════════════════════════════════════════════════════════
#  1.  MODEL
# ═════════════════════════════════════════════════════════════════════════


class SwinBranch(nn.Module):
    """Swin-Tiny feature extractor for image-forgery detection.

    Backbone
    --------
    ``swin_tiny_patch4_window7_224`` from *timm*, pretrained on ImageNet-21k.
    The classification head is removed (``num_classes=0``), leaving a
    768-dimensional global feature.

    Projection Head
    ---------------
    LayerNorm → Linear(768, 512) → GELU → Dropout(0.1)

    Parameters
    ----------
    pretrained : bool, default ``True``
        Whether to load ImageNet-21k pretrained weights.
    drop_rate : float, default ``0.1``
        Dropout probability applied after GELU.

    Returns
    -------
    torch.Tensor
        Shape ``[B, 512]`` — one feature vector per image.
    """

    def __init__(self, pretrained: bool = True, drop_rate: float = 0.1) -> None:
        super().__init__()

        # ── backbone (head removed) ──────────────────────────────────────
        self.backbone: nn.Module = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,  # removes the classification head
        )

        backbone_dim: int = self.backbone.num_features  # 768 for swin_tiny

        # ── projection head: 768 → 512 ──────────────────────────────────
        self.norm = nn.LayerNorm(backbone_dim)
        self.projection = nn.Linear(backbone_dim, 512)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a 512-d feature vector from an input image batch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[B, 3, 224, 224]``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``[B, 512]``.
        """
        features: torch.Tensor = self.backbone(x)      # [B, 768]
        features = self.norm(features)                  # [B, 768]
        features = self.projection(features)            # [B, 512]
        features = self.act(features)                   # [B, 512]
        features = self.dropout(features)                # [B, 512]
        return features


# ═════════════════════════════════════════════════════════════════════════
#  2.  TRANSFORMS
# ═════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
_IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


def get_swin_transforms(mode: str) -> transforms.Compose:
    """Return a ``torchvision.transforms.Compose`` pipeline for Swin-Tiny.

    Parameters
    ----------
    mode : str
        One of ``'train'``, ``'val'``, or ``'test'``.

    Returns
    -------
    torchvision.transforms.Compose

    Raises
    ------
    ValueError
        If *mode* is not one of the recognised strings.
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            transforms.RandomRotation(degrees=10),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    elif mode in ("val", "test"):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected 'train', 'val', or 'test'."
        )


# ═════════════════════════════════════════════════════════════════════════
#  3.  DATASET LOCATIONS & CONSTANTS  (must match the CNN branch exactly)
# ═════════════════════════════════════════════════════════════════════════
#
# These paths, caps, and the split logic in section 4 are a direct mirror
# of the CNN branch's data-prep cell, so both branches (and later the
# FFT/DCT branch) build the identical dataframe and the identical
# train/val/test split given the same SEED. Edit the paths below if your
# Kaggle dataset mount points differ — just keep them identical to the
# CNN branch's copies of the same constants.

CASIA_AU: pathlib.Path = pathlib.Path(
    "/kaggle/input/datasets/chongtrung/casia-v2/CASIA2/Au"
)
CASIA_TP: pathlib.Path = pathlib.Path(
    "/kaggle/input/datasets/chongtrung/casia-v2/CASIA2/Tp"
)
DEFACTO_CM: pathlib.Path = pathlib.Path(
    "/kaggle/input/datasets/defactodataset/defactocopymove/copymove_img/img"
)
DEFACTO_SP: pathlib.Path = pathlib.Path(
    "/kaggle/input/datasets/defactodataset/defactosplicing"
)

IMG_SIZE: int = 224
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 25
LR: float = 1e-4

_IMG_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _safe_load_image(path: str) -> Image.Image:
    """Open an image and convert to RGB, handling truncated files gracefully."""
    img = Image.open(path)
    img = img.convert("RGB")
    return img


def _collect(folder: "str | pathlib.Path", label: int, source: str) -> List[Dict[str, Any]]:
    """Recursively scan *folder* for images, tagging each row with a fixed
    ``label`` and ``source`` name.

    This is an exact mirror of the CNN branch's ``collect()`` helper, so
    both branches enumerate identical raw records (same paths, same
    labels, same source tags) before any capping or splitting happens.
    """
    rows: List[Dict[str, Any]] = []
    folder = pathlib.Path(folder)
    if not folder.exists():
        print(f"[MISSING] {folder}")
        return rows
    for p in folder.rglob("*"):
        if p.suffix.lower() in _IMG_EXTS:
            rows.append({
                "path": str(p),
                "label": label,
                "label_name": "authentic" if label == 0 else "tampered",
                "source": source,
            })
    return rows


def build_raw_records(
    casia_au: "str | pathlib.Path" = CASIA_AU,
    casia_tp: "str | pathlib.Path" = CASIA_TP,
    defacto_cm: "str | pathlib.Path" = DEFACTO_CM,
    defacto_sp: "str | pathlib.Path" = DEFACTO_SP,
) -> pd.DataFrame:
    """Scan CASIA v2 (Au/Tp) and DeFACTO (copy-move + every
    ``splicing_*_img/img`` split) into a single raw dataframe, before any
    caps or splitting are applied.

    Returns
    -------
    pandas.DataFrame
        Columns: ``path``, ``label`` (0=authentic / 1=tampered),
        ``label_name``, ``source``.
    """
    records: List[Dict[str, Any]] = []
    records += _collect(casia_au, 0, "CASIA_v2")
    records += _collect(casia_tp, 1, "CASIA_v2")
    records += _collect(defacto_cm, 1, "DeFACTO_copymove")
    for sp in sorted(pathlib.Path(defacto_sp).glob("splicing_*_img/img")):
        records += _collect(sp, 1, "DeFACTO_splicing")

    df_raw = pd.DataFrame(records)
    return df_raw


# Per-(source, label) sample caps — identical to the CNN branch's CAPS
# dict. ``None`` means "keep all". Change these here AND in the CNN branch
# together if you ever need to retune the class/source balance.
_DATASET_CAPS: Dict[Tuple[str, int], Optional[int]] = {
    ("CASIA_v2", 0):         7000,
    ("CASIA_v2", 1):         None,
    ("DeFACTO_splicing", 1): 3000,
    ("DeFACTO_copymove", 1): 1800,
}


def apply_caps(
    df_raw: pd.DataFrame,
    caps: Optional[Dict[Tuple[str, int], Optional[int]]] = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Sub-sample each ``(source, label)`` group down to its configured cap
    (or keep it whole if the cap is ``None``), then shuffle the combined
    result. Mirrors the CNN branch's capping cell exactly, including the
    ``random_state=SEED`` used for both the per-group sampling and the
    final shuffle.
    """
    caps = caps if caps is not None else _DATASET_CAPS

    parts: List[pd.DataFrame] = []
    for (src, lbl), cap in caps.items():
        grp = df_raw[(df_raw.source == src) & (df_raw.label == lbl)]
        n = len(grp) if cap is None else min(cap, len(grp))
        parts.append(grp.sample(n=n, replace=False, random_state=seed))

    df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ═════════════════════════════════════════════════════════════════════════
#  4.  STRATIFIED 75/15/10 SPLIT & DATAFRAME-BACKED DATASET / LOADERS
# ═════════════════════════════════════════════════════════════════════════
#
# NOTE: dataset-level weighting (e.g. "CASIA = 50% of every batch") has
# been removed. Composition is controlled entirely by `_DATASET_CAPS`
# above and the stratified split below — the same recipe the CNN branch
# uses — so every branch trains/validates/tests on identical images. Class
# imbalance within the resulting *training* split is handled by a plain
# class-balanced `WeightedRandomSampler`, exactly like the CNN branch.


def split_dataframe(
    df: pd.DataFrame,
    seed: int = SEED,
    test_size: float = 0.25,
    val_of_temp: float = 0.40,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 75/15/10 train/val/test split — identical two-step
    recipe to the CNN branch: split off 25% as a temp holdout (stratified
    on ``source + label``), then split that holdout 60/40 into val/test.
    That yields overall fractions of 75% / 15% / 10%.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``source`` and ``label`` columns.
    seed : int
        Random seed. Defaults to the module-level ``SEED`` (42).
    test_size : float
        Fraction held out as "temp" (val + test combined). Default 0.25.
    val_of_temp : float
        Fraction of the "temp" holdout that becomes the test set (the
        CNN branch calls this ``test_size`` in its second split). Default
        0.40, i.e. temp splits into 60% val / 40% test → 15% / 10% overall.

    Returns
    -------
    (df_train, df_val, df_test) : tuple of pandas.DataFrame
    """
    df = df.copy()
    df["strat"] = df["source"] + "_" + df["label"].astype(str)

    df_train, df_temp = train_test_split(
        df, test_size=test_size, stratify=df["strat"], random_state=seed,
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=val_of_temp, stratify=df_temp["strat"], random_state=seed,
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


class ForgeryDataFrameDataset(Dataset):
    """Dataframe-backed dataset — the single source of truth for image
    loading, shared across the CNN / Swin / FFT branches (each branch
    just passes its own transform pipeline).

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Must contain ``path`` and ``label`` columns.
    transform : optional
        ``torchvision.transforms`` pipeline — pass the output of
        :func:`get_swin_transforms` for this branch.
    """

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[transforms.Compose] = None) -> None:
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return ``(image_tensor, label)``. Falls back to a black image
        on a decode error instead of crashing the whole run, matching the
        CNN branch's defensive ``except Exception`` fallback."""
        row = self.df.iloc[idx]
        try:
            image = _safe_load_image(row["path"])
        except Exception:
            image = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def build_dataframe_loaders(
    casia_au: "str | pathlib.Path" = CASIA_AU,
    casia_tp: "str | pathlib.Path" = CASIA_TP,
    defacto_cm: "str | pathlib.Path" = DEFACTO_CM,
    defacto_sp: "str | pathlib.Path" = DEFACTO_SP,
    caps: Optional[Dict[Tuple[str, int], Optional[int]]] = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    seed: int = SEED,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Build train/val/test ``DataLoader``s using the exact same
    scan → cap → stratified-split → sample recipe as the CNN branch, so
    both branches train/validate/test on identical images.

    Returns
    -------
    (train_loader, val_loader, test_loader, (df_train, df_val, df_test))

    Notes
    -----
    * Train loader uses a class-balanced ``WeightedRandomSampler`` computed
      **only** from the training split's labels (inverse class frequency,
      same as the CNN branch). Val/test loaders use ``shuffle=False``.
    * No dataset-level batch-share weighting — composition is fully
      controlled by ``_DATASET_CAPS`` and the split above.
    """
    df_raw = build_raw_records(casia_au, casia_tp, defacto_cm, defacto_sp)
    if verbose:
        print("── Raw counts by source ──")
        print(df_raw.groupby(["source", "label_name"]).size(), "\n")

    df = apply_caps(df_raw, caps=caps, seed=seed)
    if verbose:
        print("── After sampling ──")
        print(df.groupby(["source", "label_name"]).size())
        print(f"\nAuth: {(df.label == 0).sum()} | Tamp: {(df.label == 1).sum()} | Total: {len(df)}")
        tamp = df[df.label == 1]
        if len(tamp):
            print(f"CASIA share of tampered: {(tamp.source == 'CASIA_v2').mean():.0%}\n")

    df_train, df_val, df_test = split_dataframe(df, seed=seed)

    if verbose:
        for nm, d in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
            print(f"{nm:5s} {len(d):5d} | Auth {(d.label == 0).sum():4d} | Tamp {(d.label == 1).sum():4d}")

    train_ds = ForgeryDataFrameDataset(df_train, transform=get_swin_transforms("train"))
    val_ds = ForgeryDataFrameDataset(df_val, transform=get_swin_transforms("val"))
    test_ds = ForgeryDataFrameDataset(df_test, transform=get_swin_transforms("test"))

    # ── class-balanced sampler, computed from TRAIN labels only ──────────
    counts = df_train["label"].value_counts().to_dict()
    class_weight = {c: 1.0 / n for c, n in counts.items()}
    sample_weights = df_train["label"].map(class_weight).tolist()

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, (df_train, df_val, df_test)


# ═════════════════════════════════════════════════════════════════════════
#  5.  OPTIMIZER — LAYER-WISE LEARNING RATE DECAY
# ═════════════════════════════════════════════════════════════════════════


def get_swin_optimizer(model: SwinBranch, base_lr: float = 1e-4) -> torch.optim.AdamW:
    """Create an AdamW optimizer with layer-wise learning-rate decay (LLRD).

    Learning-rate schedule per parameter group
    -------------------------------------------
    * Projection head (norm, projection, act, dropout): ``base_lr``
    * Swin stage 3 (layers.3):  ``base_lr × 0.5``
    * Swin stage 2 (layers.2):  ``base_lr × 0.25``
    * Swin stage 1 (layers.1):  ``base_lr × 0.125``
    * Swin stage 0 / patch embed (layers.0 + patch_embed): ``base_lr × 0.0625``

    Note: the disposable 2-class probe head used during training is added
    as an extra param group by :func:`train_swin_branch` *after* this
    function returns (so the scheduler also covers it).

    Parameters
    ----------
    model : SwinBranch
        The branch model to optimise.
    base_lr : float
        Peak learning rate, used directly for the projection head.

    Returns
    -------
    torch.optim.AdamW
    """
    # Categorise every parameter into one of 5 buckets.
    projection_params: List[torch.nn.Parameter] = []
    stage3_params: List[torch.nn.Parameter] = []
    stage2_params: List[torch.nn.Parameter] = []
    stage1_params: List[torch.nn.Parameter] = []
    stage0_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Parameters that belong to the projection head (everything
        # outside the backbone)
        if name.startswith(("norm.", "projection.", "act.", "dropout.")):
            projection_params.append(param)
        elif "layers.3" in name:
            stage3_params.append(param)
        elif "layers.2" in name:
            stage2_params.append(param)
        elif "layers.1" in name:
            stage1_params.append(param)
        else:
            # layers.0, patch_embed, pos_drop, norm (backbone-level), etc.
            stage0_params.append(param)

    param_groups = [
        {"params": projection_params, "lr": base_lr,          "name": "projection_head"},
        {"params": stage3_params,     "lr": base_lr * 0.5,    "name": "swin_stage3"},
        {"params": stage2_params,     "lr": base_lr * 0.25,   "name": "swin_stage2"},
        {"params": stage1_params,     "lr": base_lr * 0.125,  "name": "swin_stage1"},
        {"params": stage0_params,     "lr": base_lr * 0.0625, "name": "swin_stage0"},
    ]

    # Filter out empty groups (defensive)
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    return torch.optim.AdamW(param_groups, weight_decay=0.01)


# ═════════════════════════════════════════════════════════════════════════
#  6.  SCHEDULER
# ═════════════════════════════════════════════════════════════════════════


def get_swin_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """Return a OneCycleLR scheduler with cosine annealing and 10 % warm-up.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer returned by :func:`get_swin_optimizer` (with the probe
        head's param group already added, if applicable).
    epochs : int
        Total training epochs.
    steps_per_epoch : int
        Number of optimiser steps per epoch (= ``len(train_loader)``).

    Returns
    -------
    torch.optim.lr_scheduler.OneCycleLR
    """
    # max_lr per group — use each group's assigned LR
    max_lrs = [group["lr"] for group in optimizer.param_groups]

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=epochs * steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )


# ═════════════════════════════════════════════════════════════════════════
#  7.  TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════


def train_swin_branch(
    model: SwinBranch,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = NUM_EPOCHS,
    base_lr: float = LR,
    device: str = "cuda",
    save_dir: str = "./checkpoints",
    use_amp: bool = True,
) -> None:
    """Train the Swin Branch and save the best checkpoint (by val AUC).

    A 2-class linear probe (``nn.Linear(512, 2)``) is attached on top of the
    512-d feature vector purely to produce a supervised cross-entropy
    training signal. This probe's weights ARE saved in the checkpoint
    (under ``"head_state_dict"``), so :func:`evaluate_swin_branch` and
    :func:`predict_single_image` can reconstruct the *exact* trained
    classifier instead of scoring through a freshly-initialised random head.

    Parameters
    ----------
    model : SwinBranch
        The model to train.
    train_loader : DataLoader
        Training data loader (should use weighted sampling — see
        :func:`build_dataframe_loaders`).
    val_loader : DataLoader
        Validation data loader.
    epochs : int
        Number of training epochs.
    base_lr : float
        Base learning rate for the projection head (and probe head).
    device : str
        ``'cuda'`` or ``'cpu'``.
    save_dir : str
        Directory in which to save ``swin_branch_best.pth``.
    use_amp : bool
        Whether to use automatic mixed-precision training.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    # ── Disposable 2-class probe head (saved in the checkpoint) ──────────
    head = nn.Linear(512, 2).to(device)

    # ── Loss with class weights (authentic=1.0, forged=2.5) ──────────────
    class_weights = torch.tensor([1.0, 2.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = get_swin_optimizer(model, base_lr=base_lr)
    optimizer.add_param_group({"params": head.parameters(), "lr": base_lr, "name": "probe_head"})
    scheduler = get_swin_scheduler(optimizer, epochs, len(train_loader))

    scaler = GradScaler(enabled=use_amp)

    best_auc: float = 0.0

    for epoch in range(1, epochs + 1):
        # ── Training phase ───────────────────────────────────────────────
        model.train()
        head.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                features = model(images)   # [B, 512]
                logits = head(features)    # [B, 2]
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # ── Validation phase ─────────────────────────────────────────────
        val_loss, val_acc, val_auc = _validate(
            model, head, val_loader, criterion, device, use_amp
        )

        # ── Checkpoint ───────────────────────────────────────────────────
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = os.path.join(save_dir, "swin_branch_best.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )

        # ── Epoch summary ────────────────────────────────────────────────
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )


# ── Training helpers (private) ───────────────────────────────────────────


@torch.no_grad()
def _validate(
    model: SwinBranch,
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Run one validation pass and return ``(loss, accuracy, auc)``."""
    model.eval()
    head.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            features = model(images)
            logits = head(features)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    n = max(len(all_labels), 1)
    avg_loss = running_loss / n
    acc = accuracy_score(all_labels, all_preds)

    # AUC requires both classes present
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, acc, auc


def _load_branch_and_head(model_path: str, device: str) -> Tuple[SwinBranch, nn.Module]:
    """Reconstruct the trained ``SwinBranch`` + 2-class probe head from a
    checkpoint produced by :func:`train_swin_branch`."""
    model = SwinBranch(pretrained=False)
    head = nn.Linear(512, 2)

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    if "head_state_dict" not in ckpt:
        raise KeyError(
            "Checkpoint has no 'head_state_dict'. It was likely produced by "
            "an older version of train_swin_branch that did not save the "
            "probe head — retrain to get a checkpoint compatible with "
            "evaluate_swin_branch / predict_single_image."
        )
    head.load_state_dict(ckpt["head_state_dict"])

    model = model.to(device).eval()
    head = head.to(device).eval()
    return model, head


# ═════════════════════════════════════════════════════════════════════════
#  8.  EVALUATION
# ═════════════════════════════════════════════════════════════════════════


def evaluate_swin_branch(
    model_path: str,
    test_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load a saved checkpoint and evaluate on a test set.

    Parameters
    ----------
    model_path : str
        Path to ``swin_branch_best.pth``.
    test_loader : DataLoader
        Test-set ``DataLoader``.
    device : str
        ``'cuda'`` or ``'cpu'``.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``auc``, ``precision``, ``recall``, ``f1``,
        ``confusion_matrix``.
    """
    model, head = _load_branch_and_head(model_path, device)

    all_labels: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            features = model(images)
            logits = head(features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # ── Formatted report ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Swin Branch — Test Evaluation")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds, target_names=["Authentic", "Forged"],
        zero_division=0,
    ))
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("=" * 60 + "\n")

    return {
        "accuracy": acc,
        "auc": auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ═════════════════════════════════════════════════════════════════════════
#  9.  SINGLE-IMAGE INFERENCE
# ═════════════════════════════════════════════════════════════════════════


def predict_single_image(
    model_path: str,
    image_path: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run inference on a single image and return label, confidence,
    and the 512-d feature vector for downstream fusion.

    Parameters
    ----------
    model_path : str
        Path to ``swin_branch_best.pth``.
    image_path : str
        Path to the input image.
    device : str
        ``'cuda'`` or ``'cpu'``.

    Returns
    -------
    dict
        ``{'label': str, 'confidence': float, 'feature_vector': np.ndarray}``
        where ``feature_vector`` has shape ``(512,)``.
    """
    model, head = _load_branch_and_head(model_path, device)

    transform = get_swin_transforms(mode="test")
    image = _safe_load_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        feature_vector = model(tensor)         # [1, 512]
        logits = head(feature_vector)           # [1, 2]
        probs = torch.softmax(logits, dim=1).squeeze(0)  # [2]

    forged_prob = probs[1].item()
    label = "Forged" if forged_prob >= 0.5 else "Authentic"

    return {
        "label": label,
        "confidence": forged_prob if label == "Forged" else 1.0 - forged_prob,
        "feature_vector": feature_vector.squeeze(0).cpu().numpy(),
    }


# ═════════════════════════════════════════════════════════════════════════
#  10.  SMOKE TEST / EXAMPLE TRAINING ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Swin Branch — Smoke Test")
    print("=" * 60)

    branch = SwinBranch(pretrained=False)
    dummy_input = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        output = branch(dummy_input)

    assert output.shape == (4, 512), (
        f"Expected (4, 512), got {output.shape}"
    )
    print(f"SwinBranch output shape: {output.shape} ✓")

    total_params = sum(p.numel() for p in branch.parameters() if p.requires_grad)
    print(f"Trainable parameters:   {total_params:,}")
    print("=" * 60)

    # ── Example: real training run on Kaggle ─────────────────────────────
    # Uncomment inside a Kaggle notebook with CASIA_AU/CASIA_TP/DEFACTO_CM/
    # DEFACTO_SP mounted at the paths defined near the top of this file
    # (edit those constants if your mount points differ — just keep them
    # identical to whatever the CNN branch uses).
    #
    # train_loader, val_loader, test_loader, (df_train, df_val, df_test) = (
    #     build_dataframe_loaders(batch_size=BATCH_SIZE, num_workers=2)
    # )
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = SwinBranch(pretrained=True).to(device)
    #
    # train_swin_branch(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     epochs=NUM_EPOCHS,
    #     base_lr=LR,
    #     device=device,
    #     save_dir="./checkpoints",
    #     use_amp=(device == "cuda"),
    # )
    #
    # evaluate_swin_branch("./checkpoints/swin_branch_best.pth", test_loader, device=device)