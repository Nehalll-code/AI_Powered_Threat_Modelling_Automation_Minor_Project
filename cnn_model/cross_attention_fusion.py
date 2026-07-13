"""
cross_attention_fusion.py — Cross-Attention Fusion Module
==========================================================

Part of a hybrid image-forgery-detection system.  This module provides:

* **CrossAttentionFusion** — a Transformer-based fusion network that receives
  three 512-dimensional feature vectors (from CNN, ViT/Swin, and Frequency
  branches), models their multi-modal correlations via self-attention, and
  outputs a unified 512-d fused representation plus optional classification
  logits.

Architecture Summary
--------------------
1. Stack the three ``[B, 512]`` vectors into ``[B, 3, 512]``.
2. Add a learnable positional embedding ``[1, 3, 512]`` (one per branch).
3. Pass through a 2-layer Pre-LN Transformer Encoder (8 heads, GELU, 0.1 dropout).
4. Global Average Pool along the sequence dimension → ``[B, 512]``.
5. Optional ``nn.Linear(512, num_classes)`` classifier head.

Usage
-----
>>> from cross_attention_fusion import CrossAttentionFusion
>>> fusion = CrossAttentionFusion(num_classes=2)
>>> fused_feat, logits = fusion(cnn_feat, vit_feat, freq_feat)

Compatible with Python ≥ 3.9, PyTorch ≥ 2.0.
"""

from __future__ import annotations

# ── stdlib ───────────────────────────────────────────────────────────────
import os
from typing import Any, Dict, List, Optional, Tuple

# ── torch ────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

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


# ═════════════════════════════════════════════════════════════════════════
#  1.  MODEL
# ═════════════════════════════════════════════════════════════════════════


class CrossAttentionFusion(nn.Module):
    """Transformer-based multi-modal fusion for image forgery detection.

    Receives pre-computed feature vectors from three parallel branches
    (CNN, ViT/Swin, Frequency), stacks them into a 3-token sequence,
    applies self-attention so every branch can attend to every other branch,
    and produces a single fused 512-d vector via global average pooling.

    Parameters
    ----------
    embed_dim : int, default ``512``
        Dimensionality of each branch's feature vector and the internal
        transformer representation.
    num_heads : int, default ``8``
        Number of attention heads.  ``512 / 8 = 64`` dims per head, the
        standard size used in ViT / BERT.
    num_layers : int, default ``2``
        Number of stacked Transformer Encoder blocks.  Kept small because
        the sequence length is only 3 — deeper stacks risk over-fitting.
    dim_feedforward : int, default ``1024``
        Hidden size of the position-wise FFN inside each encoder block.
    dropout : float, default ``0.1``
        Dropout probability applied inside attention and FFN sublayers.
    num_classes : int or None, default ``2``
        If > 0, a ``nn.Linear(embed_dim, num_classes)`` classification head
        is appended.  Set to ``0`` or ``None`` to disable it (the module
        then returns ``None`` for the logits).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_classes: Optional[int] = 2,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers_count = num_layers
        self.num_classes = num_classes

        # ── Learnable Positional Embedding ───────────────────────────────
        # Shape [1, 3, embed_dim] — one vector per branch position:
        #   index 0 → CNN,  index 1 → ViT/Swin,  index 2 → Frequency
        self.pos_embed = nn.Parameter(torch.zeros(1, 3, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Pre-LayerNorm Transformer Encoder ────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            layer_norm_eps=1e-5,
            batch_first=True,   # input / output shape: [B, Seq, D]
            norm_first=True,    # Pre-LN for more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Final LayerNorm (post-transformer stabilisation) ─────────────
        self.output_norm = nn.LayerNorm(embed_dim)

        # ── Optional Classification Head ─────────────────────────────────
        if num_classes is not None and num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
        else:
            self.classifier = None

    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        cnn_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        freq_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fuse the three branch features via multi-head self-attention.

        Parameters
        ----------
        cnn_feat : torch.Tensor
            CNN branch features, shape ``[B, 512]``.
        vit_feat : torch.Tensor
            ViT / Swin branch features, shape ``[B, 512]``.
        freq_feat : torch.Tensor
            Frequency branch features, shape ``[B, 512]``.

        Returns
        -------
        fused_features : torch.Tensor
            Shape ``[B, 512]`` — the unified multi-modal representation.
        logits : torch.Tensor or None
            Shape ``[B, num_classes]`` if a classifier head is present,
            otherwise ``None``.
        """
        # ── Input validation ─────────────────────────────────────────────
        for name, feat in [("cnn", cnn_feat), ("vit", vit_feat), ("freq", freq_feat)]:
            if feat.ndim != 2 or feat.shape[1] != self.embed_dim:
                raise ValueError(
                    f"Expected '{name}' shape [B, {self.embed_dim}], "
                    f"got {list(feat.shape)}"
                )

        # ── Stack → [B, 3, 512] ──────────────────────────────────────────
        x = torch.stack([cnn_feat, vit_feat, freq_feat], dim=1)

        # ── Add positional identity ──────────────────────────────────────
        x = x + self.pos_embed

        # ── Self-Attention across modalities ─────────────────────────────
        x = self.transformer(x)             # [B, 3, 512]

        # ── Global Average Pooling ───────────────────────────────────────
        fused_features = x.mean(dim=1)      # [B, 512]
        fused_features = self.output_norm(fused_features)

        # ── Classification (optional) ────────────────────────────────────
        logits: Optional[torch.Tensor] = None
        if self.classifier is not None:
            logits = self.classifier(fused_features)    # [B, num_classes]

        return fused_features, logits

    # ─────────────────────────────────────────────────────────────────────

    def get_attention_weights(
        self,
        cnn_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        freq_feat: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Extract per-layer attention weight matrices for interpretability.

        Useful for downstream XAI analysis (e.g., which branch attends most
        strongly to which).

        Parameters
        ----------
        cnn_feat, vit_feat, freq_feat : torch.Tensor
            Same as :meth:`forward`.

        Returns
        -------
        list[torch.Tensor]
            One ``[B, num_heads, 3, 3]`` tensor per encoder layer.
        """
        x = torch.stack([cnn_feat, vit_feat, freq_feat], dim=1)
        x = x + self.pos_embed

        attn_weights: List[torch.Tensor] = []
        for layer in self.transformer.layers:
            # Manually call the self-attention sublayer to capture weights
            x_normed = layer.norm1(x)
            _, w = layer.self_attn(
                x_normed, x_normed, x_normed,
                need_weights=True,
                average_attn_weights=False,
            )
            attn_weights.append(w)  # [B, num_heads, 3, 3]
            # Continue the forward pass through the rest of the layer
            x = x + layer.dropout1(
                layer.self_attn(x_normed, x_normed, x_normed)[0]
            )
            x = x + layer._ff_block(layer.norm2(x))

        return attn_weights


# ═════════════════════════════════════════════════════════════════════════
#  2.  OPTIMIZER
# ═════════════════════════════════════════════════════════════════════════


def get_fusion_optimizer(
    model: CrossAttentionFusion,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Create an AdamW optimizer for the fusion module.

    Two parameter groups are used:

    * **Classifier head** (if present): full ``lr``.
    * **Transformer + positional embeddings**: ``lr * 0.5`` (slightly lower
      so the self-attention layers converge stably while the head adapts).

    Parameters
    ----------
    model : CrossAttentionFusion
        The fusion module.
    lr : float
        Peak learning rate for the classifier head.
    weight_decay : float
        L2 regularisation coefficient.

    Returns
    -------
    torch.optim.AdamW
    """
    classifier_params: List[nn.Parameter] = []
    transformer_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier."):
            classifier_params.append(param)
        else:
            transformer_params.append(param)

    param_groups = [
        {"params": transformer_params, "lr": lr * 0.5, "name": "transformer"},
        {"params": classifier_params,  "lr": lr,       "name": "classifier"},
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ═════════════════════════════════════════════════════════════════════════
#  3.  SCHEDULER
# ═════════════════════════════════════════════════════════════════════════


def get_fusion_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """Return a OneCycleLR scheduler with 10 % cosine warm-up.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer returned by :func:`get_fusion_optimizer`.
    epochs : int
        Total training epochs.
    steps_per_epoch : int
        Number of optimiser steps per epoch (``len(train_loader)``).

    Returns
    -------
    torch.optim.lr_scheduler.OneCycleLR
    """
    max_lrs = [group["lr"] for group in optimizer.param_groups]
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=epochs * steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )


# ═════════════════════════════════════════════════════════════════════════
#  4.  TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════


def train_fusion(
    model: CrossAttentionFusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 3e-4,
    device: str = "cuda",
    save_dir: str = "./checkpoints",
    use_amp: bool = True,
) -> None:
    """Train the Cross-Attention Fusion module end-to-end.

    **Data contract**: both ``train_loader`` and ``val_loader`` must yield
    batches of ``(cnn_feat, vit_feat, freq_feat, label)`` where each feature
    tensor has shape ``[B, 512]`` and ``label`` is a ``LongTensor`` of shape
    ``[B]`` with values in ``{0, 1}`` (authentic / forged).

    The best checkpoint (by validation AUC) is saved to
    ``<save_dir>/fusion_best.pth``.

    Parameters
    ----------
    model : CrossAttentionFusion
        The fusion module (must have a classifier head, i.e. ``num_classes > 0``).
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    epochs : int
        Number of training epochs.
    lr : float
        Base learning rate for the classifier head.
    device : str
        ``'cuda'`` or ``'cpu'``.
    save_dir : str
        Directory for saving checkpoints.
    use_amp : bool
        Whether to enable automatic mixed-precision training.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    # ── Loss with class weights (authentic=1.0, forged=2.5) ──────────────
    class_weights = torch.tensor([1.0, 2.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = get_fusion_optimizer(model, lr=lr)
    scheduler = get_fusion_scheduler(optimizer, epochs, len(train_loader))
    scaler = GradScaler(enabled=use_amp)

    best_auc: float = 0.0

    for epoch in range(1, epochs + 1):
        # ── Training ─────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for cnn_feat, vit_feat, freq_feat, labels in train_loader:
            cnn_feat = cnn_feat.to(device, non_blocking=True)
            vit_feat = vit_feat.to(device, non_blocking=True)
            freq_feat = freq_feat.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                _, logits = model(cnn_feat, vit_feat, freq_feat)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # ── Validation ───────────────────────────────────────────────────
        val_loss, val_acc, val_auc = _validate_fusion(
            model, val_loader, criterion, device, use_amp
        )

        # ── Checkpoint (best AUC) ────────────────────────────────────────
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                },
                os.path.join(save_dir, "fusion_best.pth"),
            )

        # ── Epoch summary ────────────────────────────────────────────────
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )


@torch.no_grad()
def _validate_fusion(
    model: CrossAttentionFusion,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """Run one validation pass.  Returns ``(loss, accuracy, auc)``."""
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    for cnn_feat, vit_feat, freq_feat, labels in loader:
        cnn_feat = cnn_feat.to(device, non_blocking=True)
        vit_feat = vit_feat.to(device, non_blocking=True)
        freq_feat = freq_feat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            _, logits = model(cnn_feat, vit_feat, freq_feat)
            loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    n = max(len(all_labels), 1)
    avg_loss = running_loss / n
    acc = accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, acc, auc


# ═════════════════════════════════════════════════════════════════════════
#  5.  EVALUATION
# ═════════════════════════════════════════════════════════════════════════


def evaluate_fusion(
    model_path: str,
    test_loader: DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Load a saved fusion checkpoint and evaluate on a test set.

    **Data contract**: ``test_loader`` must yield
    ``(cnn_feat, vit_feat, freq_feat, label)`` per batch.

    Parameters
    ----------
    model_path : str
        Path to ``fusion_best.pth``.
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
    model = CrossAttentionFusion(num_classes=2)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_labels: List[int] = []
    all_probs: List[float] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for cnn_feat, vit_feat, freq_feat, labels in test_loader:
            cnn_feat = cnn_feat.to(device, non_blocking=True)
            vit_feat = vit_feat.to(device, non_blocking=True)
            freq_feat = freq_feat.to(device, non_blocking=True)

            _, logits = model(cnn_feat, vit_feat, freq_feat)
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

    print("\n" + "=" * 60)
    print("  Cross-Attention Fusion — Test Evaluation")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds,
        target_names=["Authentic", "Forged"],
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
#  6.  FEATURE EXTRACTION UTILITY
# ═════════════════════════════════════════════════════════════════════════


def extract_fused_features(
    model: CrossAttentionFusion,
    cnn_feat: torch.Tensor,
    vit_feat: torch.Tensor,
    freq_feat: torch.Tensor,
) -> np.ndarray:
    """Extract the 512-d fused feature vector (no classification head).

    This is the primary interface for the downstream Point-of-Decision
    gate and XAI modules (TCAV, MAGE).

    Parameters
    ----------
    model : CrossAttentionFusion
        A loaded fusion module (on the correct device).
    cnn_feat, vit_feat, freq_feat : torch.Tensor
        Branch features, each of shape ``[B, 512]`` (or ``[1, 512]`` for
        a single image).

    Returns
    -------
    np.ndarray
        Fused features of shape ``(B, 512)`` or ``(512,)`` if B == 1.
    """
    model.eval()
    with torch.no_grad():
        fused, _ = model(cnn_feat, vit_feat, freq_feat)
    result = fused.cpu().numpy()
    if result.shape[0] == 1:
        return result.squeeze(0)    # (512,)
    return result                   # (B, 512)


# ═════════════════════════════════════════════════════════════════════════
#  7.  SINGLE-IMAGE INFERENCE (END-TO-END EXAMPLE)
# ═════════════════════════════════════════════════════════════════════════


def predict_single_image_fusion(
    fusion_model_path: str,
    cnn_feat: torch.Tensor,
    vit_feat: torch.Tensor,
    freq_feat: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run inference on a single image's pre-extracted branch features.

    Parameters
    ----------
    fusion_model_path : str
        Path to ``fusion_best.pth``.
    cnn_feat, vit_feat, freq_feat : torch.Tensor
        Pre-extracted branch features, each of shape ``[1, 512]``.
    device : str
        ``'cuda'`` or ``'cpu'``.

    Returns
    -------
    dict
        ``{'label': str, 'confidence': float, 'fused_feature': np.ndarray}``
        where ``fused_feature`` has shape ``(512,)`` for downstream use.
    """
    model = CrossAttentionFusion(num_classes=2)
    ckpt = torch.load(fusion_model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    cnn_feat = cnn_feat.to(device)
    vit_feat = vit_feat.to(device)
    freq_feat = freq_feat.to(device)

    with torch.no_grad():
        fused, logits = model(cnn_feat, vit_feat, freq_feat)
        probs = torch.softmax(logits, dim=1).squeeze(0)    # [2]

    forged_prob = probs[1].item()
    label = "Forged" if forged_prob >= 0.5 else "Authentic"

    return {
        "label": label,
        "confidence": forged_prob if label == "Forged" else 1.0 - forged_prob,
        "fused_feature": fused.squeeze(0).cpu().numpy(),    # (512,)
    }


# ═════════════════════════════════════════════════════════════════════════
#  8.  SMOKE TEST
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Cross-Attention Fusion — Smoke Test")
    print("=" * 60)

    # ── Instantiate ──────────────────────────────────────────────────────
    fusion = CrossAttentionFusion(
        embed_dim=512,
        num_heads=8,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        num_classes=2,
    )

    # ── Mock branch outputs [B=4, D=512] ─────────────────────────────────
    B = 4
    mock_cnn  = torch.randn(B, 512)
    mock_vit  = torch.randn(B, 512)
    mock_freq = torch.randn(B, 512)

    # ── Forward pass ─────────────────────────────────────────────────────
    with torch.no_grad():
        fused, logits = fusion(mock_cnn, mock_vit, mock_freq)

    # ── Shape assertions ─────────────────────────────────────────────────
    assert fused.shape == (B, 512), (
        f"Expected fused shape ({B}, 512), got {fused.shape}"
    )
    assert logits is not None and logits.shape == (B, 2), (
        f"Expected logits shape ({B}, 2), got "
        f"{logits.shape if logits is not None else None}"
    )

    print(f"Fused feature shape:   {fused.shape} OK")
    print(f"Logits shape:          {logits.shape} OK")

    # ── Attention weight extraction ──────────────────────────────────────
    attn_weights = fusion.get_attention_weights(mock_cnn, mock_vit, mock_freq)
    print(f"Attention layers:      {len(attn_weights)}")
    print(f"Attn weight shape:     {attn_weights[0].shape}  "
          f"(B, heads, seq, seq)")

    # ── Feature extraction utility ───────────────────────────────────────
    single_feat = extract_fused_features(
        fusion,
        mock_cnn[:1], mock_vit[:1], mock_freq[:1],
    )
    assert single_feat.shape == (512,), (
        f"Expected (512,), got {single_feat.shape}"
    )
    print(f"Single-image feature:  {single_feat.shape} OK")

    # ── Parameter count ──────────────────────────────────────────────────
    total = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    print(f"Trainable parameters:  {total:,}")
    print("=" * 60)
