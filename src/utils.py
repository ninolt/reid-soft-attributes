"""
Shared training utilities.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch


def get_logger(name: str = "reid") -> logging.Logger:
    """Returns a named logger configured with a standard console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _unpack_batch(batch: Any):
    if not isinstance(batch, (list, tuple)):
        raise ValueError(f"Unexpected batch type: {type(batch)}")
    if len(batch) >= 4:
        images, labels, camids, img_paths = batch[:4]
        attrs = batch[4] if len(batch) > 4 else None
        return images, labels, camids, img_paths, attrs
    if len(batch) == 2:
        images, labels = batch
        return images, labels, None, None, None
    raise ValueError(f"Unexpected batch format length: {len(batch)}")


def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _load_iqa_cache(cache_path: str) -> dict[str, float]:
    if not cache_path or not os.path.exists(cache_path):
        raise FileNotFoundError(f"IQA cache not found at: {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    if isinstance(cache, np.ndarray):
        try:
            cache = cache.item()
        except Exception:
            raise ValueError("IQA cache format not recognized (expected dict-like object).")
    if isinstance(cache, dict):
        return {str(k): float(v) for k, v in cache.items()}
    raise ValueError("IQA cache format not recognized (expected dict-like object).")


def _get_iqa_scores_from_cache(
    img_paths: list[str] | None,
    cache: dict[str, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not img_paths:
        raise ValueError("img_paths required to fetch IQA scores from cache.")
    scores = []
    missing = []
    for p in img_paths:
        key = os.path.basename(str(p))
        if key not in cache:
            missing.append(key)
        else:
            scores.append(cache[key])
    if missing:
        raise KeyError(f"Missing IQA scores for {len(missing)} image paths in cache.")
    return torch.tensor(scores, device=device, dtype=dtype)


def _compute_apan_loss(
    logits_dict: dict[str, torch.Tensor],
    attrs: dict[str, Any] | None,
    device: torch.device,
    criterion_bce: torch.nn.Module,
    criterion_ce: torch.nn.Module,
    binary_keys: list[str],
    multi_keys: dict[str, int],
    return_per_sample: bool = False,
):
    if attrs is None:
        return (None, 0.0, 0.0, None) if return_per_sample else (None, 0.0, 0.0)

    losses = []
    losses_per_sample = []
    bin_correct = 0
    bin_total = 0
    multi_correct = 0
    multi_total = 0

    for key in binary_keys:
        if key not in logits_dict or key not in attrs:
            continue
        pred = logits_dict[key]
        target = _to_tensor(attrs[key], device=device, dtype=torch.float32)
        if pred.shape != target.shape:
            target = target.view_as(pred)
        loss_ps = criterion_bce(pred, target)
        losses.append(loss_ps.mean())
        if return_per_sample:
            losses_per_sample.append(loss_ps.view(loss_ps.size(0), -1).mean(dim=1))
        with torch.no_grad():
            bin_correct += ((pred > 0).float() == target).sum().item()
            bin_total += target.numel()

    for key, ncls in multi_keys.items():
        if key not in logits_dict or key not in attrs:
            continue
        pred = logits_dict[key]
        target = _to_tensor(attrs[key], device=device, dtype=torch.long)
        if pred.dim() != 2 or pred.size(1) != ncls:
            continue
        if (target < 0).any() or (target >= ncls).any():
            continue
        loss_ps = criterion_ce(pred, target)
        losses.append(loss_ps.mean())
        if return_per_sample:
            losses_per_sample.append(loss_ps)
        with torch.no_grad():
            multi_correct += (pred.argmax(dim=1) == target).sum().item()
            multi_total += target.numel()

    if not losses:
        return (None, 0.0, 0.0, None) if return_per_sample else (None, 0.0, 0.0)
    loss = sum(losses) / len(losses)
    bin_acc = bin_correct / bin_total if bin_total > 0 else 0.0
    multi_acc = multi_correct / multi_total if multi_total > 0 else 0.0
    if not return_per_sample:
        return loss, bin_acc, multi_acc
    loss_per_sample = torch.stack(losses_per_sample).mean(dim=0) if losses_per_sample else None
    return loss, bin_acc, multi_acc, loss_per_sample


def _append_metrics_log(log_path: str, payload: dict):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays and torch tensors
            return obj.tolist()
        elif isinstance(obj, (float, int, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    record = convert_to_serializable(record)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
