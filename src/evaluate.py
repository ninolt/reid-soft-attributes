"""
Evaluation utilities for ReID metrics (rank-1/5/10 and mAP).
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch

from src.config import CFG
from src.utils import _append_metrics_log, _get_iqa_scores_from_cache, _unpack_batch, get_logger

logger = get_logger("reid.evaluate")

def _extract_features(
    model,
    loader: Iterable,
    device: torch.device,
    iqa_cache: dict[str, float],
):
    model.eval()
    feats = []
    labels = []
    cams = []
    with torch.no_grad():
        for batch in loader:
            images, batch_labels, batch_cams, img_paths, _ = _unpack_batch(batch)
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            iqa_scores = _get_iqa_scores_from_cache(
                img_paths=img_paths,
                cache=iqa_cache,
                device=device,
                dtype=images.dtype,
            )
            _, _, _, char_vec = model(images, iqa_score=iqa_scores, is_train=False)
            feats.append(char_vec.detach().cpu())
            labels.append(batch_labels.detach().cpu())
            if batch_cams is not None:
                cams.append(torch.as_tensor(batch_cams).cpu())
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    cams = torch.cat(cams, dim=0) if cams else None
    # L2 normalize for cosine distance
    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
    return feats, labels, cams


def _compute_cmc_map(distmat, q_labels, g_labels, q_cams, g_cams, max_rank=CFG["evaluation"]["max_rank"]):
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    matches = (g_labels[indices] == q_labels[:, None])

    if q_cams is None or g_cams is None:
        invalid = np.zeros_like(matches, dtype=bool)
    else:
        invalid = (g_cams[indices] == q_cams[:, None]) & matches

    cmc = np.zeros(max_rank)
    all_ap = []
    valid_q = 0
    for q_idx in range(num_q):
        valid = ~invalid[q_idx]
        match = matches[q_idx][valid]
        if not np.any(match):
            continue
        valid_q += 1
        first_hit = np.where(match)[0][0]
        if first_hit < max_rank:
            cmc[first_hit:] += 1
        num_rel = match.sum()
        precision = np.cumsum(match) / (np.arange(len(match)) + 1)
        ap = (precision * match).sum() / num_rel
        all_ap.append(ap)
    if valid_q == 0:
        return np.zeros(max_rank), 0.0
    cmc = cmc / valid_q
    mAP = float(np.mean(all_ap)) if all_ap else 0.0
    return cmc, mAP


def evaluate_reid(
    model,
    query_loader,
    gallery_loader,
    device,
    iqa_cache: dict[str, float],
    log_path: str | None = None,
    log_dir: str | None = None,
):
    qf, ql, qc = _extract_features(model, query_loader, device, iqa_cache)
    gf, gl, gc = _extract_features(model, gallery_loader, device, iqa_cache)

    distmat = 1.0 - torch.mm(qf, gf.t()).cpu().numpy()
    ql = ql.numpy()
    gl = gl.numpy()
    qc = qc.numpy() if qc is not None else None
    gc = gc.numpy() if gc is not None else None

    cmc, mAP = _compute_cmc_map(distmat, ql, gl, qc, gc, max_rank=10)
    logger.info(
        "[Eval] Rank-1: %.2f%% | Rank-5: %.2f%% | Rank-10: %.2f%% | mAP: %.2f%%",
        cmc[0] * 100, cmc[4] * 100, cmc[9] * 100, mAP * 100,
    )
    metrics = {
        "rank1": cmc[0],
        "rank5": cmc[4],
        "rank10": cmc[9],
        "mAP": mAP,
    }
    if log_dir:
        _append_metrics_log(os.path.join(log_dir, "global.jsonl"), {"scope": "global", **metrics})
    elif log_path:
        _append_metrics_log(log_path, {"scope": "global", **metrics})
    return metrics
