from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import torch

from src.config import CFG
from src.dataset import build_dataloaders
from src.iqa import compute_iqa_scores
from src.model.model import GlobalReIDNetwork
from src.train import (
    FullTrainConfig,
    train_full_model,
)
from src.utils import get_logger

logger = get_logger("reid.main")


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _save_summary(path: str, payload: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _load_best_metrics(log_path: str) -> dict:
    if not log_path or not os.path.exists(log_path):
        return {}
    best = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            scope = rec.get("scope")
            if scope == "apn":
                best.setdefault(scope, {})
                best[scope]["rank1"] = max(best[scope].get("rank1", 0.0), rec.get("rank1", 0.0))
                best[scope]["mAP"] = max(best[scope].get("mAP", 0.0), rec.get("mAP", 0.0))
            elif scope == "global":
                best.setdefault(scope, {})
                best[scope]["rank1"] = max(best[scope].get("rank1", 0.0), rec.get("rank1", 0.0))
                best[scope]["mAP"] = max(best[scope].get("mAP", 0.0), rec.get("mAP", 0.0))
            elif scope == "apan":
                best.setdefault(scope, {})
                best[scope]["acc_bin"] = max(best[scope].get("acc_bin", 0.0), rec.get("acc_bin", 0.0))
                best[scope]["acc_multi"] = max(best[scope].get("acc_multi", 0.0), rec.get("acc_multi", 0.0))
    return best


def main():
    """Main training and evaluation pipeline for Market-1501 Person Re-ID."""
    start = time.time()

    # Dataset configuration
    data_dir: str = CFG["dataset"]["data_dir"]

    # Logging
    metrics_dir: str = CFG["logging"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)

    full_cfg = FullTrainConfig(
        metrics_log_dir=metrics_dir,
    )

    # --- Global training ---
    train_loader_full, gallery_loader_full, query_loader_full, num_classes_full = build_dataloaders(
        data_dir=data_dir,
    )
    
    # Compute/update IQA cache for all splits before training
    logger.info("Computing IQA scores for training set...")
    compute_iqa_scores(train_loader_full.dataset, cache_file=full_cfg.iqa_cache_path)
    logger.info("Computing IQA scores for gallery set...")
    compute_iqa_scores(gallery_loader_full.dataset, cache_file=full_cfg.iqa_cache_path)
    logger.info("Computing IQA scores for query set...")
    compute_iqa_scores(query_loader_full.dataset, cache_file=full_cfg.iqa_cache_path)
    
    global_model = GlobalReIDNetwork(num_classes=num_classes_full, shared_backbone=full_cfg.shared_backbone)
    train_full_model(
        global_model,
        train_loader_full,
        full_cfg,
        query_loader=query_loader_full,
        gallery_loader=gallery_loader_full,
    )

    # --- Stats summary ---
    duration = time.time() - start
    stats = {
        "duration_sec": duration,
        "params": {
            "global_model": _count_params(global_model),
            "global_backbone_apn": _count_params(global_model.backbone_apn),
            "global_backbone_apan": _count_params(global_model.backbone_apan),
        },
        "configs": {
            "global": asdict(full_cfg),
        },
        "best_metrics": {
            "apn": _load_best_metrics(os.path.join(metrics_dir, "apn.jsonl")).get("apn", {}),
            "apan": _load_best_metrics(os.path.join(metrics_dir, "apan.jsonl")).get("apan", {}),
            "global": _load_best_metrics(os.path.join(metrics_dir, "global.jsonl")).get("global", {}),
        },
    }

    logger.info("=== FINAL SUMMARY ===")
    logger.info(json.dumps(stats, indent=2))
    _save_summary(os.path.join(metrics_dir, "summary.json"), stats)


if __name__ == "__main__":
    main()
