"""
Full model training workflow with separate backward passes for:
- APN branch
- APAN branch
- Global FC (after fusion)
- Alpha perceptron / fusion module
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import CFG
from src.evaluate import evaluate_reid
from src.model.model import GlobalReIDNetwork
from src.utils import (
    _compute_apan_loss,
    _get_iqa_scores_from_cache,
    _load_iqa_cache,
    _unpack_batch,
    get_logger,
)

logger = get_logger("reid.train")


class TripletLoss(nn.Module):
    """Triplet loss with hard mining (Bag of Tricks for ReID)."""
    def __init__(self, margin: float = CFG["training"]["triplet_margin"]):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss with batch hard mining."""
        dist = torch.cdist(features, features, p=2)
        
        # For each anchor, find hardest positive and negative
        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_neg = ~mask_pos
        
        # Hardest positive: max distance among same identity
        dist_ap = dist.clone()
        dist_ap[~mask_pos] = 0
        dist_ap, _ = dist_ap.max(dim=1)
        
        # Hardest negative: min distance among different identity
        dist_an = dist.clone()
        dist_an[~mask_neg] = float('inf')
        dist_an, _ = dist_an.min(dim=1)
        
        # Filter out invalid triplets (where no positive or negative exists)
        valid = (dist_ap > 0) & (dist_an < float('inf'))
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        dist_ap = dist_ap[valid]
        dist_an = dist_an[valid]
        
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing (Bag of Tricks for ReID)."""
    def __init__(self, smoothing: float = CFG["training"]["label_smoothing"]):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()


@dataclass
class FullTrainConfig:
    epochs: int = CFG["training"]["epochs"]
    lr: float = CFG["training"]["lr"]
    weight_decay: float = CFG["training"]["weight_decay"]
    log_interval: int = CFG["training"]["log_interval"]
    eval_interval: int = CFG["training"]["eval_interval"]
    save_dir: str = CFG["training"]["save_dir"]
    save_every: int = CFG["training"]["save_every"]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_iqa_cache: bool = True
    iqa_cache_path: str = CFG["iqa"]["cache_path"]
    train_apn_backbone: bool = True
    train_apan_backbone: bool = True
    shared_backbone: bool = False
    load_pretrained_branches: bool = False
    apn_checkpoint_path: str = CFG["training"]["apn_checkpoint_path"]
    apan_checkpoint_path: str = CFG["training"]["apan_checkpoint_path"]
    freeze_epochs: int = CFG["training"]["freeze_epochs"]
    branch_lr_scale: float = CFG["training"]["branch_lr_scale"]
    metrics_log_path: str | None = None
    metrics_log_dir: str | None = None
    warmup_epochs: int = CFG["training"]["warmup_epochs"]
    triplet_margin: float = CFG["training"]["triplet_margin"]
    triplet_weight: float = CFG["training"]["triplet_weight"]
    label_smoothing: float = CFG["training"]["label_smoothing"]
    center_loss_weight: float = CFG["training"]["center_loss_weight"]


def _save_checkpoint(
    model: nn.Module,
    optimizers: dict[str, optim.Optimizer],
    epoch: int,
    save_dir: str,
    filename: str,
):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    state = {"epoch": epoch, "model_state_dict": model.state_dict()}
    for name, opt in optimizers.items():
        state[f"optimizer_state_dict_{name}"] = opt.state_dict()
    torch.save(state, path)


def _forward_full(
    model: GlobalReIDNetwork,
    images: torch.Tensor,
    iqa_score: torch.Tensor | list[float] | None,
    detach_branch_features_for_fusion: bool = True,
):
    feats_apn = model.backbone_apn(images)
    # Use layer3 only for APN (1024 channels)
    apn_x = feats_apn.get("layer3")
    logits_apn, visual_feat = model.apn(apn_x, is_train=True)

    feats_apan = model.backbone_apan(images)
    logits_apan, attr_feat = model.apan(feats_apan.get("layer4"))

    iqa_score = model._prepare_iqa_score(
        iqa_score,
        batch_size=visual_feat.size(0),
        device=visual_feat.device,
        dtype=visual_feat.dtype,
    )
    alpha = model.alpha_head(iqa_score)

    if detach_branch_features_for_fusion:
        visual_for_fusion = visual_feat.detach()
        attr_for_fusion = attr_feat.detach()
    else:
        visual_for_fusion = visual_feat
        attr_for_fusion = attr_feat

    v = model.apn_proj(visual_for_fusion)
    a = model.apan_proj(attr_for_fusion)
    fused = torch.cat([(1.0 - alpha) * v, alpha * a], dim=1)

    char_vec = model.part_fc(fused)
    logits_global = model.classifier_global(char_vec)

    return {
        "logits_apn": logits_apn,
        "logits_apan": logits_apan,
        "logits_global": logits_global,
        "char_vec": char_vec,
        "fused": fused,
        "alpha": alpha,
    }


def train_full_model(
    model: GlobalReIDNetwork,
    train_loader: DataLoader,
    config: FullTrainConfig,
    query_loader: DataLoader | None = None,
    gallery_loader: DataLoader | None = None,
):
    device = torch.device(config.device)
    model.to(device)
    iqa_cache = _load_iqa_cache(config.iqa_cache_path) if config.use_iqa_cache else {}

    if config.load_pretrained_branches:
        logger.info("[FullTrain] Loading pretrained branches...")

        def load_branch_weights(
            state_dict: dict,
            backbone_module: nn.Module,
            head_module: nn.Module,
            head_prefix: str,
            backbone_prefix_ckpt: str = "backbone",
            strict: bool = False,
        ):
            """
            Loads weights into backbone and head modules from a checkpoint that might use prefixes.
            """
            # 1. Load Head
            head_keys = {}
            head_prefix_dot = f"{head_prefix}."
            
            # Special handling for MFPM in_proj which is dynamically created
            mfpm_proj_weights = {}

            for k, v in state_dict.items():
                if k.startswith(head_prefix_dot):
                    new_key = k[len(head_prefix_dot):]
                    head_keys[new_key] = v
                    
                    # Detect MFPM projection weights: mfpm.in_proj.0.weight
                    if "mfpm.in_proj" in new_key and "weight" in new_key:
                        try:
                            # Parse index: mfpm.in_proj.0.weight -> 0
                            parts = new_key.split(".")
                            idx = int(parts[2]) # mfpm, in_proj, <idx>, weight
                            mfpm_proj_weights[idx] = v
                        except (IndexError, ValueError):
                            pass

            # Pre-allocate MFPM projection layers if found
            if mfpm_proj_weights and hasattr(head_module, "mfpm") and hasattr(head_module.mfpm, "in_proj"):
                logger.info("Detected %d MFPM projection layers in checkpoint. Reconstructing...", len(mfpm_proj_weights))
                # Sort by index to append in order
                max_idx = max(mfpm_proj_weights.keys())
                current_len = len(head_module.mfpm.in_proj)
                
                # We need to ensure the list is long enough. 
                # Note: This assumes indices are contiguous starting from 0, or at least we fill up to max_idx.
                for i in range(max_idx + 1):
                    if i in mfpm_proj_weights:
                        w = mfpm_proj_weights[i]
                        # w shape: [out_channels, in_channels, k, k]
                        out_c, in_c, k, _ = w.shape
                        
                        if i < current_len:
                            layer = head_module.mfpm.in_proj[i]
                            if layer.in_channels != in_c or layer.out_channels != out_c:
                                logger.warning("Mismatch at MFPM index %d. Recreating layer.", i)
                                new_layer = nn.Conv2d(in_c, out_c, kernel_size=k, bias=False)
                                new_layer.to(w.device)
                                head_module.mfpm.in_proj[i] = new_layer
                        else:
                            logger.info("Restoring MFPM projection layer %d: Conv2d(%d, %d, k=%d)", i, in_c, out_c, k)
                            new_layer = nn.Conv2d(in_c, out_c, kernel_size=k, bias=False)
                            # Move to correct device/dtype (inferred from weight)
                            new_layer.to(device=w.device, dtype=w.dtype)
                            head_module.mfpm.in_proj.append(new_layer)

            if head_keys:
                logger.info("Loading %d keys into %s head module (strict=%s)...", len(head_keys), head_prefix, strict)
                missing, unexpected = head_module.load_state_dict(head_keys, strict=strict)
                if missing:
                    filtered_missing = [k for k in missing if "num_batches_tracked" not in k]
                    if filtered_missing:
                        logger.warning("Missing keys in %s: %s...", head_prefix, filtered_missing[:5])
                if unexpected:
                    logger.warning("Unexpected keys in %s: %s...", head_prefix, unexpected[:5])
            else:
                logger.warning("No keys found starting with '%s' for head module.", head_prefix_dot)

            # 2. Load Backbone
            bb_keys = {}
            bb_prefix_dot = f"{backbone_prefix_ckpt}."
            for k, v in state_dict.items():
                if k.startswith(bb_prefix_dot):
                    new_key = k[len(bb_prefix_dot):]
                    bb_keys[new_key] = v
            
            if bb_keys:
                logger.info("Loading %d keys into %s backbone (strict=%s)...", len(bb_keys), head_prefix, strict)
                missing, unexpected = backbone_module.load_state_dict(bb_keys, strict=strict)
                if missing:
                    filtered_missing = [k for k in missing if "num_batches_tracked" not in k]
                    if filtered_missing:
                        logger.warning("Missing keys in backbone: %s...", filtered_missing[:5])
                if unexpected:
                    logger.warning("Unexpected keys in backbone: %s...", unexpected[:5])
            else:
                logger.warning("No keys found starting with '%s' for backbone.", bb_prefix_dot)


        logger.info("Loading APN checkpoint: %s", config.apn_checkpoint_path)
        apn_ckpt = torch.load(config.apn_checkpoint_path, map_location=device)
        apn_state = apn_ckpt.get("model_state_dict", apn_ckpt)
        load_branch_weights(
            state_dict=apn_state,
            backbone_module=model.backbone_apn,
            head_module=model.apn,
            head_prefix="apn",
            backbone_prefix_ckpt="backbone",  # Checkpoint uses "backbone."
            strict=False, 
        )

        logger.info("Loading APAN checkpoint: %s", config.apan_checkpoint_path)
        apan_ckpt = torch.load(config.apan_checkpoint_path, map_location=device)
        apan_state = apan_ckpt.get("model_state_dict", apan_ckpt)
        load_branch_weights(
            state_dict=apan_state,
            backbone_module=model.backbone_apan,
            head_module=model.apan,
            head_prefix="apan",
            backbone_prefix_ckpt="backbone", 
            strict=False,
        )

        if config.freeze_epochs > 0:
            logger.info("Freezing branch parameters for %d epochs...", config.freeze_epochs)
            for p in model.backbone_apn.parameters():
                p.requires_grad = False
            for p in model.apn.parameters():
                p.requires_grad = False
            for p in model.backbone_apan.parameters():
                p.requires_grad = False
            for p in model.apan.parameters():
                p.requires_grad = False

    if not config.train_apn_backbone:
        for p in model.backbone_apn.parameters():
            p.requires_grad = False
    if not config.train_apan_backbone:
        for p in model.backbone_apan.parameters():
            p.requires_grad = False

    apn_params = list(model.backbone_apn.parameters()) + list(model.apn.parameters())
    apan_params = list(model.backbone_apan.parameters()) + list(model.apan.parameters())
    if model.shared_backbone:
        # Avoid overlapping shared backbone parameters across optimizers.
        shared_ids = {id(p) for p in model.backbone_apn.parameters()}
        apan_params = [p for p in apan_params if id(p) not in shared_ids]

    optim_apn = optim.Adam(
        apn_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    optim_apan = optim.Adam(
        apan_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    optim_global = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            list(model.part_fc.parameters())
            + list(model.classifier_global.parameters())
            + list(model.bn_neck.parameters()),
        ),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    optim_fusion = optim.Adam(
        filter(
            lambda p: p.requires_grad,
            list(model.alpha_head.parameters())
            + list(model.apn_proj.parameters())
            + list(model.apan_proj.parameters()),
        ),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Use Label Smoothing and Triplet Loss (Bag of Tricks for ReID)
    criterion_id = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    criterion_triplet = TripletLoss(margin=config.triplet_margin)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss()

    binary_keys = list(model.apan.binary_keys)
    multi_keys = dict(model.apan.multi_keys)
    best_mAP = 0.0
    best_epoch = 0
    
    # Warmup + cosine annealing scheduler
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        else:
            progress = (epoch - config.warmup_epochs) / max(1, config.epochs - config.warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)).item())
    
    scheduler_apn = optim.lr_scheduler.LambdaLR(optim_apn, lr_lambda)
    scheduler_apan = optim.lr_scheduler.LambdaLR(optim_apan, lr_lambda)
    scheduler_global = optim.lr_scheduler.LambdaLR(optim_global, lr_lambda)
    scheduler_fusion = optim.lr_scheduler.LambdaLR(optim_fusion, lr_lambda)
    for epoch in range(1, config.epochs + 1):
        if config.load_pretrained_branches and config.freeze_epochs > 0 and epoch == config.freeze_epochs + 1:
            for p in model.backbone_apn.parameters():
                p.requires_grad = True
            for p in model.apn.parameters():
                p.requires_grad = True
            for p in model.backbone_apan.parameters():
                p.requires_grad = True
            for p in model.apan.parameters():
                p.requires_grad = True
            for group in optim_apn.param_groups:
                group["lr"] = config.lr * config.branch_lr_scale
            for group in optim_apan.param_groups:
                group["lr"] = config.lr * config.branch_lr_scale

        model.train()
        running = {
            "loss_apn": 0.0,
            "loss_apan": 0.0,
            "loss_mix": 0.0,
            "loss_global": 0.0,
            "loss_triplet": 0.0,
            "acc_apn": 0.0,
            "acc_global": 0.0,
            "acc_apan_bin": 0.0,
            "acc_apan_multi": 0.0,
        }
        steps = 0

        for step, batch in enumerate(train_loader, start=1):
            images, labels, _, img_paths, attrs = _unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            if attrs:
                attrs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in attrs.items()}

            iqa_scores = _get_iqa_scores_from_cache(
                img_paths=img_paths,
                cache=iqa_cache,
                device=device,
                dtype=images.dtype,
            )

            outputs = _forward_full(
                model,
                images=images,
                iqa_score=iqa_scores,
                detach_branch_features_for_fusion=False,
            )
            
            logits_apn = outputs["logits_apn"]
            logits_apan = outputs["logits_apan"]
            logits_global = outputs["logits_global"]
            fused = outputs["fused"]
            if isinstance(logits_apn, (list, tuple)):
                apn_logits_list = list(logits_apn)
            else:
                apn_logits_list = [logits_apn]

            loss_apn = sum(criterion_id(logits, labels) for logits in apn_logits_list) / len(apn_logits_list)
            pred_apn = apn_logits_list[0].argmax(dim=1)
            acc_apn = (pred_apn == labels).float().mean().item()
            
            # Triplet loss on fused features (char_vec) for better metric learning
            char_vec = outputs["char_vec"]
            loss_triplet = criterion_triplet(char_vec, labels) * config.triplet_weight

            loss_apan, acc_apan_bin, acc_apan_multi = _compute_apan_loss(
                logits_dict=logits_apan,
                attrs=attrs,
                device=device,
                criterion_bce=criterion_bce,
                criterion_ce=criterion_ce,
                binary_keys=binary_keys,
                multi_keys=multi_keys,
            )

            # Retrieval-aligned alpha loss: use classifier on fused features,
            # but detach classifier weights so only fusion/alpha are updated here.
            char_vec_for_mix = model.part_fc(fused)
            char_vec_for_mix_bn = model.bn_neck(char_vec_for_mix)
            logits_mix = F.linear(
                char_vec_for_mix_bn,
                model.classifier_global.weight.detach(),
                None,  # No bias with BNNeck
            )
            loss_mix = criterion_id(logits_mix, labels)

            # Combined global loss: ID loss + triplet loss
            loss_global = criterion_id(logits_global, labels) + loss_triplet

            optim_apn.zero_grad(set_to_none=True)
            optim_apan.zero_grad(set_to_none=True)
            optim_global.zero_grad(set_to_none=True)
            optim_fusion.zero_grad(set_to_none=True)

            # 1) APN backward
            # retain_graph=True needed: loss_global and loss_mix also use visual_feat (no detach anymore)
            if loss_apn.requires_grad:
                loss_apn.backward(retain_graph=True)

            # 2) APAN backward
            # retain_graph=True needed: loss_global and loss_mix also use attr_feat (no detach anymore)
            if loss_apan is not None and loss_apan.requires_grad:
                loss_apan.backward(retain_graph=True)

            # 3) Global FC backward
            # retain_graph=True needed: loss_mix also backprops through the same fusion graph
            loss_global.backward(retain_graph=True)

            # 4) Alpha perceptron / fusion module backward (last one, no retain_graph needed)
            loss_mix.backward()

            optim_apn.step()
            if loss_apan is not None:
                optim_apan.step()
            optim_global.step()
            optim_fusion.step()

            steps += 1
            running["loss_apn"] += loss_apn.item()
            running["loss_mix"] += loss_mix.item()
            running["loss_global"] += loss_global.item()
            running["loss_triplet"] += loss_triplet.item()
            running["acc_apn"] += acc_apn
            running["acc_global"] += (logits_global.argmax(dim=1) == labels).float().mean().item()
            if loss_apan is not None:
                running["loss_apan"] += loss_apan.item()
                running["acc_apan_bin"] += acc_apan_bin
                running["acc_apan_multi"] += acc_apan_multi

            if config.log_interval and step % config.log_interval == 0:
                denom = max(1, steps)
                current_lr = optim_global.param_groups[0]['lr']
                logger.info(
                    "[Full][Epoch %d/%d][Step %d] lr=%.6f "
                    "loss_apn=%.4f loss_apan=%.4f loss_triplet=%.4f loss_global=%.4f "
                    "acc_apn=%.2f%% acc_global=%.2f%%",
                    epoch, config.epochs, step, current_lr,
                    running["loss_apn"] / denom, running["loss_apan"] / denom,
                    running["loss_triplet"] / denom, running["loss_global"] / denom,
                    running["acc_apn"] / denom * 100, running["acc_global"] / denom * 100,
                )

        # Step schedulers at end of epoch
        scheduler_apn.step()
        scheduler_apan.step()
        scheduler_global.step()
        scheduler_fusion.step()

        if (
            config.eval_interval
            and query_loader is not None
            and gallery_loader is not None
            and epoch % config.eval_interval == 0
        ):
            metrics = evaluate_reid(
                model=model,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                device=device,
                iqa_cache=iqa_cache,
                log_path=config.metrics_log_path,
                log_dir=config.metrics_log_dir,
            )
            current_mAP = metrics["mAP"]
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                best_epoch = epoch
                logger.info("New best mAP: %.2f%% at epoch %d", best_mAP * 100, epoch)
                _save_checkpoint(
                    model,
                    {
                        "apn": optim_apn,
                        "apan": optim_apan,
                        "global": optim_global,
                        "fusion": optim_fusion,
                    },
                    epoch,
                    config.save_dir,
                    "full_model_best.pth",
                )

        if config.save_every and epoch % config.save_every == 0:
            _save_checkpoint(
                model,
                {
                    "apn": optim_apn,
                    "apan": optim_apan,
                    "global": optim_global,
                    "fusion": optim_fusion,
                },
                epoch,
                config.save_dir,
                "full_model_epoch_latest.pth",
            )

    _save_checkpoint(
        model,
        {
            "apn": optim_apn,
            "apan": optim_apan,
            "global": optim_global,
            "fusion": optim_fusion,
        },
        config.epochs,
        config.save_dir,
        "full_model_final.pth",
    )
    
    logger.info("Training completed!")
    logger.info("Best mAP: %.2f%% at epoch %d", best_mAP * 100, best_epoch)
    logger.info("Best model saved: %s", os.path.join(config.save_dir, "full_model_best.pth"))
    logger.info("Final model saved: %s", os.path.join(config.save_dir, "full_model_final.pth"))
    
    return model
