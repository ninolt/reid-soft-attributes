"""
Global ReID Network: ResNet-50 backbones -> APN/APAN -> IQA alpha -> fusion -> classifier.
"""

import torch
import torch.nn as nn

from src.config import CFG
from src.iqa import get_iqa

from .APAN.APAN import APAN
from .APN.APN import APN
from .APN.resnet import ResNet50Backbone


class GlobalReIDNetwork(nn.Module):
    """
    Full architecture implementing:
    - APN (appearance) with BNNeck
    - APAN (attributes) with BNNeck
    - IQA score -> alpha in [0, 1]
    - Fusion + BNNeck + final classifier
    """

    def __init__(
        self,
        num_classes: int,
        num_parts: int = CFG["model"]["num_parts"],
        apn_feature_dim: int = CFG["apn"]["feature_dim"],
        apan_feature_dim: int = CFG["apan"]["feature_dim"],
        fusion_dim: int = CFG["model"]["fusion_dim"],
        apn_in_channels: int = CFG["apn"]["in_channels"],
        apan_in_channels: int = CFG["model"]["apan_in_channels"],
        pretrained_backbone: bool = True,
        shared_backbone: bool = False,
        iqa_mean: float = CFG["model"]["iqa_mean"],
        iqa_std: float = CFG["model"]["iqa_std"],
        dropout_rate: float = CFG["model"]["dropout_rate"],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.shared_backbone = shared_backbone
      
        self.backbone_apn = ResNet50Backbone(pretrained=pretrained_backbone)
        self.backbone_apan = ResNet50Backbone(pretrained=pretrained_backbone)

        self.apn = APN(
            in_channels=apn_in_channels, 
            num_classes=num_classes, 
            feature_dim=apn_feature_dim,
            dropout_rate=dropout_rate,
        )
        self.apan = APAN(
            in_channels=apan_in_channels, 
            feature_dim=apan_feature_dim,
            dropout_rate=dropout_rate,
        )

        # Learnable alpha from IQA score
        self.alpha_head = nn.Sequential(
            nn.Linear(1, CFG["model"]["alpha_head_hidden"]),
            nn.ReLU(inplace=True),
            nn.Linear(CFG["model"]["alpha_head_hidden"], 1),
            nn.Sigmoid(),
        )
        self.register_buffer("iqa_mean", torch.tensor(float(iqa_mean)))
        self.register_buffer("iqa_std", torch.tensor(float(iqa_std)))

        # Projection layers if dimensions don't match
        self.apn_proj = nn.Identity() if apn_feature_dim == fusion_dim else nn.Linear(apn_feature_dim, fusion_dim)
        self.apan_proj = nn.Identity() if apan_feature_dim == fusion_dim else nn.Linear(apan_feature_dim, fusion_dim)

        # Fusion layer: concat (visual, attr) -> reduced dim
        self.fusion_dim = fusion_dim
        self.part_fc = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
        )
        
        # BNNeck for final classification (Bag of Tricks)
        self.bn_neck = nn.BatchNorm1d(fusion_dim)
        self.bn_neck.bias.requires_grad_(False)
        
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.classifier_global = nn.Linear(fusion_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier_global.weight, std=CFG["model"]["classifier_weight_std"])

    def _prepare_iqa_score(
        self,
        iqa_score: torch.Tensor | list[float] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if iqa_score is None:
            return torch.full((batch_size, 1), CFG["model"]["default_iqa_alpha"], device=device, dtype=dtype)
        if not torch.is_tensor(iqa_score):
            iqa_score = torch.tensor(iqa_score, device=device, dtype=dtype)
        else:
            iqa_score = iqa_score.to(device=device, dtype=dtype)
        if iqa_score.dim() == 0:
            iqa_score = iqa_score.view(1)
        if iqa_score.dim() == 1:
            iqa_score = iqa_score.view(-1, 1)
        elif iqa_score.dim() > 2:
            iqa_score = iqa_score.view(iqa_score.size(0), -1)
            if iqa_score.size(1) != 1:
                iqa_score = iqa_score.mean(dim=1, keepdim=True)
        if iqa_score.size(0) != batch_size:
            if iqa_score.size(0) == 1:
                iqa_score = iqa_score.expand(batch_size, -1)
            else:
                raise ValueError(
                    f"iqa_score batch mismatch: got {iqa_score.size(0)} expected {batch_size}"
                )
        return (iqa_score - self.iqa_mean) / (self.iqa_std + CFG["model"]["iqa_std_epsilon"])

    def _compute_iqa_from_paths(self, img_paths: list[str]) -> list[float]:
        return [get_iqa(p) for p in img_paths]

    def forward(
        self,
        images: torch.Tensor | None = None,
        apn_x=None,
        apan_x=None,
        iqa_score: torch.Tensor | list[float] | None = None,
        iqa_scores: torch.Tensor | list[float] | None = None,
        img_paths: list[str] | None = None,
        is_train: bool = True,
    ):
        if iqa_score is None and iqa_scores is not None:
            iqa_score = iqa_scores

        if images is None and (apn_x is None or apan_x is None):
            raise ValueError("Provide `images` or both `apn_x` and `apan_x` for fusion.")

        if images is not None:
            feats_apn = self.backbone_apn(images)
            feats_apan = feats_apn if self.shared_backbone else self.backbone_apan(images)
            if apn_x is None:
                # Use layer3 only for APN (1024 channels)
                apn_x = feats_apn.get("layer3")
            if apan_x is None:
                apan_x = feats_apan.get("layer4")

        if apn_x is None or apan_x is None:
            raise ValueError("Both APN and APAN inputs are required for fusion.")

        logits_apn, visual_feat = self.apn(apn_x, is_train=is_train)
        logits_apan, attr_feat = self.apan(apan_x)

        if img_paths is not None and iqa_score is None:
            iqa_score = self._compute_iqa_from_paths(img_paths)

        iqa_score = self._prepare_iqa_score(
            iqa_score,
            batch_size=visual_feat.size(0),
            device=visual_feat.device,
            dtype=visual_feat.dtype,
        )
        alpha = self.alpha_head(iqa_score)

        v = self.apn_proj(visual_feat)
        a = self.apan_proj(attr_feat)
        fused = torch.cat([(1.0 - alpha) * v, alpha * a], dim=1)

        char_vec = self.part_fc(fused)
        
        # Apply BNNeck for classification
        char_vec_bn = self.bn_neck(char_vec)
        char_vec_bn = self.dropout(char_vec_bn)

        if is_train:
            logits_global = self.classifier_global(char_vec_bn)
            return logits_apn, logits_apan, logits_global, char_vec  # Return pre-BN features
        else:
            return None, None, None, char_vec  # Inference: return features for matching


class model(GlobalReIDNetwork):
    """Backward-compatible alias."""
    pass
