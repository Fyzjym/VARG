"""HCEM: Hierarchical Content Enhancement Module."""

import torch
import torch.nn as nn
from models.heu import HEU


class CAM(nn.Module):
    """Context Aggregation Module: gate HEU features using SAT features.

    This module preserves the supplied gated residual computation:
    ``sat_features + sigmoid(MLP(sat_features)) * heu_features``.
    """
    def __init__(self, embed_dim: int, hidden_dim_ratio: int = 2):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // hidden_dim_ratio),
            nn.GELU(),
            nn.Linear(embed_dim // hidden_dim_ratio, embed_dim),
            nn.Sigmoid()
        )
        with torch.no_grad():
            self.gate_network[-2].weight.zero_()
            self.gate_network[-2].bias.fill_(-1)

    def forward(self, visual_features: torch.Tensor, structural_features: torch.Tensor) -> torch.Tensor:
        gate = self.gate_network(visual_features)
        fused_features = visual_features + gate * structural_features
        return fused_features



class HCEM(nn.Module):
    """Hierarchical Content Enhancement Module: HEU plus CAM.

    ``forward(sat_features, content_features)`` returns the contextual
    condition [B, H*W, output_dim] for diffusion. Grouping the existing
    modules adds no parameters or numerical operations.
    """

    def __init__(self, in_chans=512, embed_dim=256, depth=4, output_dim=512):
        super().__init__()
        self.heu = HEU(in_chans=in_chans, embed_dim=embed_dim,
                       depth=depth, output_dim=output_dim)
        self.cam = CAM(embed_dim=output_dim)

    def forward(self, sat_features, content_features):
        structural_features = self.heu(content_features)
        return self.cam(sat_features, structural_features)
