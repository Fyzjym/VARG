import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Tuple, List, Dict, Any
from einops import repeat
import os


# ==============================================================================
# M 1: Hpy. Ge. tool 
# ==============================================================================

class HyperbolicMath:

    @staticmethod
    def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        v_norm = v.norm(dim=1, p=2, keepdim=True).clamp(min=1e-8)
        second_term = torch.tanh(v_norm) * v / v_norm
        return HyperbolicMath.mobius_add(x, second_term)

    @staticmethod
    def log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = HyperbolicMath.mobius_add(-x, y)
        diff_norm = diff.norm(dim=1, p=2, keepdim=True).clamp(min=1e-8)
        safe_diff_norm = diff_norm.clamp(max=1.0 - 1e-6)
        return (2.0 / 1.0) * torch.atanh(safe_diff_norm) * diff / diff_norm

    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, dim: int = 1) -> torch.Tensor:
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * xy + y2) * x + (1 - x2) * y
        den = 1 + 2 * xy + x2 * y2
        return num / den.clamp(min=1e-8)


# ==============================================================================
# M 2: Hyperbolic CNN
# ==============================================================================

class HyperbolicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()

        self.conv_euc = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x_hyp: torch.Tensor) -> torch.Tensor:

        x_hyp_transposed = x_hyp.permute(0, 2, 3, 1)  # -> [B, H, W, C]


        origin = torch.zeros_like(x_hyp_transposed)
        x_tan = HyperbolicMath.log_map(origin, x_hyp_transposed)  # -> [B, H, W, C]


        x_tan_permuted = x_tan.permute(0, 3, 1, 2)  # -> [B, C, H, W]
        convolved_tan = self.conv_euc(x_tan_permuted)  # -> [B, C_out, H', W']


        convolved_tan_permuted = convolved_tan.permute(0, 2, 3, 1)  # -> [B, H', W', C_out]
        origin_out = torch.zeros_like(convolved_tan_permuted)
        new_x_hyp_permuted = HyperbolicMath.exp_map(origin_out, convolved_tan_permuted)


        new_x_hyp = new_x_hyp_permuted.permute(0, 3, 1, 2)
        return new_x_hyp


class HyperbolicCNNEncoder(nn.Module):

    def __init__(self, in_chans: int = 512, embed_dim: int = 256, depth: int = 2, output_dim: int = 512):
        super().__init__()

        self.input_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            block = nn.Sequential(
                HyperbolicConv2d(embed_dim, embed_dim, kernel_size=3, padding=1),

                nn.ReLU()  #  log->relu->exp
            )
            self.blocks.append(block)

        self.output_proj = nn.Conv2d(embed_dim, output_dim, kernel_size=1)

    def forward(self, x_visual: torch.Tensor) -> torch.Tensor:
        # x_visual: [B, 512, 16, 16]

        x_euc = self.input_proj(x_visual)

        x_hyp = HyperbolicMath.exp_map(torch.zeros_like(x_euc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), x_euc)

        for block in self.blocks:
            x_hyp = block(x_hyp)

        x_tan = HyperbolicMath.log_map(torch.zeros_like(x_hyp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), x_hyp)

        output_feat_map = self.output_proj(x_tan)  # [B, 512, 16, 16]

        return output_feat_map.flatten(2).transpose(1, 2)  # [B, 256, 512]
