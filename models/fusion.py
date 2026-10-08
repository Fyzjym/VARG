import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from models.transformer import PositionalEncoding2D
from einops import rearrange, repeat
import math
from models.resnet_dilation import resnet18 as resnet18_dilation
from models.hcem import HCEM

from typing import Tuple, List
from torch.nn import functional as F


# ==============================================================================
# M 1: Core Transformer Blocks
# ==============================================================================

def precompute_freqs_cis(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """预计算旋转位置编码 (RoPE) 所需的频率参数。"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs_cis)


class SelfAttention_RoPE(nn.Module):
    """Self-Attention with RoPE"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def _apply_rope(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        freqs_cis = freqs_cis.view(1, x.shape[1], 1, -1, 2)
        x_out = torch.stack([
            x_[..., 0] * freqs_cis[..., 0] - x_[..., 1] * freqs_cis[..., 1],
            x_[..., 1] * freqs_cis[..., 0] + x_[..., 0] * freqs_cis[..., 1],
        ], -1)
        return x_out.flatten(-2).type_as(x)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.num_heads, self.head_dim) for t in (q, k, v)]
        q, k = self._apply_rope(q, freqs_cis), self._apply_rope(k, freqs_cis)
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]

        # ==============================================
        # scaled_dot_product_attention. PyTorch < 2.0

        # 1. scaled dot-product
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        # 2. Attention mask
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias

        # 3. softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 4. weighted sum of values
        out = attn_weights @ v
        # ============================================

        return self.proj(out.permute(0, 2, 1, 3).contiguous().view(B, L, C))


class FFN(nn.Module):
    """Feed-Forward Network"""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Transformer with AdaLN and RoPE"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, norm_eps: float = 1e-6):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(embed_dim, eps=norm_eps), SelfAttention_RoPE(embed_dim, num_heads)
        self.norm2, self.ffn = nn.LayerNorm(embed_dim, eps=norm_eps), FFN(embed_dim, mlp_ratio)
        self.ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 4 * embed_dim))

    def _modulate(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    def forward(self, x: torch.Tensor, style_vector: torch.Tensor, freqs_cis: torch.Tensor,
                attn_bias: torch.Tensor) -> torch.Tensor:
        mod_params = self.ada_lin(style_vector)
        scale1, shift1, scale2, shift2 = mod_params.chunk(4, dim=1)
        x = x + self.attn(self._modulate(self.norm1(x), scale1.unsqueeze(1), shift1.unsqueeze(1)), freqs_cis, attn_bias)
        x = x + self.ffn(self._modulate(self.norm2(x), scale2.unsqueeze(1), shift2.unsqueeze(1)))
        return x


# ==============================================================================
# M 2: condition module and  Patch module
# ==============================================================================

class PatchEmbed(nn.Module):
    """2D feature maps 2 1D Patch embedding by CONV."""

    def __init__(self, patch_size: int = 1, in_chans: int = 512, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, C_in, H, W] -> [B, C_embed, H_grid, W_grid]
        return x.flatten(2).transpose(1, 2)  # -> [B, N_patches, C_embed]


class StyleEncoder(nn.Module):
    """
    style feature maps 2 style vecotr
    AVG and MLP
    """

    def __init__(self, in_chans: int = 512, embed_dim: int = 768, hidden_dim_ratio: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = embed_dim * hidden_dim_ratio
        self.mlp = nn.Sequential(
            nn.Linear(in_chans, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)  # [B, C_in, H, W] -> [B, C_in, 1, 1]
        x = x.flatten(1)  # -> [B, C_in]
        return self.mlp(x)  # -> [B, C_embed]


# ==============================================================================
# SAT: Style-Aware Transformer
# ==============================================================================

class SAT(nn.Module):
    """Style-Aware Transformer with style-modulated blockwise causal attention.

    Five scales (1, 2, 4, 8, 16) construct 341 continuous content tokens.
    Each query can see its own scale and all coarser scales. The 256
    finest-scale tokens provide context for HCEM and the diffusion denoiser.
    """

    def __init__(self,
                 in_chans: int = 512,
                 depth: int = 6,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 output_dim: int = 512,
                 patch_nums: Tuple[int, ...] = (1, 2, 4, 8, 16)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_nums = patch_nums
        self.num_stages = len(patch_nums)

        # 1. F_s 2 s
        # [B, 512, 16, 16] -> [B, 768]
        self.style_encoder = StyleEncoder(in_chans=in_chans, embed_dim=embed_dim)

        # 2. Patch
        self.content_patch_embeds = nn.ModuleList([
            PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dim)
            for _ in patch_nums
        ])

        # 3. level
        self.lvl_embed = nn.Embedding(self.num_stages, embed_dim)

        # 4. Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0) for _ in range(depth)
        ])

        # 5. output
        self.norm_out = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def _prepare_multiscale_inputs(self, content_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        patches_by_scale = []
        for i, pn in enumerate(self.patch_nums):

            pooled_features = F.adaptive_avg_pool2d(content_features, (pn, pn))

            patches = self.content_patch_embeds[i](pooled_features)
            patches_by_scale.append(patches)

        full_sequence = torch.cat(patches_by_scale, dim=1)

        level_ids = torch.cat([
            torch.full((pn * pn,), i, dtype=torch.long, device=full_sequence.device)
            for i, pn in enumerate(self.patch_nums)
        ], dim=0)

        return full_sequence, level_ids

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        """

        Args:
            content_feat (torch.Tensor): Shape: [B, 512, 16, 16]
            style_feat (torch.Tensor): Shape: [B, 512, 16, 16]

        Returns:
            torch.Tensor: context。Shape: [B, 256, 512]
        """
        style_vector = self.style_encoder(style_feat)

        x, level_ids = self._prepare_multiscale_inputs(content_feat)

        x = x + self.lvl_embed(level_ids).unsqueeze(0)

        L_total = x.shape[1]
        freqs_cis = precompute_freqs_cis(self.embed_dim // self.num_heads, L_total).to(x.device)


        # Attention mask for VAR
        d = level_ids.view(1, L_total, 1)
        dT = level_ids.view(1, 1, L_total)
        attn_bias = torch.where(dT > d, float('-inf'), 0.0).unsqueeze(1)

        for block in self.blocks:
            x = block(x, style_vector=style_vector, freqs_cis=freqs_cis, attn_bias=attn_bias)

        # final scale 2 next module
        num_finest_patches = self.patch_nums[-1] ** 2
        finest_patches = x[:, -num_finest_patches:, :]

        finest_patches = self.norm_out(finest_patches)
        output = self.output_proj(finest_patches)

        return output

# ==============================================================================
# VARG conditioning: visual encoders -> SAT -> HCEM
# ==============================================================================


### merge the handwriting style and printed content
class VARGConditioner(nn.Module):
    """Construct diffusion context from printed content and reference handwriting.

    ``sat`` is the Style-Aware Transformer; ``hcem.heu`` and ``hcem.cam``
    expose the Hyperbolic Embedding Unit and Context Aggregation Module.
    The registered hierarchy and public names follow the manuscript.
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 normalize_before=True):
        super(VARGConditioner, self).__init__()
        

        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model) # add 2D position encoding
        self.style_projector = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))


        # Separate style and content encoders, with independent parameters.
        self.style_encoder = self.build_visual_encoder()
        self.style_dilation_layer = resnet18_dilation().conv5_x
        
        self.content_encoder = self.build_visual_encoder()
        self.content_dilation_layer = resnet18_dilation().conv5_x

        self.sat = SAT(
            in_chans=512,
            depth=6,
            embed_dim=768,
            num_heads=8,
            output_dim=512,
            patch_nums=(1, 2, 4, 8, 16))

        self.hcem = HCEM(
            in_chans=512,
            embed_dim=256,
            depth=4,
            output_dim=512
        )

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def build_visual_encoder(self,):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer4 = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()
        return resnet

    def encode_feature_map(self, encoder, dilation_layer, style, add_position2D):
        style = encoder(style)
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=16).contiguous()
        style = dilation_layer(style)
        style = add_position2D(style) # B, 512, 16, 16
        style_seq = rearrange(style, 'n c h w ->(h w) n c').contiguous() # 256, B, 512
        # style = style_encoder(style)
        return style_seq, style

    
    def encode_style(self, style):

        return self.encode_feature_map(self.style_encoder, self.style_dilation_layer, style, self.add_position2D)


    def encode_content(self, content):

        return self.encode_feature_map(self.content_encoder, self.content_dilation_layer, content, self.add_position2D)

    
    def forward(self, style, laplace=None, content=None, latex=None):
        """Return context [B, 256, 512] and style embeddings [B, 2, 256].

        ``style`` contains two views from the same writer. ``content`` is
        the rendered expression image. ``laplace`` and ``latex`` are reserved
        input slots retained for existing data adapters; neither enters SAT
        or HCEM in the current implementation.
        """


        # Two same-writer style views for the supervised contrastive objective.
        anchor_style = style[:, 0, :, :].clone().unsqueeze(1).contiguous()
        pos_style = style[:, 1, :, :].clone().unsqueeze(1).contiguous()

        # Project and normalize the style views.
        anchor_low = anchor_style
        anchor_low_feature, anchor_low_feature_patch = self.encode_style(anchor_low)
        anchor_low_nce = self.style_projector(anchor_low_feature) # t n c
        anchor_low_nce = torch.mean(anchor_low_nce, dim=0)

        pos_low = pos_style 
        pos_low_feature, pos_low_feature_patch = self.encode_style(pos_low)
        pos_low_nce = self.style_projector(pos_low_feature)
        pos_low_nce = torch.mean(pos_low_nce, dim=0)

        style_embeddings = torch.stack([anchor_low_nce, pos_low_nce], dim=1) # B 2 C
        style_embeddings = nn.functional.normalize(style_embeddings, p=2, dim=2)


        # content encoder
        if content.shape[1] == 1:
            anchor_content = content
        else:
            anchor_content = content[:, 0, :, :].unsqueeze(1).contiguous()

        content_feat, content_feat_patch = self.encode_content(anchor_content)

        # SAT context followed by HEU and CAM refinement in HCEM.
        style_hs = self.sat(content_feat_patch, anchor_low_feature_patch)
        style_hs = self.hcem(style_hs, content_feat_patch)

        return style_hs.contiguous(), style_embeddings # n t c # 32 256 512


    def generate(self, style, laplace=None, content=None, latex=None):
        """Return diffusion context from a single style reference and content image."""
        if style.shape[1] == 1:
            anchor_style = style
        else:
            anchor_style = style[:, 0, :, :].unsqueeze(1).contiguous()

        # Encode the reference handwriting style.
        anchor_low = anchor_style
        anchor_low_feature, anchor_low_feature_patch = self.encode_style(anchor_low)


        # content encoder
        if content.shape[1] == 1:
            anchor_content = content
        else:
            anchor_content = content[:, 0, :, :].unsqueeze(1).contiguous()
        content_feat, content_feat_patch = self.encode_content(anchor_content)

        # fusion of content and style features
        # SAT context followed by HEU and CAM refinement in HCEM.
        style_hs = self.sat(content_feat_patch, anchor_low_feature_patch)
        style_hs = self.hcem(style_hs, content_feat_patch)

        return style_hs.contiguous()
