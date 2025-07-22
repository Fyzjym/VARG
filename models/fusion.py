import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from models.transformer import *
from einops import rearrange, repeat
import math
from models.resnet_dilation import resnet18 as resnet18_dilation
# from models.hblc_gnn import HGE
from models.hblc_cnn import HyperbolicCNNEncoder

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
# M 3: STA
# ==============================================================================

class SAT(nn.Module):
    """
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
# CAM
# ==============================================================================

class CAM(nn.Module):
    """
    CAM
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




### merge the handwriting style and printed content
class Mix_TR(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=1, num_decoder_layers=1,
                 dim_feedforward=2048, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 normalize_before=True):
        super(Mix_TR, self).__init__()
        

        self.add_position2D = PositionalEncoding2D(dropout=0.1, d_model=d_model) # add 2D position encoding
        self.low_pro_mlp = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))


        ### low frequency style encoder
        self.Feat_Encoder = self.initialize_resnet18()
        self.style_dilation_layer = resnet18_dilation().conv5_x
        
        self.content_encoder = self.initialize_resnet18()
        self.content_dilation_layer = resnet18_dilation().conv5_x

        self.var = SAT(
            in_chans=512,
            depth=6,
            embed_dim=768,
            num_heads=8,
            output_dim=512,
            patch_nums=(1, 2, 4, 8, 16))

        self.h_cnn_encoder = HyperbolicCNNEncoder(
            in_chans=512,
            embed_dim=256,
            depth=4,
            output_dim=512
        )
        self.cont_arg_m = CAM(embed_dim=512)

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def initialize_resnet18(self,):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.layer4 = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()
        return resnet

    def process_style_feature(self, encoder, dilation_layer, style, add_position2D):
        style = encoder(style)
        style = rearrange(style, 'n (c h w) ->n c h w', c=256, h=16).contiguous()
        style = dilation_layer(style)
        style = add_position2D(style) # B, 512, 16, 16
        style_seq = rearrange(style, 'n c h w ->(h w) n c').contiguous() # 256, B, 512
        # style = style_encoder(style)
        return style_seq, style

    
    def get_low_style_feature(self, style):

        return self.process_style_feature(self.Feat_Encoder, self.style_dilation_layer, style, self.add_position2D)



    def get_content_style_feature(self, content):

        return self.process_style_feature(self.content_encoder, self.content_dilation_layer, content, self.add_position2D)

    
    def forward(self, style, laplace, content, latex):
        """

        :param style:
        :param laplace:
        :param content:
        :param latex: list, including B raw strings, like ['a _ { 1 } + a _ { 2 }', '\\cos ^ { 3 } y = \\frac { 1 } { 4 } ( \\cos 3 y + 3 \\cos y )']
        :return:
        """


        # get the high frequency and style feature
        anchor_style = style[:, 0, :, :].clone().unsqueeze(1).contiguous()
        pos_style = style[:, 1, :, :].clone().unsqueeze(1).contiguous()

        # get the low frequency and style feature
        anchor_low = anchor_style
        anchor_low_feature, anchor_low_feature_patch = self.get_low_style_feature(anchor_low)
        anchor_low_nce = self.low_pro_mlp(anchor_low_feature) # t n c
        anchor_low_nce = torch.mean(anchor_low_nce, dim=0)

        pos_low = pos_style 
        pos_low_feature, pos_low_feature_patch = self.get_low_style_feature(pos_low)
        pos_low_nce = self.low_pro_mlp(pos_low_feature)
        pos_low_nce = torch.mean(pos_low_nce, dim=0)

        low_nce_emb = torch.stack([anchor_low_nce, pos_low_nce], dim=1) # B 2 C
        low_nce_emb = nn.functional.normalize(low_nce_emb, p=2, dim=2)


        # content encoder
        if content.shape[1] == 1:
            anchor_content = content
        else:
            anchor_content = content[:, 0, :, :].unsqueeze(1).contiguous()

        content_feat, content_feat_patch = self.get_content_style_feature(anchor_content)

        # SFRD
        # style_hs = self.decoder(content_feat, anchor_low_feature, tgt_mask=None)
        # VAR
        style_hs = self.var(content_feat_patch, anchor_low_feature_patch)
        structural_features = self.h_cnn_encoder(content_feat_patch)
        style_hs = self.cont_arg_m(style_hs, structural_features)

        return style_hs.contiguous(), low_nce_emb # n t c # 32 256 512



    def generate(self, style, laplace, content, latex):
        if style.shape[1] == 1:
            anchor_style = style
            # anchor_high = laplace
        else:
            anchor_style = style[:, 0, :, :].unsqueeze(1).contiguous()
            # anchor_high = laplace[:, 0, :, :].unsqueeze(1).contiguous()

        # get the highg frequency and style feature
        # anchor_high_feature = self.get_high_style_feature(anchor_high) # t n c
        # get the low frequency and style feature
        anchor_low = anchor_style
        # anchor_low_feature, = self.get_low_style_feature(anchor_low)
        anchor_low_feature, anchor_low_feature_patch = self.get_low_style_feature(anchor_low)

        # anchor_mask = self.low_feature_filter(anchor_low_feature)
        # anchor_low_feature = anchor_low_feature * anchor_mask

        # content encoder
        if content.shape[1] == 1:
            anchor_content = content
        else:
            anchor_content = content[:, 0, :, :].unsqueeze(1).contiguous()
        # content_feat = self.get_content_style_feature(anchor_content)
        content_feat, content_feat_patch = self.get_content_style_feature(anchor_content)

        # fusion of content and style features
        # SFRD
        # style_hs = self.decoder(content_feat, anchor_low_feature, tgt_mask=None)
        # hs = self.fre_decoder(style_hs[0], anchor_high_feature, tgt_mask=None)
        # VAR
        style_hs = self.var(content_feat_patch, anchor_low_feature_patch)
        structural_features = self.h_cnn_encoder(content_feat_patch)
        style_hs = self.cont_arg_m(style_hs, structural_features)

        # return hs[0].permute(1, 0, 2).contiguous()
        # return style_hs[0].permute(1, 0, 2).contiguous()
        return style_hs.contiguous()