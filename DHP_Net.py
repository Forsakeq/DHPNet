# DHP-Net.py
# ------------------------------------------------------------
# Your DHP-Net (MHFF-based) with:
#   - MIRF: MIRFFusionStage (main branch)
#   - DINF: DINFFusionStage (aux branch)
#   - MHFF (MHFFFusionStage): omega gate with **learnable omega_scale**
#       omega = sigmoid(omega_logits) * sigmoid(omega_scale_logit)
#       -> omega in (0, 1), and omega_scale is trained (no fixed omega_max clamp)
#   - ERA head: MS learnable Gaussian aggregation (MS-EGA-like)
#
# NOTE:
#   This file keeps the network functionality the same EXCEPT the omega_max clamp:
#   - previously: omega in [0, omega_max] (omega_max fixed float)
#   - now:        omega in (0, 1) with a learnable scale parameter
#
# ------------------------------------------------------------

import os
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from swin.transformer import SwinTransformerBackbone

try:
    from timm.layers import trunc_normal_
except Exception:
    from timm.models.layers import trunc_normal_


# ============================================================
# Basic Utils: Patch <-> Image
# ============================================================
def PatchToImage(feature: torch.Tensor) -> torch.Tensor:
    # (B, L, C) -> (B, C, H, W), assume square
    assert feature.dim() == 3, f"PatchToImage expects (B,L,C), got {feature.shape}"
    b, l, c = feature.shape
    h = int(round(math.sqrt(l)))
    assert h * h == l, f"PatchToImage expects L to be square, got L={l}"
    return feature.permute(0, 2, 1).contiguous().view(b, c, h, h)


def ImageToPatch(feature: torch.Tensor) -> torch.Tensor:
    # (B, C, H, W) -> (B, L, C)
    assert feature.dim() == 4, f"ImageToPatch expects (B,C,H,W), got {feature.shape}"
    return feature.flatten(-2).permute(0, 2, 1).contiguous()


# ============================================================
# Decoder UpSampling (unchanged)
# ============================================================
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand = nn.Linear(in_dim, 4 * out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):  # (B, H*W, C)
        H, W = self.input_resolution
        B, L, _ = x.shape
        assert L == H * W, f"PatchExpand: L mismatch {L} vs {H}*{W}"
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, (dim_scale ** 2) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):  # (B, H*W, C)
        H, W = self.input_resolution
        B, L, _ = x.shape
        assert L == H * W, f"FinalPatchExpand_X4: L mismatch {L} vs {H}*{W}"
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(
            x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
            p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2)
        )
        x = x.view(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)
        x = self.norm(x)
        x = rearrange(x, 'b h w c-> b c h w').contiguous()
        return x


# ============================================================
# Head Modules (unchanged)
# ============================================================
class ScoreModule(nn.Module):
    def __init__(self, channels, image_size=None):
        super().__init__()
        d = 1
        self.extra_model = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=d),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=d),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=d),
            nn.ReLU()
        )
        self.conv_1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1)
        self.image_size = image_size

    def forward(self, x):
        x = self.extra_model(x)
        x = self.conv_1(x)
        if self.image_size is not None:
            x = F.interpolate(x, size=self.image_size, mode='bilinear', align_corners=True)
        return x


class Conv3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        d = 1
        self.extra_model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=d),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, padding=d),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, padding=d)
        )

    def forward(self, x):
        return self.extra_model(x)


# ============================================================
# Edge Module: edge_prior (morph grad) + MS Learnable Gaussian Aggregation (EGA-like)
# ============================================================
class MorphologicalGradient(nn.Module):
    """Differentiable-ish morphological gradient using max/min pooling."""
    def __init__(self, k=3):
        super().__init__()
        assert k % 2 == 1, "k must be odd"
        self.k = k
        self.pad = k // 2

    def forward(self, x):  # x: (B,1,H,W) in [0,1]
        dil = F.max_pool2d(x, kernel_size=self.k, stride=1, padding=self.pad)
        ero = -F.max_pool2d(-x, kernel_size=self.k, stride=1, padding=self.pad)
        grad = (dil - ero).clamp(min=0.0, max=1.0)
        return grad


class LearnableGaussian2D(nn.Module):
    """Learnable isotropic Gaussian filter (single-channel) with fixed kernel size."""
    def __init__(self, kernel_size=5, init_sigma=2.0, eps=1e-6):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.ks = kernel_size
        self.eps = eps
        self.log_sigma = nn.Parameter(torch.tensor(float(init_sigma)).log())

        # precompute grid
        r = kernel_size // 2
        ys, xs = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing='ij')
        self.register_buffer('xs', xs.float())
        self.register_buffer('ys', ys.float())

    def forward(self, x):  # x: (B,1,H,W)
        sigma = self.log_sigma.exp().clamp(min=self.eps)
        g = torch.exp(-(self.xs ** 2 + self.ys ** 2) / (2.0 * sigma ** 2))
        g = g / (g.sum() + self.eps)
        kernel = g.view(1, 1, self.ks, self.ks)
        return F.conv2d(x, kernel, padding=self.ks // 2)


class MSLearnableGaussianAgg(nn.Module):
    """Multi-scale learnable Gaussian aggregation with learnable scale weights."""
    def __init__(self, kernel_size=5, init_sigmas=(0.8, 1.2, 2.0)):
        super().__init__()
        self.gausses = nn.ModuleList([LearnableGaussian2D(kernel_size, s) for s in init_sigmas])

        # alpha for [identity(edge0), g1(edge0), g2(edge0), ...]
        self.alpha = nn.Parameter(torch.zeros(1 + len(init_sigmas)))
        with torch.no_grad():
            self.alpha[0].fill_(1.0)  # slightly prefer identity at start

    def forward(self, edge0):  # (B,1,H,W)
        outs = [edge0]
        for g in self.gausses:
            outs.append(g(edge0))
        w = torch.softmax(self.alpha, dim=0)  # (K,)
        y = 0
        for wi, oi in zip(w, outs):
            y = y + wi * oi
        return y


class ERA_MS_EGA(nn.Module):
    """
    Scheme2+3 combo:
      1) edge0(edge_prior) from coarse via morphological gradient (k=morph_ks)
      2) MS learnable Gaussian aggregation -> edge_ms
      3) inject edge_ms into edge_feature (C channels) and predict contour
    """
    def __init__(self, channels, gauss_ks=5, init_sigmas=(0.8, 1.2, 2.0), morph_ks=3,
                 mask_smooth_ks=5, mask_dilate_ks=7):
        super().__init__()
        self.channels = channels
        self.morph = MorphologicalGradient(k=morph_ks)
        self.ms = MSLearnableGaussianAgg(kernel_size=gauss_ks, init_sigmas=init_sigmas)

        # coarse mask: smooth + dilate (soft mask) to reduce speckle noise
        self.mask_smooth_ks = int(mask_smooth_ks)
        self.mask_dilate_ks = int(mask_dilate_ks)

        self.extract_context = Conv3(3, channels)
        self.fuse_context_region = Conv3(2 * channels, channels)

        # map 1ch edge_ms -> C and fuse
        self.edge_to_c = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # contour predictor
        self.contour = ScoreModule(channels)


    def _soft_coarse_mask(self, coarse_prob: torch.Tensor) -> torch.Tensor:
        """Smooth + dilate a probability mask (B,1,H,W) -> (B,1,H,W)."""
        x = coarse_prob
        if self.mask_smooth_ks and self.mask_smooth_ks > 1:
            k = self.mask_smooth_ks
            x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        if self.mask_dilate_ks and self.mask_dilate_ks > 1:
            k = self.mask_dilate_ks
            x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        return x.clamp(0.0, 1.0)

    def forward(self, fused_fea_img, rgb, coarse_logits):
        coarse_prob = torch.sigmoid(coarse_logits)
        coarse_mask = self._soft_coarse_mask(coarse_prob)

        edge0 = self.morph(coarse_mask)      # (B,1,H,W)
        edge_ms = self.ms(edge0)             # (B,1,H,W)


        context_features = self.extract_context(rgb)
        edge_feature = self.fuse_context_region(
            torch.cat((fused_fea_img, context_features * coarse_mask), dim=1)
        )

        edge_feature = edge_feature + self.edge_to_c(edge_ms)

        contour_logits = self.contour(edge_feature)
        return edge_feature, contour_logits




# ============================================================
# Init Helpers
# ============================================================
def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=(2, 3)).sum(dim=0), grad_output.sum(dim=(2, 3)).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


# ============================================================
# Depth Branch: MDFE (UNCHANGED LOGIC)
# ============================================================
class MDFEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c3 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.c5 = nn.Conv2d(in_ch, out_ch, 5, padding=2, bias=False)
        self.c7 = nn.Conv2d(in_ch, out_ch, 7, padding=3, bias=False)

        self.fuse = nn.Conv2d(out_ch * 5, out_ch, 1, 1, 0, bias=True)
        self.norm = LayerNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        initialize_weights([self.c3, self.c5, self.c7, self.fuse], 0.1)

    def forward(self, x):
        f3 = self.c3(x)
        f5 = self.c5(x)
        f7 = self.c7(x)
        d35 = f3 - f5
        d57 = f5 - f7
        y = torch.cat([f3, f5, f7, d35, d57], dim=1)
        y = self.fuse(y)
        y = self.norm(y)
        y = self.act(y)
        return y


class MDFEEncoder(nn.Module):
    def __init__(self, in_ch=1, embed_dim=96):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, embed_dim, kernel_size=4, stride=4, padding=0)

        self.s1 = MDFEBlock(embed_dim, embed_dim)
        self.down1 = nn.Conv2d(embed_dim, embed_dim * 2, 2, 2, 0)

        self.s2 = MDFEBlock(embed_dim * 2, embed_dim * 2)
        self.down2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, 2, 2, 0)

        self.s3 = MDFEBlock(embed_dim * 4, embed_dim * 4)
        self.down3 = nn.Conv2d(embed_dim * 4, embed_dim * 8, 2, 2, 0)

        self.s4 = MDFEBlock(embed_dim * 8, embed_dim * 8)

        initialize_weights([self.stem, self.down1, self.down2, self.down3], 0.1)

    def forward(self, x):
        x = self.stem(x)  # (B,96,56,56)
        s1 = self.s1(x)

        x = self.down1(s1)  # (B,192,28,28)
        s2 = self.s2(x)

        x = self.down2(s2)  # (B,384,14,14)
        s3 = self.s3(x)

        x = self.down3(s3)  # (B,768,7,7)
        s4 = self.s4(x)

        outs = [s1, s2, s3, s4]
        seqs = [ImageToPatch(o) for o in outs]
        return seqs


# ============================================================
# MIRF: main fusion branch
# ============================================================
class MIRFCore(nn.Module):
    def __init__(self, nf=96):
        super().__init__()
        self.conv1 = ResidualBlock_noBN(nf)  # for x_p
        self.conv2 = ResidualBlock_noBN(nf)  # for x_n

        self.convf1 = nn.Conv2d(nf * 3, nf, 1, 1, 0)
        self.convf2 = nn.Conv2d(nf * 3, nf, 1, 1, 0)

        self.norm_s = LayerNorm2d(nf)
        self.clustering = nn.Conv2d(nf, nf, 1, 1, 0)
        self.unclustering = nn.Conv2d(nf * 2, nf, 1, 1, 0)

        # spatial attention
        self.q_s = nn.Conv2d(nf, nf, 1, 1, 0)
        self.k_s = nn.Conv2d(nf * 2, nf, 1, 1, 0)
        self.v_s = nn.Conv2d(nf * 2, nf, 1, 1, 0)

        # channel attention
        self.k_nd = nn.Conv2d(nf * 2, nf, 1, 1, 0)
        self.v_nd = nn.Conv2d(nf * 2, nf, 1, 1, 0)

        self.gated1 = nn.Sequential(nn.Conv2d(nf, nf, 1, 1, 0), nn.Sigmoid())
        self.gated2 = nn.Sequential(nn.Conv2d(nf, nf, 1, 1, 0), nn.Sigmoid())

        self.scale_s = nf ** -0.5

        initialize_weights([self.convf1, self.convf2, self.clustering, self.unclustering,
                            self.q_s, self.k_s, self.v_s, self.k_nd, self.v_nd], 0.1)

    def forward(self, x_int, x_n, x_p, x_d):
        b, c, h, w = x_int.shape

        x_p_ = self.conv1(x_p)
        x_n_ = self.conv2(x_n)

        center_p = self.clustering(self.norm_s(self.convf1(torch.cat([x_int, x_p, x_d], dim=1))))
        center_n = self.clustering(self.norm_s(self.convf2(torch.cat([x_int, x_n, x_d], dim=1))))

        # ---- spatial attention ----
        q_left = self.q_s(center_p).contiguous().view(b, c, -1).permute(0, 2, 1)  # (B,HW,C)
        k_feat = self.k_s(torch.cat([x_n, x_d], dim=1))
        v_feat = self.v_s(torch.cat([x_n, x_d], dim=1))
        k_left = k_feat.contiguous().view(b, c, -1)                               # (B,C,HW)
        v_left = v_feat.contiguous().view(b, c, -1).permute(0, 2, 1)              # (B,HW,C)

        att_left = torch.bmm(q_left, k_left) * self.scale_s                       # (B,HW,HW)
        z_spatial = torch.bmm(torch.softmax(att_left, dim=-1), v_left)            # (B,HW,C)
        z_spatial = z_spatial.permute(0, 2, 1).contiguous().view(b, c, h, w)

        # ---- channel attention ----
        shared_xn = center_n.contiguous().view(b, c, -1)                           # (B,C,HW)
        k_right = self.k_nd(torch.cat([x_n, x_d], dim=1)).contiguous().view(b, c, -1)
        v_right = self.v_nd(torch.cat([x_n, x_d], dim=1)).contiguous().view(b, c, -1)

        scale_ch = (h * w) ** -0.5
        att_ch = torch.bmm(shared_xn, k_right.permute(0, 2, 1)) * scale_ch         # (B,C,C)
        z_nd = torch.bmm(torch.softmax(att_ch, dim=-1), v_right).contiguous().view(b, c, h, w)

        W_p = self.gated1(z_spatial + x_n_)
        Y_p = W_p * z_spatial + (1.0 - W_p) * x_n_

        W_n = self.gated2(z_nd + x_p_)
        Y_n = W_n * z_nd + (1.0 - W_n) * x_p_

        Y_int = self.unclustering(torch.cat([center_p, center_n], dim=1)) + x_int
        return Y_p, Y_n, Y_int


class MIRFBranchAggregator(nn.Module):
    def __init__(self, C, hidden_ratio=0.25, temperature=3.0,
                 use_pixel_gate=False, use_channel_gate=True,
                 detach_for_gate=True, eps=1e-6,
                 logit_scale=0.5, mix=0.6, use_input_norm=True):
        super().__init__()
        self.tau = temperature
        self.use_pixel_gate = use_pixel_gate
        self.use_channel_gate = use_channel_gate
        self.detach_for_gate = detach_for_gate
        self.eps = eps
        self.logit_scale = logit_scale
        self.mix = mix

        hidden = max(1, int(C * hidden_ratio))
        inC = 3 * C

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            self.norm_gate = nn.GroupNorm(num_groups=1, num_channels=inC)

        if use_pixel_gate:
            self.pixel_gate = nn.Sequential(
                nn.Conv2d(inC, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 3, 1, bias=True)
            )
            nn.init.constant_(self.pixel_gate[-1].bias, 0.0)
            with torch.no_grad():
                self.pixel_gate[-1].bias[2].add_(0.2)

        if use_channel_gate:
            self.avg = nn.AdaptiveAvgPool2d(1)
            self.chan_gate = nn.Sequential(
                nn.Conv2d(inC, hidden, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 3 * C, 1, bias=True)
            )
            nn.init.constant_(self.chan_gate[-1].bias, 0.0)
            with torch.no_grad():
                self.chan_gate[-1].bias[2 * C:].add_(0.2)

        self.last_aux = None

    def forward(self, Y_p, Y_n, Y_int, return_weights=False):
        B, C, H, W = Y_p.shape
        feats = torch.cat([Y_p, Y_n, Y_int], dim=1)
        gate_in = feats.detach() if self.detach_for_gate else feats
        if self.use_input_norm:
            gate_in = self.norm_gate(gate_in)

        logits = 0
        aux = {}

        if self.use_pixel_gate:
            lp = self.pixel_gate(gate_in)  # (B,3,H,W)
            logits = logits + lp.unsqueeze(2)  # (B,3,1,H,W)
            aux['pixel_logits'] = lp

        if self.use_channel_gate:
            gc = self.chan_gate(self.avg(gate_in))  # (B,3C,1,1)
            lc = gc.view(B, 3, C, 1, 1)
            logits = logits + lc
            aux['channel_logits'] = lc

        if isinstance(logits, int):
            raise RuntimeError("No gate enabled in MIRFBranchAggregator")

        logits = logits * self.logit_scale
        w = torch.softmax(logits / self.tau, dim=1).clamp_min(self.eps)

        fused_std = (w[:, 0] * Y_p) + (w[:, 1] * Y_n) + (w[:, 2] * Y_int)
        Y = Y_int + self.mix * (fused_std - Y_int)

        with torch.no_grad():
            ent = -(w * (w + 1e-12).log()).sum(dim=1).mean()
        aux['entropy'] = ent
        aux['weights'] = w

        self.last_aux = aux
        return (Y, aux) if return_weights else Y

    def set_temperature(self, tau: float):
        self.tau = float(tau)

    def set_mix(self, mix: float):
        self.mix = float(mix)

    def enable_pixel_gate(self, enabled: bool = True):
        if enabled and not hasattr(self, 'pixel_gate'):
            raise RuntimeError("pixel_gate not constructed; create with use_pixel_gate=True")
        self.use_pixel_gate = bool(enabled)

    def get_entropy(self):
        if self.last_aux is None:
            return None
        return self.last_aux.get('entropy', None)


class MIRFFusionStage(nn.Module):
    """
    stage-level MIRF fusion:
      input tokens: rgb_seq (B,N,C), fs_seq (B*S,N,C), depth_seq (B,N,C)
      output tokens: fused_seq (B,N,C)
    """
    def __init__(self, dim, fea_reso, num_slices=12,
                 gate_temperature=3.0, gate_mix=0.6, gate_logit_scale=0.5):
        super().__init__()
        self.dim = dim
        self.fea_reso = fea_reso
        self.S = int(num_slices)

        self.biem = MIRFCore(nf=dim)

        # focal-stack slice pooling (token-wise)
        self.fs_reduce_n = nn.Linear(dim, dim)
        self.fs_reduce_p = nn.Linear(dim, dim)
        self.slice_score = nn.Linear(dim, 1)

        self.branch_fuse = MIRFBranchAggregator(
            C=dim,
            temperature=gate_temperature,
            use_pixel_gate=False,
            use_channel_gate=True,
            logit_scale=gate_logit_scale,
            mix=gate_mix,
            detach_for_gate=True,
            use_input_norm=True
        )

    def _aggregate_fs(self, fs_seq: torch.Tensor, B: int, N: int, C: int) -> torch.Tensor:
        # fs_seq: (B*S,N,C) or (B,N,C)
        assert fs_seq.dim() == 3, f"fs_seq must be 3D, got {fs_seq.shape}"
        BS = fs_seq.shape[0]
        if BS == B:
            return fs_seq
        assert BS % B == 0, f"fs_seq first dim {BS} not divisible by B={B}"
        S = BS // B
        fs_view = fs_seq.view(B, S, N, C)
        scores = self.slice_score(fs_view)        # (B,S,N,1)
        alpha = torch.softmax(scores, dim=1)      # over slices
        fs_agg = (alpha * fs_view).sum(dim=1)     # (B,N,C)
        return fs_agg

    def forward(self, rgb_seq, fs_seq, depth_seq, return_aux=False):
        B, N, C = rgb_seq.shape
        H = W = self.fea_reso
        assert N == H * W, f"[MIRFFusionStage] N mismatch: {N} vs {H}*{W}"
        assert depth_seq.shape == (B, N, C), f"[MIRFFusionStage] depth_seq mismatch: {depth_seq.shape}"

        Fa_img = PatchToImage(rgb_seq)
        Fd_img = PatchToImage(depth_seq)

        fs_agg = self._aggregate_fs(fs_seq, B=B, N=N, C=C)  # (B,N,C)

        x_n_seq = self.fs_reduce_n(fs_agg)
        x_p_seq = self.fs_reduce_p(fs_agg)

        x_n_img = PatchToImage(x_n_seq)
        x_p_img = PatchToImage(x_p_seq)

        Y_p, Y_n, Y_int = self.biem(Fa_img, x_n_img, x_p_img, Fd_img)
        fused_img, aux = self.branch_fuse(Y_p, Y_n, Y_int, return_weights=True)

        fused_seq = ImageToPatch(fused_img)
        return (fused_seq, aux) if return_aux else fused_seq


# ============================================================
# DINF: auxiliary fusion branch (2-mod)
# ============================================================
class GatedDWFFN(nn.Module):
    def __init__(self, dim, expansion=2.0, bias=True):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in  = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv      = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class ChannelCrossAttention2d(nn.Module):
    def __init__(self, dim, norm_layer=None):
        super().__init__()
        self.scale = dim ** -0.5
        self.conv_fuse = nn.Conv2d(dim * 2, dim, 1, bias=True)
        self.norm = norm_layer(dim) if norm_layer is not None else nn.Identity()
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, query_base, src):
        assert query_base.shape == src.shape, f"ChannelCrossAttention2d mismatch: {query_base.shape} vs {src.shape}"
        B, C, H, W = query_base.shape

        q = self.q_proj(self.norm(self.conv_fuse(torch.cat([query_base, src], dim=1))))
        q = q.view(B, C, -1)  # (B,C,HW)

        k = self.k_proj(src).view(B, C, -1).permute(0, 2, 1).contiguous()  # (B,HW,C)
        v = self.v_proj(src).view(B, C, -1)                                # (B,C,HW)

        att = torch.bmm(q, k) * self.scale  # (B,C,C)
        att = torch.softmax(att, dim=-1)

        out = torch.bmm(att, v).view(B, C, H, W)
        return out


class DepthGuidedSliceDistribution(nn.Module):
    """
    depth-guided per-pixel slice selection:
      alpha(B,S,1,H,W) ; x_focal = sum alpha * fs_slice
      u_f = normalized entropy(alpha) in [0,1]
    NOTE:
      lambda_prior is stored as RAW parameter, and used as softplus(lambda_prior) >= 0
      Initialize raw to negative so effective prior strength starts near 0.
    """
    def __init__(self, dim, num_slices=12, hidden_ratio=0.5,
                 init_tau=1.5, init_sigma=1.2, learn_sigma=True,
                 init_lambda_prior_raw=-6.0):
        super().__init__()
        self.S = int(num_slices)
        hidden = max(8, int(dim * hidden_ratio))

        self.logit_net = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True)
        )
        self.depth_to_d = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        if learn_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma)))
        else:
            self.register_buffer("log_sigma", torch.tensor(math.log(init_sigma)))

        # RAW parameter (can be negative); effective = softplus(raw) >= 0
        self.lambda_prior = nn.Parameter(torch.tensor(float(init_lambda_prior_raw)))

    def forward(self, fs_slices, x_rgb, x_depth):
        assert fs_slices.dim() == 5, f"fs_slices must be 5D, got {fs_slices.shape}"
        if fs_slices.shape[0] == self.S and fs_slices.shape[1] != self.S:
            fs_slices = fs_slices.permute(1, 0, 2, 3, 4).contiguous()

        B, S, C, H, W = fs_slices.shape
        assert S == self.S, f"Expected S={self.S}, got {S}"
        assert x_rgb.shape == (B, C, H, W) and x_depth.shape == (B, C, H, W), \
            f"x_rgb/x_depth must match fs feature shape, got {x_rgb.shape}/{x_depth.shape}"

        # depth prior: mu=(S-1)*d
        d = self.depth_to_d(x_depth)          # (B,1,H,W) in [0,1]
        mu = (S - 1) * d                      # (B,1,H,W)
        sigma = F.softplus(self.log_sigma) + 1e-6

        s_idx = torch.arange(S, device=fs_slices.device, dtype=fs_slices.dtype).view(1, S, 1, 1)
        log_prior = -((s_idx - mu) ** 2) / (2.0 * sigma * sigma)  # (B,S,H,W)
        log_prior = log_prior.view(B, S, 1, H, W)

        # data logits
        rgb_rep = x_rgb.unsqueeze(1).expand(B, S, C, H, W)
        dep_rep = x_depth.unsqueeze(1).expand(B, S, C, H, W)
        inp = torch.cat([fs_slices, rgb_rep, dep_rep], dim=2)              # (B,S,3C,H,W)
        inp = inp.view(B * S, 3 * C, H, W)
        logits_data = self.logit_net(inp).view(B, S, 1, H, W)

        lambda_prior_eff = F.softplus(self.lambda_prior)  # >=0
        logits = logits_data + lambda_prior_eff * log_prior

        tau = torch.clamp(self.tau, 0.3, 5.0)
        alpha = torch.softmax(logits / tau, dim=1)                         # (B,S,1,H,W)

        x_focal = torch.sum(alpha * fs_slices, dim=1)                      # (B,C,H,W)

        p = alpha.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=1)                                # (B,1,H,W)
        u_f = entropy / math.log(S)                                        # normalized [0,1]

        return x_focal, alpha, u_f, lambda_prior_eff


class DINFCore(nn.Module):
    def __init__(self, dim, num_slices=12, norm_layer=None, ffn_expansion=2.0):
        super().__init__()
        self.dim = dim

        self.slice_gating = DepthGuidedSliceDistribution(dim, num_slices=num_slices)

        self.pos_attn_f = ChannelCrossAttention2d(dim, norm_layer=norm_layer)
        self.pos_attn_d = ChannelCrossAttention2d(dim, norm_layer=norm_layer)

        self.neg_attn_f = ChannelCrossAttention2d(dim, norm_layer=norm_layer)
        self.neg_attn_d = ChannelCrossAttention2d(dim, norm_layer=norm_layer)

        self.sim_head_f  = nn.Conv2d(1, 1, 1, bias=True)
        self.sim_head_d  = nn.Conv2d(1, 1, 1, bias=True)
        self.conf_head_f = nn.Conv2d(1, 1, 1, bias=True)
        self.conf_head_d = nn.Conv2d(1, 1, 1, bias=True)

        self.beta_head = nn.Conv2d(3, 2, 1, bias=True)  # sim_f, sim_d, (1-u_f)
        self.eta_head  = nn.Conv2d(3, 2, 1, bias=True)  # conf_f, conf_d, u_f

        self.gate_pos = nn.Conv2d(5, dim, 1, bias=True)
        self.gate_neg = nn.Conv2d(5, dim, 1, bias=True)

        self.ffn_pos = GatedDWFFN(dim, expansion=ffn_expansion)
        self.ffn_neg = GatedDWFFN(dim, expansion=ffn_expansion)

        self.gamma_pos = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-3)
        self.gamma_neg = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-3)

        self.out_fuse = nn.Conv2d(dim * 3, dim, 1, bias=True)

        with torch.no_grad():
            self.gate_neg.bias.fill_(-2.0)

    def _sim_conf(self, x_rgb, x_mod, sim_head, conf_head):
        sim  = torch.sigmoid(sim_head((x_rgb * x_mod).mean(dim=1, keepdim=True)))
        conf = torch.sigmoid(conf_head((x_rgb - x_mod).abs().mean(dim=1, keepdim=True)))
        return sim, conf

    def forward(self, x_rgb, fs_slices, x_depth, return_aux=False):
        B, C, H, W = x_rgb.shape
        assert x_depth.shape == x_rgb.shape, f"x_depth mismatch: {x_depth.shape} vs {x_rgb.shape}"
        assert fs_slices.dim() == 5 and fs_slices.shape[0] == B and fs_slices.shape[2] == C, \
            f"fs_slices expects (B,S,C,H,W), got {fs_slices.shape}"

        # 1) slice gating
        x_focal, alpha, u_f, lambda_prior_eff = self.slice_gating(fs_slices, x_rgb, x_depth)

        # 2) sim/conf
        sim_f, conf_f = self._sim_conf(x_rgb, x_focal, self.sim_head_f, self.conf_head_f)
        sim_d, conf_d = self._sim_conf(x_rgb, x_depth, self.sim_head_d, self.conf_head_d)

        # 3) pos messages
        m_pos_f = self.pos_attn_f(x_rgb, x_focal)
        m_pos_d = self.pos_attn_d(x_rgb, x_depth)

        beta_logits = self.beta_head(torch.cat([sim_f, sim_d, (1.0 - u_f)], dim=1))
        beta = torch.softmax(beta_logits, dim=1)
        m_pos = beta[:, 0:1] * m_pos_f + beta[:, 1:2] * m_pos_d

        # 4) neg messages
        Df = x_rgb - x_focal
        Dd = x_rgb - x_depth
        m_neg_f = self.neg_attn_f(Df, x_focal)
        m_neg_d = self.neg_attn_d(Dd, x_depth)

        eta_logits = self.eta_head(torch.cat([conf_f, conf_d, u_f], dim=1))
        eta = torch.softmax(eta_logits, dim=1)
        m_neg = eta[:, 0:1] * m_neg_f + eta[:, 1:2] * m_neg_d

        # 5) gates & update  (THIS is where g_pos/g_neg come from)
        gate_in = torch.cat([sim_f, sim_d, conf_f, conf_d, u_f], dim=1)  # (B,5,H,W)
        g_pos = torch.sigmoid(self.gate_pos(gate_in))  # (B,C,H,W)
        g_neg = torch.sigmoid(self.gate_neg(gate_in))  # (B,C,H,W)

        u_pos = self.ffn_pos(m_pos)
        u_neg = self.ffn_neg(m_neg)

        Y_p = x_rgb + self.gamma_pos * (g_pos * u_pos)
        Y_n = x_rgb - self.gamma_neg * (g_neg * u_neg)

        Y = self.out_fuse(torch.cat([Y_p, Y_n, x_rgb], dim=1)) + x_rgb

        if return_aux:
            aux = {
                "alpha": alpha, "u_f": u_f,
                "beta": beta, "eta": eta,
                "sim": (sim_f, sim_d),
                "conf": (conf_f, conf_d),
                "g_pos_mean": g_pos.mean(),
                "g_neg_mean": g_neg.mean(),
                "lambda_prior_eff": lambda_prior_eff.detach()
            }
            return Y, aux
        return Y


class DINFFusionStage(nn.Module):
    """
    stage-level DINF fusion:
      input tokens: rgb_seq (B,N,C), fs_seq (B*S,N,C), depth_seq (B,N,C)
      output tokens: fused_seq (B,N,C)
    """
    def __init__(self, dim, fea_reso, num_slices=12, ffn_expansion=2.0):
        super().__init__()
        self.dim = dim
        self.fea_reso = fea_reso
        self.S = int(num_slices)

        self.core = DINFCore(
            dim=dim,
            num_slices=num_slices,
            norm_layer=None,
            ffn_expansion=ffn_expansion
        )

    def _fsseq_to_slices(self, fs_seq: torch.Tensor, B: int, N: int, C: int, H: int, W: int) -> torch.Tensor:
        assert fs_seq.dim() == 3, f"fs_seq must be 3D, got {fs_seq.shape}"
        BS = fs_seq.shape[0]
        if BS == B:
            return PatchToImage(fs_seq).unsqueeze(1)  # (B,1,C,H,W)

        assert BS % B == 0, f"[DINF] fs_seq first dim {BS} not divisible by B={B}"
        S = BS // B
        assert S == self.S, f"[DINF] Expected S={self.S}, got S={S}"
        return PatchToImage(fs_seq).view(B, S, C, H, W)

    def forward(self, rgb_seq, fs_seq, depth_seq, return_aux=False):
        B, N, C = rgb_seq.shape
        H = W = self.fea_reso
        assert N == H * W, f"[DINF] N mismatch: N={N} vs {H}*{W}"
        assert depth_seq.shape == (B, N, C), f"[DINF] depth_seq mismatch: {depth_seq.shape}"

        x_rgb = PatchToImage(rgb_seq)
        x_depth = PatchToImage(depth_seq)
        fs_slices = self._fsseq_to_slices(fs_seq, B=B, N=N, C=C, H=H, W=W)

        if return_aux:
            Y, aux = self.core(x_rgb, fs_slices, x_depth, return_aux=True)
            return ImageToPatch(Y), aux
        else:
            Y = self.core(x_rgb, fs_slices, x_depth, return_aux=False)
            return ImageToPatch(Y)


# ============================================================
# MHFF Fusion Stage: MIRF(main) + DINF(aux) + omega gate (omega_scale is trainable)
# ============================================================
class MHFFFusionStage(nn.Module):
    """
    If fea_reso >= enable_parallel_when_reso_ge:
      compute MIRF and DINF, then:
        F = MIRF + omega*(DINF-MIRF)
      omega = sigmoid(Conv1x1(cat(MIRF_img,DINF_img,u_f))) * omega_scale
      omega_scale = sigmoid(omega_scale_logit) is LEARNABLE, in (0,1).
    Else:
      only MIRF is computed.
    """
    def __init__(self, dim, fea_reso, num_slices=12, ffn_expansion=2.0,
                 enable_parallel_when_reso_ge=28,
                 omega_init_bias=-2.0,
                 omega_scale_init=0.5,
                 learnable_omega_scale=True):
        super().__init__()
        self.dim = dim
        self.fea_reso = fea_reso
        self.enable_parallel = bool(fea_reso >= enable_parallel_when_reso_ge)
        self.learnable_omega_scale = bool(learnable_omega_scale)

        self.mirf_stage = MIRFFusionStage(dim=dim, fea_reso=fea_reso, num_slices=num_slices)
        self.dinf_stage = DINFFusionStage(dim=dim, fea_reso=fea_reso, num_slices=num_slices, ffn_expansion=ffn_expansion)

        if self.enable_parallel:
            self.omega_gate = nn.Conv2d(2 * dim + 1, 1, kernel_size=1, bias=True)
            nn.init.constant_(self.omega_gate.bias, float(omega_init_bias))

            # learnable scale in (0,1)
            eps = 1e-4
            init = float(omega_scale_init)
            init = max(eps, min(1.0 - eps, init))
            init_logit = math.log(init / (1.0 - init))
            if self.learnable_omega_scale:
                self.omega_scale_logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
            else:
                self.register_buffer("omega_scale_const", torch.tensor(init, dtype=torch.float32))
        else:
            self.omega_gate = None

    def omega_scale(self) -> torch.Tensor:
        if not self.enable_parallel:
            return torch.tensor(0.0)
        if self.learnable_omega_scale:
            return torch.sigmoid(self.omega_scale_logit)
        return self.omega_scale_const

    def forward(self, rgb_seq, fs_seq, depth_seq, return_aux=False):
        # MIRF always computed
        if return_aux:
            mirf_seq, aux_mirf = self.mirf_stage(rgb_seq, fs_seq, depth_seq, return_aux=True)
        else:
            mirf_seq = self.mirf_stage(rgb_seq, fs_seq, depth_seq, return_aux=False)
            aux_mirf = None

        if (not self.enable_parallel) or (self.omega_gate is None):
            return (mirf_seq, aux_mirf) if return_aux else mirf_seq

        # DINF only when enabled
        if return_aux:
            dinf_seq, aux_dinf = self.dinf_stage(rgb_seq, fs_seq, depth_seq, return_aux=True)
        else:
            dinf_seq = self.dinf_stage(rgb_seq, fs_seq, depth_seq, return_aux=False)
            aux_dinf = None

        mirf_img = PatchToImage(mirf_seq)
        dinf_img = PatchToImage(dinf_seq)

        # u_f from F3 (B,1,H,W)
        if (aux_dinf is not None) and ("u_f" in aux_dinf) and (aux_dinf["u_f"] is not None):
            u_f = aux_dinf["u_f"]
        else:
            B, C, H, W = mirf_img.shape
            u_f = torch.zeros((B, 1, H, W), device=mirf_img.device, dtype=mirf_img.dtype)

        omega_logits = self.omega_gate(torch.cat([mirf_img, dinf_img, u_f], dim=1))
        omega_base = torch.sigmoid(omega_logits)  # (0,1)
        omega = omega_base * self.omega_scale().to(omega_base.device).to(omega_base.dtype)

        fused_img = mirf_img + omega * (dinf_img - mirf_img)
        fused_seq = ImageToPatch(fused_img)

        if return_aux:
            aux = {
                "h2_enabled": True,
                "omega": omega,
                "omega_mean": omega.mean().detach(),
                "omega_scale": self.omega_scale().detach(),
                "omega_base_mean": omega_base.mean().detach(),
                # keep F3 uncertainty for your analysis
                "u_f": aux_dinf.get("u_f", None) if aux_dinf is not None else None,
                "alpha": aux_dinf.get("alpha", None) if aux_dinf is not None else None,
                "lambda_prior_eff": aux_dinf.get("lambda_prior_eff", None) if aux_dinf is not None else None,
                "aux_mirf": aux_mirf
            }
            return fused_seq, aux
        return fused_seq


# ============================================================
# DHPNet (H2 fusion)
# ============================================================
class DHPNet(nn.Module):
    def __init__(self, backbone_type='swin',
                 num_slices=12,
                 ffn_expansion=2.0,
                 enable_parallel_when_reso_ge=28,
                 omega_init_bias=-2.0,
                 omega_scale_init=0.5,
                 learnable_omega_scale=True):
        super().__init__()
        assert backbone_type == 'swin', "This file is prepared for Swin backbone version."

        self.num_slices = int(num_slices)

        img_size = 224
        depths = [2, 2, 6, 2]
        patch_size = 4
        embed_dim = 96
        self.channels = 96

        # RGB backbone
        self.backbone_rgb = SwinTransformerBackbone(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            embed_dim=embed_dim, depths=depths, num_heads=[3, 6, 12, 24], window_size=7
        )
        # Focal stack backbone
        self.backbone_fs = SwinTransformerBackbone(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            embed_dim=embed_dim, depths=depths, num_heads=[3, 6, 12, 24], window_size=7
        )
        # Depth encoder (unchanged)
        self.mdfe_encoder = MDFEEncoder(in_ch=1, embed_dim=embed_dim)

        self.image_size = img_size
        self.num_layers = len(depths)
        self.patch_reso = img_size // patch_size  # 56

        self.mhff_stages = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            fea_reso = self.patch_reso // (2 ** i_layer)  # 56,28,14,7
            dim = (2 ** i_layer) * embed_dim

            fusion_layer = MHFFFusionStage(
                dim=dim,
                fea_reso=fea_reso,
                num_slices=self.num_slices,
                ffn_expansion=ffn_expansion,
                enable_parallel_when_reso_ge=enable_parallel_when_reso_ge,
                omega_init_bias=omega_init_bias,
                omega_scale_init=omega_scale_init,
                learnable_omega_scale=learnable_omega_scale
            )
            self.mhff_stages.append(fusion_layer)

            upsample = PatchExpand(
                input_resolution=[fea_reso, fea_reso],
                in_dim=dim,
                out_dim=int(dim / 2),
                norm_layer=nn.LayerNorm
            )
            self.upsample.append(upsample)

        # reverse for decoder (deep->shallow)
        self.mhff_stages = self.mhff_stages[::-1]
        self.upsample = self.upsample[::-1]

        self.upsample_x4 = FinalPatchExpand_X4(
            input_resolution=[self.patch_reso, self.patch_reso],
            dim=embed_dim,
            dim_scale=4,
            norm_layer=nn.LayerNorm
        )

        # ---- Decoder heads ----
        self.score_module = ScoreModule(self.channels)
        self.score_module_coarse = ScoreModule(self.channels)

        self.era = ERA_MS_EGA(
            channels=self.channels,
            gauss_ks=5,
            init_sigmas=(0.8, 1.2, 2.0),
            morph_ks=3,
            mask_smooth_ks=5,
            mask_dilate_ks=7
        )

        self.fuse_edge_region = Conv3(2 * self.channels, self.channels)

    def load_pretrained(self, load_path):
        if not os.path.exists(load_path):
            print("[WARN] pretrained model path not exist:", load_path)
            return
        pretrained = torch.load(load_path, map_location='cpu')
        pretrained_dict = pretrained['model'] if isinstance(pretrained, dict) and 'model' in pretrained else pretrained

        model_dict = self.backbone_rgb.state_dict()
        renamed_dict = {}
        for k, v in pretrained_dict.items():
            k2 = k.replace('layers.0.downsample', 'downsamples.0')
            k2 = k2.replace('layers.1.downsample', 'downsamples.1')
            k2 = k2.replace('layers.2.downsample', 'downsamples.2')
            if k2 in model_dict:
                renamed_dict[k2] = v
        model_dict.update(renamed_dict)

        self.backbone_rgb.load_state_dict(model_dict, strict=True)
        self.backbone_fs.load_state_dict(model_dict, strict=True)
        print("RGB/FS Pretrained Loaded. (Depth conv encoder keeps random init)")

    def _normalize_fs_input(self, fs, B: int):
        """
        Normalize fs to (B*S,3,H,W).
        Accept:
          - (B,S,3,H,W)
          - (S,3,H,W) when B==1
          - (B*S,3,H,W)
        Returns:
          fs_in, S
        """
        assert fs.dim() in (4, 5), f"Unsupported fs shape: {fs.shape}"
        if fs.dim() == 5:
            b2, S, c, h, w = fs.shape
            assert b2 == B and c == 3, f"fs must be (B,S,3,H,W), got {fs.shape}"
            return fs.view(B * S, 3, h, w).contiguous(), int(S)

        # fs.dim()==4
        assert fs.size(1) == 3, f"fs must have 3 channels per slice, got {fs.shape}"
        if fs.shape[0] == self.num_slices and B == 1:
            return fs.contiguous(), int(self.num_slices)

        BS = fs.shape[0]
        assert BS % B == 0, f"fs first dim {BS} not divisible by B={B}"
        S = BS // B
        return fs.contiguous(), int(S)

    def forward(self, fs, rgb, depth=None, return_aux=False):
        """
        fs:    (S,3,H,W) or (B,S,3,H,W) or (B*S,3,H,W)
        rgb:   (B,3,H,W)
        depth: (B,1,H,W)
        """
        assert rgb.dim() == 4 and rgb.size(1) == 3, f"rgb must be (B,3,H,W), got {rgb.shape}"
        B, _, H, W = rgb.shape
        if depth is None:
            raise ValueError("This DHPNet version expects depth input (B,1,H,W).")
        assert depth.dim() == 4 and depth.size(1) == 1, f"depth must be (B,1,H,W), got {depth.shape}"

        fs_in, S = self._normalize_fs_input(fs, B=B)

        # 1) backbones (shallow->deep)
        side_rgb_x = self.backbone_rgb(rgb)      # [(B,N,C)...]
        side_fs_x  = self.backbone_fs(fs_in)     # [(B*S,N,C)...]
        side_dep_x = self.mdfe_encoder(depth)  # [(B,N,C)...]

        # 2) deep->shallow
        side_rgb_x = side_rgb_x[::-1]
        side_fs_x  = side_fs_x[::-1]
        side_dep_x = side_dep_x[::-1]

        aux_all = [] if return_aux else None

        # 3) stage fusion (deep->shallow with upsampling)
        if return_aux:
            fused_fea, aux0 = self.mhff_stages[0](side_rgb_x[0], side_fs_x[0], side_dep_x[0], return_aux=True)
            aux_all.append(aux0)
        else:
            fused_fea = self.mhff_stages[0](side_rgb_x[0], side_fs_x[0], side_dep_x[0], return_aux=False)

        for i in range(1, self.num_layers):
            fused_fea = self.upsample[i - 1](fused_fea)
            rgb_in = side_rgb_x[i] + fused_fea
            if return_aux:
                fused_fea, auxi = self.mhff_stages[i](rgb_in, side_fs_x[i], side_dep_x[i], return_aux=True)
                aux_all.append(auxi)
            else:
                fused_fea = self.mhff_stages[i](rgb_in, side_fs_x[i], side_dep_x[i], return_aux=False)

        # 4) decode
        fused_fea_img = self.upsample_x4(fused_fea)  # (B,96,224,224)

        coarse = self.score_module_coarse(fused_fea_img)

        edge_feature, contour = self.era(fused_fea_img, rgb, coarse)

        final_feat = self.fuse_edge_region(torch.cat((edge_feature, fused_fea_img), dim=1))
        pred = self.score_module(final_feat)

        if return_aux:
            return pred, contour, coarse, aux_all
        return pred, contour, coarse


# ============================================================
# init_weights (keep your style)
# ============================================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


if __name__ == "__main__":
    net = DHPNet(
        backbone_type="swin",
        num_slices=12,
        ffn_expansion=2.0,
        enable_parallel_when_reso_ge=28,  # only 56/28 parallel
        omega_init_bias=-2.0,
        omega_scale_init=0.5,
        learnable_omega_scale=True
    )
    net.apply(init_weights)
    print("Params (M):", sum(p.numel() for p in net.parameters()) / 1e6)

    rgb = torch.randn(1, 3, 224, 224)
    depth = torch.randn(1, 1, 224, 224)
    fs = torch.randn(12, 3, 224, 224)

    with torch.no_grad():
        pred, contour, coarse, aux = net(fs, rgb, depth, return_aux=True)
        print(pred.shape, contour.shape, coarse.shape, len(aux))
        print("aux keys:", list(aux[-1].keys()) if aux[-1] is not None else None)
