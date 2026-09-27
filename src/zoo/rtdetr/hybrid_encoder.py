'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class XLCEBlock(nn.Module):
    """
    X-ray Local Contrast Enhancement Block.

    This block enhances local contrast and weak boundary cues in low-level
    high-resolution X-ray features. The residual scaling parameter alpha is
    initialized to zero, so the block behaves like an identity mapping at the
    beginning of training.
    """
    def __init__(self, channels, act="silu"):
        super().__init__()

        self.local_enhance = nn.Sequential(
            ConvNormLayer(channels, channels, 3, 1, padding=1, act=act),
            ConvNormLayer(channels, channels, 3, 1, padding=1, act=act),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        enhanced = self.local_enhance(x)
        gate = self.gate(x)
        return x + self.alpha * gate * enhanced


class SCXLCEBlock(nn.Module):
    """
    Scale-Consistent X-ray Local Context Enhancement Block.

    It keeps the XLCE local enhancement branch, but gates the residual with
    consistency between the current feature and an adjacent P3/P4 reference.
    """
    def __init__(
        self,
        channels,
        ref_channels=None,
        act="silu",
        use_background_suppression=False,
        bs_lambda=0.1,
    ):
        super().__init__()

        ref_channels = channels if ref_channels is None else ref_channels

        self.local_enhance = nn.Sequential(
            ConvNormLayer(channels, channels, 3, 1, padding=1, act=act),
            ConvNormLayer(channels, channels, 3, 1, padding=1, act=act),
        )

        if ref_channels == channels:
            self.ref_align = nn.Identity()
        else:
            self.ref_align = ConvNormLayer(ref_channels, channels, 1, 1, padding=0, act=act)

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))
        self.use_background_suppression = use_background_suppression
        self.bs_lambda = bs_lambda
        self.last_debug_shapes = None

    def forward(self, x, ref):
        ref = self.ref_align(ref)
        if ref.shape[-2:] != x.shape[-2:]:
            ref = F.interpolate(
                ref,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        enhanced = self.local_enhance(x)
        gate = self.gate(torch.cat([x, ref, torch.abs(x - ref)], dim=1))
        residual = gate * enhanced

        if self.use_background_suppression:
            residual = residual - self.bs_lambda * (1.0 - gate) * enhanced

        self.last_debug_shapes = {
            "current": tuple(x.shape),
            "reference": tuple(ref.shape),
            "gate": tuple(gate.shape),
        }

        return x + self.alpha * residual


class SCXLCESoftBlock(nn.Module):
    """
    Residual SC-XLCE with a soft scale-consistency gate.

    Default behavior is exactly the original SC-XLCE-Soft:
        x + (0.5 + 0.5 * sigmoid(logit)) * (XLCE(x) - x)

    The extra tuning parameters do not introduce learnable parameters and do
    not change state_dict keys:
        - sc_gate_temperature: smooths/sharpens the consistency gate logits.
        - sc_gate_floor: applies an additional weak-cue-preserving floor after
          the original soft gate, so default 0.0 keeps the original behavior.
        - enhance_alpha / scale_alpha: residual strength multipliers.
    """
    def __init__(
        self,
        channels,
        ref_channels=None,
        act="silu",
        sc_gate_floor=0.0,
        sc_gate_temperature=1.0,
        sc_soft_base=0.5,
        sc_relation_mode='full',
        sc_payload_injection=False,
        enhance_alpha=1.0,
        scale_name=None,
    ):
        super().__init__()

        ref_channels = channels if ref_channels is None else ref_channels

        self.xlce = XLCEBlock(channels, act=act)

        if ref_channels == channels:
            self.ref_align = nn.Identity()
        else:
            self.ref_align = ConvNormLayer(ref_channels, channels, 1, 1, padding=0, act=act)

        # Keep the Sequential structure to preserve checkpoint state_dict keys:
        # consistency_gate.0.weight / consistency_gate.0.bias.
        self.consistency_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.sc_gate_floor = float(sc_gate_floor)
        self.sc_gate_temperature = float(sc_gate_temperature)
        self.sc_soft_base = float(sc_soft_base)
        self.sc_relation_mode = str(sc_relation_mode)
        self.sc_payload_injection = bool(sc_payload_injection)
        self.enhance_alpha = float(enhance_alpha)
        self.scale_name = scale_name

        if self.sc_gate_temperature <= 0:
            raise ValueError(f"sc_gate_temperature must be > 0, got {self.sc_gate_temperature}")
        if not (0.0 <= self.sc_gate_floor < 1.0):
            raise ValueError(f"sc_gate_floor must be in [0, 1), got {self.sc_gate_floor}")
        if not (0.0 <= self.sc_soft_base < 1.0):
            raise ValueError(f"sc_soft_base must be in [0, 1), got {self.sc_soft_base}")
        if self.sc_relation_mode not in ['full', 'no_diff']:
            raise ValueError(
                f"sc_relation_mode must be 'full' or 'no_diff', got {self.sc_relation_mode}"
            )

        self.last_debug_shapes = None

    def forward(self, x, ref):
        xlce_out = self.xlce(x)
        residual = xlce_out - x

        ref = self.ref_align(ref)
        if ref.shape[-2:] != x.shape[-2:]:
            ref = F.interpolate(
                ref,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        diff = torch.abs(x - ref)

        if self.sc_relation_mode == 'full':
            gate_input = torch.cat([x, ref, diff], dim=1)
        elif self.sc_relation_mode == 'no_diff':
            # Zero-pad the relation slot so gate dimensionality and
            # parameter count remain identical to the full model.
            gate_input = torch.cat([x, ref, torch.zeros_like(diff)], dim=1)

        gate_logit = self.consistency_gate[0](gate_input)
        gate = torch.sigmoid(gate_logit / self.sc_gate_temperature)

        # Residual-preserving scale-consistency modulation.
        # sc_soft_base=0.5 exactly reproduces the original Soft behavior.
        # sc_soft_base=0.0 gives the fair hard-gating ablation.
        gate_soft = self.sc_soft_base + (1.0 - self.sc_soft_base) * gate

        # Additional weak-cue-preserving floor.
        # default sc_gate_floor=0.0 keeps gate_soft unchanged.
        gate_soft = self.sc_gate_floor + (1.0 - self.sc_gate_floor) * gate_soft

        self.last_debug_shapes = {
            "scale_name": self.scale_name,
            "current": tuple(x.shape),
            "reference": tuple(ref.shape),
            "gate": tuple(gate.shape),
            "gate_soft": tuple(gate_soft.shape),
            "sc_gate_floor": self.sc_gate_floor,
            "sc_gate_temperature": self.sc_gate_temperature,
            "sc_soft_base": self.sc_soft_base,
            "sc_relation_mode": self.sc_relation_mode,
            "sc_payload_injection": self.sc_payload_injection,
            "enhance_alpha": self.enhance_alpha,
        }

        if self.sc_payload_injection:
            # Controlled cross-scale payload ablation:
            # keep the same LEM, relation modeling and soft gate,
            # while additionally injecting the aligned cross-scale residual.
            payload_residual = ref - x
            return x + self.enhance_alpha * gate_soft * (
                residual + payload_residual
            )

        return x + self.enhance_alpha * gate_soft * residual


class SCXLCESoftAWPBlock(nn.Module):
    """
    Adaptive Weak-Cue Preserving SC-XLCE-Soft.

    This block keeps the original SC-XLCE-Soft residual form and adds two
    learnable scalar controls:
      1) weak_floor: a bounded adaptive lower-bound for soft gate preservation.
      2) residual_scale: a bounded adaptive residual scale.

    Initialization is designed to be close to the original SC-XLCE-Soft:
      weak_floor ~= 0 when awp_floor_init is a large negative value.
      residual_scale = 1 when awp_scale_init = 0.
    """
    def __init__(
        self,
        channels,
        ref_channels=None,
        act="silu",
        max_floor=0.2,
        max_scale_delta=0.2,
        floor_init=-10.0,
        scale_init=0.0,
        scale_name=None,
    ):
        super().__init__()

        ref_channels = channels if ref_channels is None else ref_channels

        self.xlce = XLCEBlock(channels, act=act)

        if ref_channels == channels:
            self.ref_align = nn.Identity()
        else:
            self.ref_align = ConvNormLayer(ref_channels, channels, 1, 1, padding=0, act=act)

        self.consistency_gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.floor_logit = nn.Parameter(torch.tensor(float(floor_init)))
        self.scale_logit = nn.Parameter(torch.tensor(float(scale_init)))

        self.max_floor = float(max_floor)
        self.max_scale_delta = float(max_scale_delta)
        self.scale_name = scale_name
        self.last_debug_shapes = None

    def forward(self, x, ref):
        xlce_out = self.xlce(x)
        residual = xlce_out - x

        ref = self.ref_align(ref)
        if ref.shape[-2:] != x.shape[-2:]:
            ref = F.interpolate(
                ref,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        gate = self.consistency_gate(torch.cat([x, ref, torch.abs(x - ref)], dim=1))

        # Original SC-XLCE-Soft gate.
        gate_soft = 0.5 + 0.5 * gate

        # Adaptive weak-cue preservation.
        weak_floor = self.max_floor * torch.sigmoid(self.floor_logit)
        gate_soft = weak_floor + (1.0 - weak_floor) * gate_soft

        # Adaptive residual scaling, bounded around 1.0.
        residual_scale = 1.0 + self.max_scale_delta * torch.tanh(self.scale_logit)

        self.last_debug_shapes = {
            "scale_name": self.scale_name,
            "current": tuple(x.shape),
            "reference": tuple(ref.shape),
            "gate": tuple(gate.shape),
            "gate_soft": tuple(gate_soft.shape),
            "weak_floor": float(weak_floor.detach().cpu()),
            "residual_scale": float(residual_scale.detach().cpu()),
        }

        return x + residual_scale * gate_soft * residual


class HGIFBlock(nn.Module):
    """
    High-level Guided Interaction Fusion Block.

    It uses high-level semantic features to guide low-level high-resolution
    features before FPN fusion. The residual scaling parameter beta is
    initialized to zero for stable training.
    """
    def __init__(self, channels, act="silu"):
        super().__init__()

        self.low_proj = ConvNormLayer(channels, channels, 1, 1, padding=0, act=act)
        self.high_proj = ConvNormLayer(channels, channels, 1, 1, padding=0, act=act)

        self.interact = nn.Sequential(
            ConvNormLayer(channels * 2, channels, 3, 1, padding=1, act=act),
            ConvNormLayer(channels, channels, 3, 1, padding=1, act=act),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, feat_low, feat_high):
        low = self.low_proj(feat_low)
        high = self.high_proj(feat_high)

        fusion = torch.cat([low, high], dim=1)
        guidance = self.interact(fusion)
        gate = self.gate(fusion)

        return feat_low + self.beta * gate * guidance


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 use_xray_enhance=False,
                 xray_enhance_indices=None,
                 xray_enhance_type='xlce',
                 use_scale_consistency=False,
                 use_background_suppression=False,
                 bs_lambda=0.1,
                 sc_gate_floor=0.0,
                 sc_gate_temperature=1.0,
                 sc_soft_base=0.5,
                 sc_relation_mode='full',
                 sc_payload_injection=False,
                 sc_enhance_alpha=1.0,
                 sc_alpha_p3=1.0,
                 sc_alpha_p4=1.0,
                 awp_max_floor=0.2,
                 awp_max_scale_delta=0.2,
                 awp_floor_init=-10.0,
                 awp_scale_init=0.0,
                 use_hgif=False):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_xray_enhance = use_xray_enhance
        self.xray_enhance_type = xray_enhance_type
        self.use_scale_consistency = use_scale_consistency
        self.use_background_suppression = use_background_suppression
        self.bs_lambda = bs_lambda
        self.sc_gate_floor = float(sc_gate_floor)
        self.sc_gate_temperature = float(sc_gate_temperature)
        self.sc_soft_base = float(sc_soft_base)
        self.sc_relation_mode = str(sc_relation_mode)
        self.sc_payload_injection = bool(sc_payload_injection)
        self.sc_enhance_alpha = float(sc_enhance_alpha)
        self.sc_alpha_p3 = float(sc_alpha_p3)
        self.sc_alpha_p4 = float(sc_alpha_p4)
        self.awp_max_floor = float(awp_max_floor)
        self.awp_max_scale_delta = float(awp_max_scale_delta)
        self.awp_floor_init = float(awp_floor_init)
        self.awp_scale_init = float(awp_scale_init)
        if self.sc_gate_temperature <= 0:
            raise ValueError(f"sc_gate_temperature must be > 0, got {self.sc_gate_temperature}")
        if not (0.0 <= self.sc_gate_floor < 1.0):
            raise ValueError(f"sc_gate_floor must be in [0, 1), got {self.sc_gate_floor}")
        if not (0.0 <= self.sc_soft_base < 1.0):
            raise ValueError(f"sc_soft_base must be in [0, 1), got {self.sc_soft_base}")
        self._sc_xlce_debug_printed = False
        if xray_enhance_indices is None:
            self.xray_enhance_indices = None
        else:
            self.xray_enhance_indices = set(xray_enhance_indices)
        self.use_hgif = use_hgif
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        if self.use_xray_enhance:
            if self.xray_enhance_type == 'sc_xlce':
                self.xray_enhance = nn.ModuleList([
                    SCXLCEBlock(
                        hidden_dim,
                        ref_channels=hidden_dim,
                        act=act,
                        use_background_suppression=use_background_suppression,
                        bs_lambda=bs_lambda,
                    )
                    for _ in range(len(in_channels) - 1)
                ])
            elif self.xray_enhance_type == 'sc_xlce_soft_awp':
                self.xray_enhance = nn.ModuleList()
                for fpn_idx in range(len(in_channels) - 1):
                    if fpn_idx == 0:
                        scale_name = 'P4'
                    elif fpn_idx == 1:
                        scale_name = 'P3'
                    else:
                        scale_name = f'P_unknown_{fpn_idx}'

                    self.xray_enhance.append(
                        SCXLCESoftAWPBlock(
                            hidden_dim,
                            ref_channels=hidden_dim,
                            act=act,
                            max_floor=self.awp_max_floor,
                            max_scale_delta=self.awp_max_scale_delta,
                            floor_init=self.awp_floor_init,
                            scale_init=self.awp_scale_init,
                            scale_name=scale_name,
                        )
                    )
            elif self.xray_enhance_type in ['sc_xlce_soft', 'sc_xlce_soft_awp']:
                self.xray_enhance = nn.ModuleList()
                for fpn_idx in range(len(in_channels) - 1):
                    # Confirmed mapping in this HybridEncoder:
                    # fpn_idx=0 -> P4, fpn_idx=1 -> P3.
                    if fpn_idx == 0:
                        scale_name = 'P4'
                        scale_alpha = self.sc_alpha_p4
                    elif fpn_idx == 1:
                        scale_name = 'P3'
                        scale_alpha = self.sc_alpha_p3
                    else:
                        scale_name = f'P_unknown_{fpn_idx}'
                        scale_alpha = 1.0

                    self.xray_enhance.append(
                        SCXLCESoftBlock(
                            hidden_dim,
                            ref_channels=hidden_dim,
                            act=act,
                            sc_gate_floor=self.sc_gate_floor,
                            sc_gate_temperature=self.sc_gate_temperature,
                            sc_soft_base=self.sc_soft_base,
                            sc_relation_mode=self.sc_relation_mode,
                            sc_payload_injection=self.sc_payload_injection,
                            enhance_alpha=self.sc_enhance_alpha * scale_alpha,
                            scale_name=scale_name,
                        )
                    )
            else:
                self.xray_enhance = nn.ModuleList([
                    XLCEBlock(hidden_dim, act=act)
                    for _ in range(len(in_channels) - 1)
                ])
        else:
            self.xray_enhance = nn.ModuleList()

        if self.use_hgif:
            self.hgif_blocks = nn.ModuleList([
                HGIFBlock(hidden_dim, act=act)
                for _ in range(len(in_channels) - 1)
            ])
        else:
            self.hgif_blocks = nn.ModuleList()

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def _get_sc_xlce_reference(self, proj_feats, fpn_idx):
        current_idx = len(self.in_channels) - 2 - fpn_idx

        if current_idx == 0 and len(proj_feats) > 1:
            return proj_feats[1]
        if current_idx == 1 and len(proj_feats) > 0:
            return proj_feats[0]

        return None

    def _print_sc_xlce_debug_once(self, fpn_idx, block):
        if self._sc_xlce_debug_printed:
            return

        indices = None
        if self.xray_enhance_indices is not None:
            indices = sorted(self.xray_enhance_indices)

        print(f"SC-XLCE enabled: {self.xray_enhance_type}")
        print(f"xray_enhance_indices: {indices}")
        print(f"fpn_idx: {fpn_idx}")
        print(f"feature shapes: {block.last_debug_shapes}")
        print(f"use_background_suppression: {self.use_background_suppression}")
        print(f"bs_lambda: {self.bs_lambda}")

        self._sc_xlce_debug_printed = True

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]

            fpn_idx = len(self.in_channels) - 1 - idx

            if self.use_xray_enhance and (
                self.xray_enhance_indices is None or fpn_idx in self.xray_enhance_indices
            ):
                if self.xray_enhance_type in ['sc_xlce', 'sc_xlce_soft', 'sc_xlce_soft_awp']:
                    ref_feat = self._get_sc_xlce_reference(proj_feats, fpn_idx)
                    if ref_feat is None:
                        raise ValueError(
                            "SC-XLCE requires adjacent P3/P4 reference features."
                        )
                    feat_low = self.xray_enhance[fpn_idx](feat_low, ref_feat)
                    self._print_sc_xlce_debug_once(fpn_idx, self.xray_enhance[fpn_idx])
                else:
                    feat_low = self.xray_enhance[fpn_idx](feat_low)

            feat_high = self.lateral_convs[fpn_idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')

            if self.use_hgif:
                feat_low = self.hgif_blocks[fpn_idx](feat_low, upsample_feat)

            inner_out = self.fpn_blocks[fpn_idx](
                torch.concat([upsample_feat, feat_low], dim=1)
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs
