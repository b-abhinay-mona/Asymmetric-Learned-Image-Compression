# src/models/hierarchical_cnn.py - LIGHTWEIGHT VERSION with MBConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) for lightweight processing"""
    def __init__(self, in_channels, out_channels, expansion=4, kernel_size=3, stride=1):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim)
        ) if expansion > 1 else nn.Identity()
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim)
        )
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable conv: depthwise (groups=in_ch) followed by pointwise conv."""
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: Optional[int] = None, bias: bool = False):
        super().__init__()
        if p is None:
            p = k // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))

class SimpleHierarchicalCNN(nn.Module):
    """Lightweight hierarchical structure with depthwise separable"""
    def __init__(self, in_channels, num_stages=1, base_width=64):
        super().__init__()
        
        # Use depthwise separable convolutions
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, base_width, 3),
            nn.BatchNorm2d(base_width),
            nn.PReLU(base_width)
        )

        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv2d(base_width, base_width, 3),
            nn.BatchNorm2d(base_width),
            nn.PReLU(base_width)
        )

        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv2d(base_width, base_width, 3),
            nn.BatchNorm2d(base_width),
            nn.PReLU(base_width)
        )

        # Simple residual connection
        if in_channels != base_width:
            self.skip_proj = nn.Conv2d(in_channels, base_width, 1)
        else:
            self.skip_proj = nn.Identity()

        self.output_channels = base_width

    def forward(self, x):
        # Simple sequential processing
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Basic residual connection
        skip = self.skip_proj(x)
        if skip.shape[-2:] != out.shape[-2:]:
            skip = F.interpolate(skip, size=out.shape[-2:], mode='nearest')
        return out + skip

class ImprovedHierarchicalCNN(nn.Module):
    """Lightweight 2-stage with MBConv blocks"""
    def __init__(self, in_channels, base_width=64):
        super().__init__()
        
        # Use MBConv instead of standard convs
        self.stage1 = MBConvBlock(in_channels, base_width, expansion=4, kernel_size=3)
        self.stage2 = MBConvBlock(base_width, base_width, expansion=2, kernel_size=3)
        
        # Skip connection
        if in_channels != base_width:
            self.skip_proj = nn.Conv2d(in_channels, base_width, 1)
        else:
            self.skip_proj = nn.Identity()

        self.output_channels = base_width

    def forward(self, x):
        # Stage 1: Initial feature extraction
        s1 = self.stage1(x)
        # Stage 2: Refined processing
        s2 = self.stage2(s1)

        # Residual connection
        skip = self.skip_proj(x)
        if skip.shape[-2:] != s2.shape[-2:]:
            skip = F.interpolate(skip, size=s2.shape[-2:], mode='nearest')
        return s2 + skip

# Keep existing complex classes for future use

class HierarchicalMultiheadBlock(nn.Module):
    """Hierarchical Multiheaded Convolution Network block - replacement for MSRB"""
    def __init__(self, in_ch: int, width: int, stages_k: List[int], use_attention: bool = True):
        super().__init__()
        assert width >= 2, "Width must be >=2 to allow splitting."
        self.in_ch = in_ch
        self.width = width
        self.stages_k = stages_k
        self.part_a_ch = width // 2
        self.num_stages = len(stages_k)
        self.use_attention = use_attention

        # Convolutions for each stage
        self.convs = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv2d(in_ch if i == 0 else self.part_a_ch, width, k),
                nn.BatchNorm2d(width),
                nn.PReLU(width)
            ) for i, k in enumerate(stages_k)
        ])

        # Channel attention for each stage
        if self.use_attention:
            self.channel_attentions = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(self.part_a_ch, self.part_a_ch // 4, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.part_a_ch // 4, self.part_a_ch, 1),
                    nn.Sigmoid()
                ) for _ in range(self.num_stages)
            ])

        # Residual projection if needed
        out_ch_total = self.part_a_ch * self.num_stages
        if in_ch != out_ch_total:
            self.skip_proj = nn.Conv2d(in_ch, out_ch_total, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_ch_total, out_ch_total, 3, padding=1, groups=out_ch_total//4),
            nn.BatchNorm2d(out_ch_total),
            nn.PReLU(out_ch_total),
            nn.Conv2d(out_ch_total, out_ch_total, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        inp = x
        for i, conv in enumerate(self.convs):
            y = conv(inp)
            part_a = y[:, :self.part_a_ch, :, :]
            if self.use_attention:
                att = self.channel_attentions[i](part_a)
                part_a = part_a * att
            outs.append(part_a)
            inp = part_a

        out = torch.cat(outs, dim=1)
        out = self.fusion(out)
        skip = self.skip_proj(x)
        if skip.shape[-2:] != out.shape[-2:]:
            skip = F.interpolate(skip, size=out.shape[-2:], mode='nearest')
        return out + skip

class HierarchicalCNN(nn.Module):
    """Complete Hierarchical CNN module with multiple stages - ORIGINAL COMPLEX VERSION"""
    def __init__(self, in_channels, num_stages=3, base_width=64, stage_configs=None):
        super().__init__()
        if stage_configs is None:
            stage_configs = [
                {'width': base_width, 'stages_k': [3, 5, 7]},
                {'width': base_width * 2, 'stages_k': [3, 5]},
                {'width': base_width * 4, 'stages_k': [3, 7]}
            ]

        self.stages = nn.ModuleList()
        current_channels = in_channels
        for i, config in enumerate(stage_configs[:num_stages]):
            stage = HierarchicalMultiheadBlock(
                current_channels,
                config['width'],
                config['stages_k'],
                use_attention=(i > 0)
            )

            self.stages.append(stage)
            current_channels = (config['width'] // 2) * len(config['stages_k'])

        self.output_channels = current_channels

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
