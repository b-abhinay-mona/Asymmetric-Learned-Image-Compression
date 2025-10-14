# src/models/fixed_compression_model.py - LIGHTWEIGHT with Depthwise Separable + LN+PReLU

import torch
import torch.nn as nn
import torch.nn.functional as F
from .hierarchical_cnn import SimpleHierarchicalCNN, ImprovedHierarchicalCNN, MBConvBlock
from .wavelet_layers import DWT_2D, IDWT_2D

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        # Depthwise
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, 
                                   padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.prelu1 = nn.PReLU(in_ch)
        
        # Pointwise
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.prelu2 = nn.PReLU(out_ch)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        return x

class PostQuantizationFilter(nn.Module):
    """Lightweight PQF with depthwise separable convolutions"""
    def __init__(self, N=128):
        super().__init__()
        self.pqf = nn.Sequential(
            DepthwiseSeparableConv(N, N//2, 3),
            DepthwiseSeparableConv(N//2, N//2, 3),
            nn.Conv2d(N//2, N, 1),
            nn.Tanh()
        )
        # Keep WAM but simplified
        self.wam_pqf = PaperDWANAttention(N)

    def forward(self, x):
        # Apply PQF
        filtered = x + 0.1 * self.pqf(x)
        # Apply WAM for spectral refinement
        refined, _ = self.wam_pqf(filtered)
        return refined

class SSMContextModel(nn.Module):
    """Lightweight context model with depthwise separable convs"""
    def __init__(self, N=128):
        super().__init__()
        # Local context with depthwise separable
        self.context_prediction = nn.Sequential(
            DepthwiseSeparableConv(N, N, 3),
            DepthwiseSeparableConv(N, N, 3)
        )
        
        # Global context (keep 1x1 convs as they're already efficient)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(N, N//4, 1),
            nn.PReLU(N//4),
            nn.Conv2d(N//4, N, 1),
            nn.Sigmoid()
        )
        
        # Simplified - remove WAM here for efficiency
        self.output_layer = nn.Conv2d(N, N*2, 1)

    def forward(self, x):
        # Local context prediction
        local_ctx = self.context_prediction(x)
        # Global context (SSM-style)
        global_ctx = self.global_context(x)
        
        # Combine local + global (simplified fusion)
        fused_ctx = local_ctx * global_ctx
        ctx = self.output_layer(fused_ctx)
        
        mean, scale = torch.chunk(ctx, 2, dim=1)
        scale = F.softplus(scale) + 0.1
        return mean, scale

class PaperDWANAttention(nn.Module):
    """Lightweight DWAN with depthwise separable convolutions"""
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = DWT_2D(wavename='haar')
        self.importance_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels * 4, in_channels // 2, 3),
            DepthwiseSeparableConv(in_channels // 2, in_channels // 4, 3),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()  # Use sigmoid for gentle attention
        )

    def forward(self, x):
        B, C, H, W = x.shape
        LL, LH, HL, HH = self.dwt(x)
        wavelet_features = torch.cat([LL, LH, HL, HH], dim=1)
        importance = self.importance_conv(wavelet_features)
        importance = F.interpolate(importance, size=(H, W), mode='bilinear', align_corners=False)
        # MODERATE: Gentle attention that preserves information
        importance = 0.3 + 0.7 * importance  # Range: [0.3, 1.0]
        masked_x = x * importance
        return masked_x, importance

class LightweightNorm(nn.Module):
    """Lightweight normalization: LN + PReLU replaces GDN/IGDN"""
    def __init__(self, channels, inverse=False):
        super().__init__()
        self.inverse = inverse
        # LayerNorm normalized over C dimension
        self.norm = nn.LayerNorm([channels], elementwise_affine=True)
        self.activation = nn.PReLU(channels)
    
    def forward(self, x):
        # x: B, C, H, W
        B, C, H, W = x.shape
        # Permute to B, H, W, C for LayerNorm
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # Back to B, C, H, W
        x = self.activation(x)
        return x

class PaperEncoder(nn.Module):
    """Lightweight encoder with MBConv and depthwise separable"""
    def __init__(self, N=128):
        super().__init__()
        self.N = N
        
        # Initial conv (keep standard for first layer)
        self.conv1 = nn.Conv2d(3, N, 5, stride=2, padding=2)
        self.norm1 = LightweightNorm(N)  # Replace GDN
        
        # Stage 1: MBConv block
        self.hierarchical_cnn1 = MBConvBlock(N, N, expansion=4, kernel_size=3)
        
        # Downsample with depthwise separable
        self.conv2 = DepthwiseSeparableConv(N, N, kernel_size=5, stride=2, padding=2)
        self.norm2 = LightweightNorm(N)
        
        # Stage 2: Single lightweight block
        self.hierarchical_cnn2 = MBConvBlock(N, N, expansion=2, kernel_size=3)
        
        # Downsample
        self.conv3 = DepthwiseSeparableConv(N, N, kernel_size=5, stride=2, padding=2)
        self.norm3 = LightweightNorm(N)
        
        # Stage 3: Final MBConv
        self.hierarchical_cnn3 = MBConvBlock(N, N, expansion=4, kernel_size=3)
        
        # Output projection
        self.conv_out = nn.Conv2d(N, N, 3, padding=1)

    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = self.hierarchical_cnn1(x)
        x = self.norm2(self.conv2(x))
        x = self.hierarchical_cnn2(x)
        x = self.norm3(self.conv3(x))
        x = self.hierarchical_cnn3(x)
        x = self.conv_out(x)
        return x

class PaperDecoder(nn.Module):
    """Lightweight asymmetric decoder - single lightweight block"""
    def __init__(self, N=128):
        super().__init__()
        
        # Lightweight PQF
        self.pqf = PostQuantizationFilter(N)
        
        # Initial conv
        self.conv1 = nn.Conv2d(N, N, 3, padding=1)
        
        # Single lightweight block instead of hierarchical CNN
        self.hierarchical_cnn = MBConvBlock(N, N, expansion=2, kernel_size=3)
        
        # Upsampling layers
        self.deconv1 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.norm1 = LightweightNorm(N)  # Replace IGDN
        
        self.deconv2 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.norm2 = LightweightNorm(N)
        
        # Final upsampling
        self.deconv3 = nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x_filtered = self.pqf(x)  # Enhanced PQF with WAM
        x = self.conv1(x_filtered)
        x = self.hierarchical_cnn(x)
        x = self.norm1(self.deconv1(x))
        x = self.norm2(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        return x

class PaperHyperprior(nn.Module):
    """Lightweight hyperprior with depthwise separable convolutions"""
    def __init__(self, N=128, M=192):
        super().__init__()
        self.hyper_encoder = nn.Sequential(
            DepthwiseSeparableConv(N, N, 3, stride=1),
            DepthwiseSeparableConv(N, N, 5, stride=2, padding=2),
            DepthwiseSeparableConv(N, M, 5, stride=2, padding=2)
        )
        
        # Single frequency attention at bottleneck
        self.wam_hyper = PaperDWANAttention(M)
        
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(N),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(N),
            nn.Conv2d(N, N, 3, stride=1, padding=1)
        )

    def forward(self, x):
        z = self.hyper_encoder(x)
        # Apply WAM for spectral band-aware processing
        z_attended, _ = self.wam_hyper(z)
        hyper_params = self.hyper_decoder(z_attended)
        return z_attended, hyper_params

class PaperCompliantModel(nn.Module):
    """Complete lightweight model with depthwise separable + LN+PReLU"""
    def __init__(self, N=128, M=192):
        super().__init__()
        self.N = N
        self.M = M

        self.encoder = PaperEncoder(N=N)
        self.decoder = PaperDecoder(N=N)
        self.importance_map = PaperDWANAttention(N)
        self.hyperprior = PaperHyperprior(N=N, M=M)
        self.context_model = SSMContextModel(N=N)

    def forward(self, x):
        # Encoder
        y = self.encoder(x)
        y_before_importance = y.clone()

        # Importance mapping (existing DWAN)
        y_masked, importance = self.importance_map(y)
        y_before_quantization = y_masked.clone()

        # Quantization
        if self.training:
            quantization_noise = torch.empty_like(y_masked).uniform_(-0.5, 0.5)
            y_quantized = y_masked + quantization_noise
        else:
            y_quantized = torch.round(y_masked)

        # Hyperprior (enhanced)
        z, hyper_params = self.hyperprior(y_masked)
        if self.training:
            z_quantized = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        else:
            z_quantized = torch.round(z)

        # Context modeling (enhanced)
        ctx_mean, ctx_scale = self.context_model(y_quantized)

        # Decoder
        x_hat = self.decoder(y_quantized)

        return {
            'x_hat': x_hat,
            'y': y_quantized,
            'y_quantized': y_quantized,
            'z': z_quantized,
            'z_quantized': z_quantized,
            'hyper_params': hyper_params,
            'ctx_mean': ctx_mean,
            'ctx_scale': ctx_scale,
            'importance': importance,
            'y_before_quantization': y_before_quantization,
            'y_before_importance': y_before_importance
        }
