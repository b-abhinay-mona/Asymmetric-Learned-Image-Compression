# src/test.py - COMPLETE ARCHITECTURE HYBRID FIX

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import glob
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# 🔧 ARCHITECTURE HYBRID: Define BOTH old and new architectures

# Import wavelet layers first
from models.wavelet_layers import DWT_2D, IDWT_2D

# ==================== LIGHTWEIGHT ARCHITECTURE (What your trained model uses) ====================

class DepthwiseSeparableConv(nn.Module):
    """Efficient depthwise separable convolution"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, 
                                   padding, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.prelu1 = nn.PReLU(in_ch)
        
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

class LightweightNorm(nn.Module):
    """Lightweight normalization: LN + PReLU replaces GDN/IGDN"""
    def __init__(self, channels, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.norm = nn.LayerNorm([channels], elementwise_affine=True)
        self.activation = nn.PReLU(channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # Back to B, C, H, W
        x = self.activation(x)
        return x

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv)"""
    def __init__(self, in_channels, out_channels, expansion=4, kernel_size=3, stride=1):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim)
        ) if expansion > 1 else nn.Identity()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim)
        )
        
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

class LightweightPaperDWANAttention(nn.Module):
    """Lightweight DWAN with depthwise separable convolutions"""
    def __init__(self, in_channels):
        super().__init__()
        self.dwt = DWT_2D(wavename='haar')
        self.importance_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels * 4, in_channels // 2, 3),
            DepthwiseSeparableConv(in_channels // 2, in_channels // 4, 3),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        LL, LH, HL, HH = self.dwt(x)
        wavelet_features = torch.cat([LL, LH, HL, HH], dim=1)
        importance = self.importance_conv(wavelet_features)
        importance = F.interpolate(importance, size=(H, W), mode='bilinear', align_corners=False)
        importance = 0.3 + 0.7 * importance
        masked_x = x * importance
        return masked_x, importance

class LightweightPostQuantizationFilter(nn.Module):
    """Lightweight PQF with depthwise separable convolutions"""
    def __init__(self, N=128):
        super().__init__()
        self.pqf = nn.Sequential(
            DepthwiseSeparableConv(N, N//2, 3),
            DepthwiseSeparableConv(N//2, N//2, 3),
            nn.Conv2d(N//2, N, 1),
            nn.Tanh()
        )
        self.wam_pqf = LightweightPaperDWANAttention(N)

    def forward(self, x):
        filtered = x + 0.1 * self.pqf(x)
        refined, _ = self.wam_pqf(filtered)
        return refined

class LightweightSSMContextModel(nn.Module):
    """Lightweight context model with depthwise separable convs"""
    def __init__(self, N=128):
        super().__init__()
        self.context_prediction = nn.Sequential(
            DepthwiseSeparableConv(N, N, 3),
            DepthwiseSeparableConv(N, N, 3)
        )
        
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(N, N//4, 1),
            nn.PReLU(N//4),
            nn.Conv2d(N//4, N, 1),
            nn.Sigmoid()
        )
        
        self.output_layer = nn.Conv2d(N, N*2, 1)

    def forward(self, x):
        local_ctx = self.context_prediction(x)
        global_ctx = self.global_context(x)
        fused_ctx = local_ctx * global_ctx
        ctx = self.output_layer(fused_ctx)
        mean, scale = torch.chunk(ctx, 2, dim=1)
        scale = F.softplus(scale) + 0.1
        return mean, scale

class LightweightPaperEncoder(nn.Module):
    """Lightweight encoder with MBConv and depthwise separable"""
    def __init__(self, N=128):
        super().__init__()
        self.N = N
        self.conv1 = nn.Conv2d(3, N, 5, stride=2, padding=2)
        self.norm1 = LightweightNorm(N)  # Instead of gdn1
        self.hierarchical_cnn1 = MBConvBlock(N, N, expansion=4, kernel_size=3)
        self.conv2 = DepthwiseSeparableConv(N, N, kernel_size=5, stride=2, padding=2)
        self.norm2 = LightweightNorm(N)  # Instead of gdn2
        self.hierarchical_cnn2 = MBConvBlock(N, N, expansion=2, kernel_size=3)
        self.conv3 = DepthwiseSeparableConv(N, N, kernel_size=5, stride=2, padding=2)
        self.norm3 = LightweightNorm(N)  # Instead of gdn3
        self.hierarchical_cnn3 = MBConvBlock(N, N, expansion=4, kernel_size=3)
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

class LightweightPaperDecoder(nn.Module):
    """Lightweight asymmetric decoder - single lightweight block"""
    def __init__(self, N=128):
        super().__init__()
        self.pqf = LightweightPostQuantizationFilter(N)
        self.conv1 = nn.Conv2d(N, N, 3, padding=1)
        self.hierarchical_cnn = MBConvBlock(N, N, expansion=2, kernel_size=3)
        self.deconv1 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.norm1 = LightweightNorm(N)  # Instead of igdn1
        self.deconv2 = nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1)
        self.norm2 = LightweightNorm(N)  # Instead of igdn2
        self.deconv3 = nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x_filtered = self.pqf(x)
        x = self.conv1(x_filtered)
        x = self.hierarchical_cnn(x)
        x = self.norm1(self.deconv1(x))
        x = self.norm2(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        return x

class LightweightPaperHyperprior(nn.Module):
    """Lightweight hyperprior with depthwise separable convolutions"""
    def __init__(self, N=128, M=192):
        super().__init__()
        self.hyper_encoder = nn.Sequential(
            DepthwiseSeparableConv(N, N, 3, stride=1),
            DepthwiseSeparableConv(N, N, 5, stride=2, padding=2),
            DepthwiseSeparableConv(N, M, 5, stride=2, padding=2)
        )
        self.wam_hyper = LightweightPaperDWANAttention(M)
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(N),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            nn.PReLU(N),
            nn.Conv2d(N, N, 3, stride=1, padding=1)
        )

    def forward(self, x):
        z = self.hyper_encoder(x)
        z_attended, _ = self.wam_hyper(z)
        hyper_params = self.hyper_decoder(z_attended)
        return z_attended, hyper_params

class LightweightPaperCompliantModel(nn.Module):
    """Complete lightweight model with depthwise separable + LN+PReLU"""
    def __init__(self, N=128, M=192):
        super().__init__()
        self.N = N
        self.M = M
        self.encoder = LightweightPaperEncoder(N=N)
        self.decoder = LightweightPaperDecoder(N=N)
        self.importance_map = LightweightPaperDWANAttention(N)
        self.hyperprior = LightweightPaperHyperprior(N=N, M=M)
        self.context_model = LightweightSSMContextModel(N=N)

    def forward(self, x):
        # Encoder
        y = self.encoder(x)
        y_before_importance = y.clone()

        # Importance mapping
        y_masked, importance = self.importance_map(y)
        y_before_quantization = y_masked.clone()

        # Quantization
        if self.training:
            quantization_noise = torch.empty_like(y_masked).uniform_(-0.5, 0.5)
            y_quantized = y_masked + quantization_noise
        else:
            y_quantized = torch.round(y_masked)

        # Hyperprior
        z, hyper_params = self.hyperprior(y_masked)
        if self.training:
            z_quantized = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        else:
            z_quantized = torch.round(z)

        # Context modeling
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

# ==================== MONKEY PATCH INJECTION ====================

print("🔧 Injecting lightweight architecture into modules...")

# Import modules
import models.fixed_compression_model as fcm
import models.hierarchical_cnn as hcnn

# Inject lightweight classes
fcm.LightweightNorm = LightweightNorm
fcm.DepthwiseSeparableConv = DepthwiseSeparableConv
fcm.MBConvBlock = MBConvBlock
fcm.PaperDWANAttention = LightweightPaperDWANAttention  # Replace with lightweight version
fcm.PostQuantizationFilter = LightweightPostQuantizationFilter
fcm.SSMContextModel = LightweightSSMContextModel
fcm.PaperEncoder = LightweightPaperEncoder  # Replace with lightweight version
fcm.PaperDecoder = LightweightPaperDecoder
fcm.PaperHyperprior = LightweightPaperHyperprior
fcm.PaperCompliantModel = LightweightPaperCompliantModel

hcnn.MBConvBlock = MBConvBlock

# Update sys.modules
sys.modules['models.fixed_compression_model'].LightweightNorm = LightweightNorm
sys.modules['models.fixed_compression_model'].DepthwiseSeparableConv = DepthwiseSeparableConv
sys.modules['models.fixed_compression_model'].MBConvBlock = MBConvBlock
sys.modules['models.fixed_compression_model'].PaperDWANAttention = LightweightPaperDWANAttention
sys.modules['models.fixed_compression_model'].PostQuantizationFilter = LightweightPostQuantizationFilter
sys.modules['models.fixed_compression_model'].SSMContextModel = LightweightSSMContextModel
sys.modules['models.fixed_compression_model'].PaperEncoder = LightweightPaperEncoder
sys.modules['models.fixed_compression_model'].PaperDecoder = LightweightPaperDecoder
sys.modules['models.fixed_compression_model'].PaperHyperprior = LightweightPaperHyperprior
sys.modules['models.fixed_compression_model'].PaperCompliantModel = LightweightPaperCompliantModel
sys.modules['models.hierarchical_cnn'].MBConvBlock = MBConvBlock

print("✅ Lightweight architecture injection complete")

class MultiObjectiveCompressionTester:
    """🎯 Multi-objective testing suite with complete architecture compatibility"""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model_hybrid(model_path)
        self.results = []

        # Target metrics from training
        self.TARGET_BPP = 0.5
        self.TARGET_PSNR = 30.0
        self.TARGET_SSIM = 0.90
        self.SCORE_WEIGHTS = (1.0, 2.5, 1.5)

    def load_model_hybrid(self, model_path):
        """🔧 HYBRID ARCHITECTURE COMPATIBLE model loading"""
        print(f'🔄 Loading trained model from {model_path}')
        print(f'🔧 Using hybrid architecture compatibility')
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                    print(f'✅ Model loaded successfully using hybrid compatibility')
                    
                    # Print training metrics if available
                    if 'metrics' in checkpoint:
                        metrics = checkpoint['metrics']
                        print(f'📊 Training Results:')
                        print(f'   BPP: {metrics["bpp"]:.4f} | PSNR: {metrics["psnr"]:.2f} dB | SSIM: {metrics["ssim"]:.4f}')
                        
                    if 'combined_score' in checkpoint:
                        print(f'🎯 Training Combined Score: {checkpoint["combined_score"]:.4f}')
                        
                    model = model.to(self.device)
                    model.eval()
                    
                    total_params = sum(p.numel() for p in model.parameters())
                    print(f'📊 Model parameters: {total_params:,}')
                    
                    return model
                    
                elif 'model_state_dict' in checkpoint:
                    # Create lightweight model and load state dict
                    model = LightweightPaperCompliantModel(N=128, M=192)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f'✅ Model created from state_dict using hybrid compatibility')
                    return model.to(self.device)
                    
            else:
                # Direct model
                model = checkpoint.to(self.device)
                model.eval()
                print(f'✅ Direct model loaded using hybrid compatibility')
                return model

        except Exception as e:
            print(f'❌ Hybrid loading failed: {e}')
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Could not load trained model with hybrid compatibility: {e}")

    def compute_psnr(self, img1, img2, data_range=2.0):
        """Compute PSNR - same method as training"""
        mse = torch.mean((img1 - img2) ** 2, dim=[1,2,3])
        mse = torch.clamp(mse, min=1e-8)
        psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
        return torch.mean(psnr).item()

    def compute_ssim(self, img1, img2, window_size=11, data_range=2.0):
        """Compute SSIM - same method as training"""
        def gaussian_window(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32)
            coords -= size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            g2d = g[:, None] * g[None, :]
            return g2d.unsqueeze(0).unsqueeze(0)

        def ssim_single_channel(img1, img2):
            B, C, H, W = img1.shape
            window = gaussian_window(window_size).to(img1.device)

            mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2)
            mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2)

            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = torch.nn.functional.conv2d(img1**2, window, padding=window_size//2) - mu1_sq
            sigma2_sq = torch.nn.functional.conv2d(img2**2, window, padding=window_size//2) - mu2_sq
            sigma12 = torch.nn.functional.conv2d(img1*img2, window, padding=window_size//2) - mu1_mu2

            C1 = (0.01 * data_range) ** 2
            C2 = (0.03 * data_range) ** 2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return torch.mean(ssim_map, dim=[2, 3])

        ssim_values = []
        for c in range(img1.shape[1]):
            ssim_c = ssim_single_channel(img1[:, c:c+1], img2[:, c:c+1])
            ssim_values.append(ssim_c)
        ssim_tensor = torch.stack(ssim_values, dim=1)
        return torch.mean(ssim_tensor).item()

    def compute_combined_score(self, bpp, psnr, ssim, weights=None):
        """Compute combined score - same method as training"""
        if weights is None:
            weights = self.SCORE_WEIGHTS

        bpp_normalized = max(0, 1.0 - bpp)
        psnr_normalized = min(1.0, max(0, (psnr - 20) / 20))
        ssim_normalized = max(0, ssim)

        score = (weights[0] * bpp_normalized +
                weights[1] * psnr_normalized +
                weights[2] * ssim_normalized) / sum(weights)
        return score

    def preprocess_image(self, image_path, target_size=(384, 384)):
        """Load and preprocess image matching training format"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        image = image.resize(target_size, Image.LANCZOS)
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor * 2.0 - 1.0  # [-1, 1] range
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device), original_size

    def postprocess_image(self, image_tensor):
        """Convert model output back to PIL Image"""
        image_tensor = torch.clamp(image_tensor, -1, 1)
        image_tensor = (image_tensor + 1.0) / 2.0
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np)

    def calculate_bpp(self, outputs, image_size):
        """Enhanced BPP calculation"""
        y = outputs['y']
        z = outputs['z']
        ctx_mean = outputs.get('ctx_mean', torch.zeros_like(y))
        ctx_scale = outputs.get('ctx_scale', torch.ones_like(y))
        
        H, W = image_size
        total_pixels = H * W

        with torch.no_grad():
            # Y entropy using context model
            y_probs = 0.5 * (
                torch.erf((y - ctx_mean + 0.5) / (ctx_scale * torch.sqrt(torch.tensor(2.0)) + 1e-8)) -
                torch.erf((y - ctx_mean - 0.5) / (ctx_scale * torch.sqrt(torch.tensor(2.0)) + 1e-8))
            )
            y_rate = -torch.sum(torch.log2(torch.clamp(y_probs, min=1e-10)))

            # Z entropy using hyperprior
            z_probs = 0.5 * (
                torch.erf((z + 0.5) / torch.sqrt(torch.tensor(2.0))) -
                torch.erf((z - 0.5) / torch.sqrt(torch.tensor(2.0)))
            )
            z_rate = -torch.sum(torch.log2(torch.clamp(z_probs, min=1e-10)))

            total_rate = y_rate + z_rate
            bpp = total_rate / total_pixels

        return bpp.item()

    def test_single_image(self, image_path, save_results=True, output_dir='test_results'):
        """🎯 Test single image"""
        print(f'🧪 Testing: {os.path.basename(image_path)}')
        start_time = time.time()

        try:
            # Load and process image
            image_tensor, original_size = self.preprocess_image(image_path)
            original_image = Image.open(image_path).convert('RGB').resize((384, 384), Image.LANCZOS)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                x_hat = outputs['x_hat']
                reconstructed_image = self.postprocess_image(x_hat)

            processing_time = time.time() - start_time

            # Calculate metrics
            bpp = self.calculate_bpp(outputs, image_tensor.shape[-2:])
            psnr_value = self.compute_psnr(image_tensor, x_hat)
            ssim_value = self.compute_ssim(image_tensor, x_hat)
            combined_score = self.compute_combined_score(bpp, psnr_value, ssim_value)

            # File size metrics
            original_size_bytes = os.path.getsize(image_path)
            estimated_compressed_size = (bpp * 384 * 384) / 8
            compression_ratio = original_size_bytes / max(estimated_compressed_size, 1)

            # Target achievement
            target_achievement = {
                'bpp_target': bpp <= self.TARGET_BPP,
                'psnr_target': psnr_value >= self.TARGET_PSNR,
                'ssim_target': ssim_value >= self.TARGET_SSIM
            }

            result = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'original_size': original_size,
                'processed_size': (384, 384),
                'bpp': bpp,
                'psnr': psnr_value,
                'ssim': ssim_value,
                'combined_score': combined_score,
                'compression_ratio': compression_ratio,
                'processing_time': processing_time,
                'original_file_size_kb': original_size_bytes / 1024,
                'estimated_compressed_size_kb': estimated_compressed_size / 1024,
                'target_achievement': target_achievement,
                'targets': {
                    'bpp': self.TARGET_BPP,
                    'psnr': self.TARGET_PSNR,
                    'ssim': self.TARGET_SSIM
                }
            }

            self.results.append(result)

            if save_results:
                self.save_single_result(original_image, reconstructed_image, result, output_dir)

            # Progress reporting
            print(f'   📊 BPP: {bpp:.4f} (Target: ≤{self.TARGET_BPP}) {"✅" if target_achievement["bpp_target"] else "❌"}')
            print(f'   📊 PSNR: {psnr_value:.2f} dB (Target: ≥{self.TARGET_PSNR}) {"✅" if target_achievement["psnr_target"] else "❌"}')
            print(f'   📊 SSIM: {ssim_value:.4f} (Target: ≥{self.TARGET_SSIM}) {"✅" if target_achievement["ssim_target"] else "❌"}')
            print(f'   🎯 Combined Score: {combined_score:.4f}')
            print(f'   📊 Compression: {compression_ratio:.2f}x | Time: {processing_time:.3f}s')

            return result

        except Exception as e:
            print(f'❌ Error processing {image_path}: {e}')
            import traceback
            traceback.print_exc()
            return None

    def save_single_result(self, original, reconstructed, result, output_dir):
        """Save comparison results"""
        os.makedirs(output_dir, exist_ok=True)
        image_name = Path(result['image_name']).stem

        original.save(os.path.join(output_dir, f"{image_name}_original.png"))
        reconstructed.save(os.path.join(output_dir, f"{image_name}_reconstructed.png"))

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        axes[0].imshow(original)
        axes[0].set_title(f'Original\n{result["original_size"]} → 384×384')
        axes[0].axis('off')

        bpp_status = "✅" if result['target_achievement']['bpp_target'] else "❌"
        psnr_status = "✅" if result['target_achievement']['psnr_target'] else "❌"
        ssim_status = "✅" if result['target_achievement']['ssim_target'] else "❌"

        title = f'Reconstructed\n'
        title += f'BPP: {result["bpp"]:.4f} {bpp_status} | PSNR: {result["psnr"]:.2f} dB {psnr_status}\n'
        title += f'SSIM: {result["ssim"]:.4f} {ssim_status} | Score: {result["combined_score"]:.4f}'

        axes[1].imshow(reconstructed)
        axes[1].set_title(title)
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{image_name}_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def test_directory(self, test_dir, output_dir='test_results', max_images=None):
        """Test directory of images"""
        print(f'🔍 Multi-Objective Testing: {test_dir}')
        print(f'🎯 Targets: BPP≤{self.TARGET_BPP}, PSNR≥{self.TARGET_PSNR}, SSIM≥{self.TARGET_SSIM}')
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(test_dir, '**', ext), recursive=True))
        
        if max_images:
            image_files = image_files[:max_images]
            
        print(f'📁 Found {len(image_files)} images to test')

        for i, image_path in enumerate(image_files, 1):
            print(f'\n[{i}/{len(image_files)}]', end=' ')
            try:
                result = self.test_single_image(image_path, save_results=True, output_dir=output_dir)
                if result is None:
                    print(f'❌ Skipped {image_path}')
                    continue
            except Exception as e:
                print(f'❌ Error processing {image_path}: {e}')
                continue

        self.generate_report(output_dir)

    def generate_report(self, output_dir):
        """Generate comprehensive report"""
        if not self.results:
            print('❌ No results to report')
            return

        print(f'\n📊 Generating test report...')

        # Extract metrics
        bpps = [r['bpp'] for r in self.results]
        psnrs = [r['psnr'] for r in self.results]
        ssims = [r['ssim'] for r in self.results]
        scores = [r['combined_score'] for r in self.results]
        compression_ratios = [r['compression_ratio'] for r in self.results]
        processing_times = [r['processing_time'] for r in self.results]

        # Target achievements
        bpp_achieved = sum(1 for r in self.results if r['target_achievement']['bpp_target'])
        psnr_achieved = sum(1 for r in self.results if r['target_achievement']['psnr_target'])
        ssim_achieved = sum(1 for r in self.results if r['target_achievement']['ssim_target'])
        all_targets_achieved = sum(1 for r in self.results if all(r['target_achievement'].values()))

        summary = {
            'total_images': len(self.results),
            'image_size': '384×384 (training format)',
            'computation_method': 'Hybrid architecture compatible with training',
            # Average metrics
            'average_bpp': float(np.mean(bpps)),
            'average_psnr': float(np.mean(psnrs)),
            'average_ssim': float(np.mean(ssims)),
            'average_combined_score': float(np.mean(scores)),
            'average_compression_ratio': float(np.mean(compression_ratios)),
            'average_processing_time': float(np.mean(processing_times)),
            # Standard deviations
            'bpp_std': float(np.std(bpps)),
            'psnr_std': float(np.std(psnrs)),
            'ssim_std': float(np.std(ssims)),
            'score_std': float(np.std(scores)),
            # Ranges
            'bpp_range': [float(np.min(bpps)), float(np.max(bpps))],
            'psnr_range': [float(np.min(psnrs)), float(np.max(psnrs))],
            'ssim_range': [float(np.min(ssims)), float(np.max(ssims))],
            # Target achievements
            'targets': {
                'bpp': self.TARGET_BPP,
                'psnr': self.TARGET_PSNR,
                'ssim': self.TARGET_SSIM
            },
            'target_achievement': {
                'bpp_achieved': bpp_achieved,
                'psnr_achieved': psnr_achieved,
                'ssim_achieved': ssim_achieved,
                'all_targets_achieved': all_targets_achieved,
                'bpp_achievement_rate': bpp_achieved / len(self.results),
                'psnr_achievement_rate': psnr_achieved / len(self.results),
                'ssim_achievement_rate': ssim_achieved / len(self.results),
                'all_targets_achievement_rate': all_targets_achieved / len(self.results)
            },
            'score_weights': self.SCORE_WEIGHTS
        }

        # Save report
        report_data = {
            'summary': summary,
            'detailed_results': self.results
        }

        report_path = os.path.join(output_dir, 'hybrid_test_report.json')
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f'✅ Report saved: {report_path}')
        except Exception as e:
            print(f'⚠️ JSON save error: {e}')
            import pickle
            pickle_path = report_path.replace('.json', '.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(report_data, f)
            print(f'💾 Saved as pickle: {pickle_path}')

        self.create_summary_plots(output_dir)

        # FINAL REPORT
        print(f'\n🎯 MULTI-OBJECTIVE TEST RESULTS:')
        print(f'=' * 70)
        print(f'📊 Images Tested: {summary["total_images"]}')
        print(f'🖼️ Image Size: All resized to 384×384')
        print(f'🔧 Architecture: Hybrid Compatible')
        print(f'')
        print(f'📊 AVERAGE METRICS:')
        print(f'   BPP: {summary["average_bpp"]:.4f} ± {summary["bpp_std"]:.4f}')
        print(f'   PSNR: {summary["average_psnr"]:.2f} ± {summary["psnr_std"]:.2f} dB')
        print(f'   SSIM: {summary["average_ssim"]:.4f} ± {summary["ssim_std"]:.4f}')
        print(f'   Combined Score: {summary["average_combined_score"]:.4f} ± {summary["score_std"]:.4f}')
        print(f'   Compression: {summary["average_compression_ratio"]:.2f}x')
        print(f'   Processing Time: {summary["average_processing_time"]:.3f}s')
        print(f'')
        print(f'🎯 TARGET ACHIEVEMENT:')
        print(f'   BPP ≤ {self.TARGET_BPP}: {bpp_achieved}/{len(self.results)} ({100*bpp_achieved/len(self.results):.1f}%)')
        print(f'   PSNR ≥ {self.TARGET_PSNR}: {psnr_achieved}/{len(self.results)} ({100*psnr_achieved/len(self.results):.1f}%)')
        print(f'   SSIM ≥ {self.TARGET_SSIM}: {ssim_achieved}/{len(self.results)} ({100*ssim_achieved/len(self.results):.1f}%)')
        print(f'   All Targets: {all_targets_achieved}/{len(self.results)} ({100*all_targets_achieved/len(self.results):.1f}%)')
        print(f'=' * 70)

        # Success assessment
        if all_targets_achieved / len(self.results) >= 0.8:
            print('🎉 EXCELLENT: 80%+ images achieve all targets!')
        elif all_targets_achieved / len(self.results) >= 0.5:
            print('✅ GOOD: 50%+ images achieve all targets!')
        elif (psnr_achieved + ssim_achieved) / (2 * len(self.results)) >= 0.7:
            print('⚠️ MODERATE: Good quality, but compression can improve')
        else:
            print('🔧 NEEDS IMPROVEMENT: Consider retraining with different parameters')

        return summary

    def create_summary_plots(self, output_dir):
        """Create summary plots with target lines"""
        if not self.results:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        bpps = [r['bpp'] for r in self.results]
        psnrs = [r['psnr'] for r in self.results]
        ssims = [r['ssim'] for r in self.results]
        scores = [r['combined_score'] for r in self.results]

        # BPP distribution
        axes[0, 0].hist(bpps, bins=20, alpha=0.7, color='blue', label='BPP Values')
        axes[0, 0].axvline(self.TARGET_BPP, color='red', linestyle='--', linewidth=2, label=f'Target: ≤{self.TARGET_BPP}')
        axes[0, 0].axvline(np.mean(bpps), color='green', linestyle=':', label=f'Mean: {np.mean(bpps):.4f}')
        axes[0, 0].set_title('BPP Distribution')
        axes[0, 0].set_xlabel('Bits per Pixel')
        axes[0, 0].legend()

        # PSNR distribution
        axes[0, 1].hist(psnrs, bins=20, alpha=0.7, color='green', label='PSNR Values')
        axes[0, 1].axvline(self.TARGET_PSNR, color='red', linestyle='--', linewidth=2, label=f'Target: ≥{self.TARGET_PSNR}')
        axes[0, 1].axvline(np.mean(psnrs), color='blue', linestyle=':', label=f'Mean: {np.mean(psnrs):.2f}')
        axes[0, 1].set_title('PSNR Distribution')
        axes[0, 1].set_xlabel('PSNR (dB)')
        axes[0, 1].legend()

        # SSIM distribution
        axes[0, 2].hist(ssims, bins=20, alpha=0.7, color='orange', label='SSIM Values')
        axes[0, 2].axvline(self.TARGET_SSIM, color='red', linestyle='--', linewidth=2, label=f'Target: ≥{self.TARGET_SSIM}')
        axes[0, 2].axvline(np.mean(ssims), color='blue', linestyle=':', label=f'Mean: {np.mean(ssims):.4f}')
        axes[0, 2].set_title('SSIM Distribution')
        axes[0, 2].set_xlabel('SSIM')
        axes[0, 2].legend()

        # Combined score distribution
        axes[1, 0].hist(scores, bins=20, alpha=0.7, color='purple')
        axes[1, 0].axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.4f}')
        axes[1, 0].set_title('Combined Score Distribution')
        axes[1, 0].set_xlabel('Combined Score')
        axes[1, 0].legend()

        # Rate-Distortion curve
        axes[1, 1].scatter(bpps, psnrs, alpha=0.6, color='purple', s=20)
        axes[1, 1].axvline(self.TARGET_BPP, color='red', linestyle='--', alpha=0.7, label=f'Target BPP: {self.TARGET_BPP}')
        axes[1, 1].axhline(self.TARGET_PSNR, color='red', linestyle='--', alpha=0.7, label=f'Target PSNR: {self.TARGET_PSNR}')
        axes[1, 1].set_title('Rate-Distortion Curve')
        axes[1, 1].set_xlabel('BPP')
        axes[1, 1].set_ylabel('PSNR (dB)')
        axes[1, 1].legend()

        # Target achievement pie chart
        achieved = sum(1 for r in self.results if all(r['target_achievement'].values()))
        not_achieved = len(self.results) - achieved
        labels = [f'All Targets\nAchieved\n({achieved})', f'Some Targets\nMissed\n({not_achieved})']
        sizes = [achieved, not_achieved]
        colors = ['lightgreen', 'lightcoral']
        
        axes[1, 2].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Target Achievement')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hybrid_summary_plots.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'📊 Summary plots saved to: {output_dir}/hybrid_summary_plots.png')


def main():
    """🔧 HYBRID COMPATIBLE testing main function"""
    # ✏️ EDIT THESE PATHS:
    model_path = 'checkpoints_paper/best_model_full.pth'  # Your trained model
    test_image_or_dir = 'data/test/'  # Test directory
    output_dir = 'hybrid_test_results'
    max_images = None  # Test all images

    print('🎯 HYBRID COMPATIBLE Multi-Objective Testing')
    print('=' * 70)
    print(f'📂 Model: {model_path}')
    print(f'📂 Test Data: {test_image_or_dir}')
    print(f'📂 Output: {output_dir}')
    print(f'🎯 Targets: BPP≤0.5, PSNR≥30dB, SSIM≥0.9')
    print(f'🔧 Method: Hybrid Architecture Compatibility (Complete Fix)')
    print('=' * 70)

    try:
        tester = MultiObjectiveCompressionTester(model_path)

        if os.path.isfile(test_image_or_dir):
            print('🧪 Testing single image...')
            tester.test_single_image(test_image_or_dir, save_results=True, output_dir=output_dir)
            tester.generate_report(output_dir)
        elif os.path.isdir(test_image_or_dir):
            print('🧪 Testing directory...')
            tester.test_directory(test_image_or_dir, output_dir, max_images)
        else:
            print(f'❌ Path not found: {test_image_or_dir}')
            return

        print(f'\n✅ Hybrid compatible testing complete!')
        print(f'📁 Results saved in: {output_dir}')

    except Exception as e:
        print(f'❌ Testing failed: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
