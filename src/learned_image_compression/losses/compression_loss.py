# src/losses/compression_loss.py - CORRECTED VERSION
import torch
import torch.nn as nn
import torch.nn.functional as F

class PaperCompliantLoss(nn.Module):
    def __init__(self, lambda_rd=0.01, lambda_pqf=0.1, distortion_metric='mse'):
        super().__init__()
        self.lambda_rd = lambda_rd
        self.lambda_pqf = lambda_pqf
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, outputs, target, iteration=0, epoch=0):
        x_hat = outputs['x_hat']
        y = outputs['y']
        z = outputs['z']
        ctx_mean = outputs['ctx_mean']
        ctx_scale = outputs['ctx_scale']
        
        device = y.device
        
        # 🔧 CRITICAL: Proper reconstruction loss
        distortion = F.mse_loss(x_hat, target, reduction='mean')
        
        # 🔧 CRITICAL: Proper entropy calculation (not simplified!)
        H_y_total, H_y_avg = self.calculate_entropy_y(y, ctx_mean, ctx_scale)
        H_z_total, H_z_avg = self.calculate_entropy_z(z)
        
        # 🔧 GENTLE sparsity encouragement (much smaller than before)
        sparsity_bonus = torch.mean((torch.abs(y) < 0.01).float()) * 0.05  # Very small bonus
        
        total_rate = H_y_avg + H_z_avg - sparsity_bonus
        
        # PQF loss
        pqf_loss = torch.tensor(0.0, device=device)
        if iteration <= 20000 and hasattr(self, 'lambda_pqf') and self.lambda_pqf > 0:
            y_before_quantization = outputs.get('y_before_quantization', None)
            if y_before_quantization is not None:
                pqf_loss = F.mse_loss(y, y_before_quantization, reduction='mean')
        
        # 🔧 CORRECTED: Prioritize reconstruction, gentle rate penalty
        total_loss = distortion + self.lambda_rd * total_rate + self.lambda_pqf * pqf_loss
        
        # 🔧 CORRECT BPP calculation using actual entropy
        H, W = target.shape[-2:]
        estimated_bpp = (H_y_total + H_z_total) / (H * W)
        
        # Monitoring
        zero_ratio = torch.mean((torch.abs(y) < 0.01).float())
        
        return {
            'total_loss': total_loss,
            'distortion': distortion,
            'rate': total_rate,
            'rate_y': H_y_avg,
            'rate_z': H_z_avg,
            'pqf_loss': pqf_loss,
            'estimated_bpp': estimated_bpp,
            'sparsity_ratio': zero_ratio
        }

    def calculate_entropy_y(self, y, ctx_mean, ctx_scale):
        """🔧 CORRECTED: Proper entropy calculation"""
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=y.device))
        
        y_probs = 0.5 * (
            torch.erf((y - ctx_mean + 0.5) / (ctx_scale * sqrt_2 + 1e-8)) -
            torch.erf((y - ctx_mean - 0.5) / (ctx_scale * sqrt_2 + 1e-8))
        )
        
        y_probs = torch.clamp(y_probs, min=1e-6, max=1.0)
        y_rate_bits = -torch.log2(y_probs)
        
        # 🔧 Conservative clipping (not too aggressive)
        y_rate_bits = torch.clamp(y_rate_bits, min=0.0, max=10.0)
        
        # Total entropy per image, averaged over batch
        H_y_total = torch.mean(torch.sum(y_rate_bits, dim=[1, 2, 3]))
        # Average bits per coefficient
        H_y_avg = torch.mean(y_rate_bits)
        
        return H_y_total, H_y_avg

    def calculate_entropy_z(self, z):
        """🔧 CORRECTED: Proper entropy calculation"""
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=z.device))
        
        z_probs = 0.5 * (
            torch.erf((z + 0.5) / sqrt_2) - 
            torch.erf((z - 0.5) / sqrt_2)
        )
        
        z_probs = torch.clamp(z_probs, min=1e-6, max=1.0)
        z_rate_bits = -torch.log2(z_probs)
        
        # 🔧 Conservative clipping
        z_rate_bits = torch.clamp(z_rate_bits, min=0.0, max=8.0)
        
        # Total entropy per image, averaged over batch
        H_z_total = torch.mean(torch.sum(z_rate_bits, dim=[1, 2, 3]))
        # Average bits per coefficient
        H_z_avg = torch.mean(z_rate_bits)
        
        return H_z_total, H_z_avg
