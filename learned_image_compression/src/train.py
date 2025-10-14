# src/fixed_train.py - FIXED Multi-Objective Training with Corrected SSIM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
import logging
import json
import time
from models.fixed_compression_model import PaperCompliantModel
from losses.compression_loss import PaperCompliantLoss

# 🔧 FIXED: PSNR and SSIM computation utilities
def compute_psnr(img1, img2, data_range=2.0):
    """
    Compute PSNR between two image tensors
    Args:
        img1, img2: torch tensors, shape BCHW, range [-1, 1]
        data_range: 2.0 for [-1,1] range
    """
    mse = torch.mean((img1 - img2) ** 2, dim=[1,2,3])
    mse = torch.clamp(mse, min=1e-8)  # Avoid division by zero
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return torch.mean(psnr).item()

def compute_ssim(img1, img2, window_size=11, data_range=2.0):
    """
    🔧 COMPLETELY FIXED: Compute SSIM between two image tensors
    Args:
        img1, img2: torch tensors, shape BCHW, range [-1, 1]
    """
    def gaussian_window(size, sigma=1.5):
        """Create a 2D Gaussian window for SSIM computation"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # 🔧 FIX: Create proper 2D gaussian kernel
        g2d = g[:, None] * g[None, :]  # Outer product to create 2D kernel
        
        # 🔧 FIX: Reshape to (1, 1, kernel_size, kernel_size) for conv2d
        return g2d.unsqueeze(0).unsqueeze(0)
    
    def ssim_single_channel(img1, img2):
        B, C, H, W = img1.shape
        window = gaussian_window(window_size).to(img1.device)
        
        # 🔧 FIX: All conv2d operations with proper parameters
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
        return torch.mean(ssim_map, dim=[2, 3])  # Average over H, W
    
    # Compute SSIM for each channel and average
    ssim_values = []
    for c in range(img1.shape[1]):
        ssim_c = ssim_single_channel(img1[:, c:c+1], img2[:, c:c+1])
        ssim_values.append(ssim_c)
    
    ssim_tensor = torch.stack(ssim_values, dim=1)  # B x C
    return torch.mean(ssim_tensor).item()

def compute_combined_score(bpp, psnr, ssim, weights=(1.0, 2.0, 1.0)):
    """
    Multi-objective scoring function
    Args:
        bpp: bits per pixel (lower is better, target: 0.0-1.0)
        psnr: peak signal-to-noise ratio (higher is better, target: 28-35+)  
        ssim: structural similarity (higher is better, target: 0.9-1.0)
        weights: (bpp_weight, psnr_weight, ssim_weight)
    
    Returns:
        Combined score (higher is better)
    """
    # Normalize metrics to 0-1 scale
    bpp_normalized = max(0, 1.0 - bpp)  # Invert BPP (lower is better)
    psnr_normalized = min(1.0, max(0, (psnr - 20) / 20))  # PSNR 20-40 -> 0-1
    ssim_normalized = max(0, ssim)  # SSIM already 0-1
    
    # Weighted combination
    score = (weights[0] * bpp_normalized + 
             weights[1] * psnr_normalized + 
             weights[2] * ssim_normalized) / sum(weights)
    
    return score

class PaperConfig:
    """Configuration with gradient accumulation for effective batch size"""
    
    NUM_EPOCHS = 150
    BATCH_SIZE = 8
    ACCUMULATION_STEPS = 1  # Set to 1 for local testing
    NUM_WORKERS = 8
    
    RATE_CONFIGS = {
        'mse': [
            {'lambda': 0.05, 'N': 128, 'M': 192},   # Current good results
            {'lambda': 0.075, 'N': 128, 'M': 192}, # Better compression
            {'lambda': 0.1, 'N': 128, 'M': 192},   # Target range
        ]
    }
    CURRENT_RATE_CONFIG = 0
    DISTORTION_METRIC = 'mse'
    
    BASE_LR = 1e-4
    DECAY_START_EPOCH = 75
    DECAY_EVERY_EPOCH = 10
    DECAY_FACTOR = 0.5
    
    PQF_ACTIVE_EPOCHS = 10
    LAMBDA_PQF_ACTIVE = 5.0
    LAMBDA_PQF_INACTIVE = 0.0
    
    IMAGE_RANGE = [-1, 1]
    PATCH_SIZE = 384
    
    DATASET_FOLDER = 'data/train'
    CHECKPOINT_DIR = 'checkpoints_paper'
    LOG_DIR = 'logs_paper'
    TEST_DIR = 'data/test'
    
    USE_MIXED_PRECISION = True
    
    # Multi-objective optimization settings
    SCORE_WEIGHTS = (1.0, 2.5, 1.5)  # (BPP, PSNR, SSIM) - prioritize PSNR slightly
    TARGET_BPP = 0.5   # Target BPP for good compression
    TARGET_PSNR = 30.0 # Target PSNR for good quality
    TARGET_SSIM = 0.90 # Target SSIM for good quality
    
    def __init__(self):
        self.create_directories()
        self.update_rate_config()
        
    def create_directories(self):
        directories = [self.CHECKPOINT_DIR, self.LOG_DIR, self.TEST_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Directory ready: {directory}")
    
    def update_rate_config(self):
        config = self.RATE_CONFIGS[self.DISTORTION_METRIC][self.CURRENT_RATE_CONFIG]
        self.N = config['N']
        self.M = config['M']
        self.LAMBDA_RD = config['lambda']
        
        print(f"📊 Multi-Objective Training Configuration:")
        print(f"   Lambda (λ): {self.LAMBDA_RD}")
        print(f"   N (channels): {self.N}")
        print(f"   M (hyperprior): {self.M}")
        print(f"   Distortion: {self.DISTORTION_METRIC.upper()}")
        print(f"🎯 Optimization Targets:")
        print(f"   Target BPP: ≤{self.TARGET_BPP}")
        print(f"   Target PSNR: ≥{self.TARGET_PSNR} dB")
        print(f"   Target SSIM: ≥{self.TARGET_SSIM}")
        print(f"🚀 Gradient Accumulation:")
        print(f"   Physical batch size: {self.BATCH_SIZE}")
        print(f"   Accumulation steps: {self.ACCUMULATION_STEPS}")
        print(f"   Effective batch size: {self.BATCH_SIZE * self.ACCUMULATION_STEPS}")

class PaperDataset(Dataset):
    """Dataset loader for large-scale training"""
    
    def __init__(self, folder_path, patch_size=384, image_range=[-1, 1]):
        self.folder_path = folder_path
        self.patch_size = patch_size
        self.image_range = image_range
        
        subfolders = ['clic', 'flickr', 'LIU4K']
        self.image_files = []
        
        print("📁 Loading dataset...")
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.exists(subfolder_path):
                png_files = glob.glob(os.path.join(subfolder_path, '*.png'))
                jpg_files = glob.glob(os.path.join(subfolder_path, '*.jpg'))
                jpeg_files = glob.glob(os.path.join(subfolder_path, '*.jpeg'))
                self.image_files.extend(png_files + jpg_files + jpeg_files)
                print(f"   📂 {subfolder}: {len(png_files + jpg_files + jpeg_files)} images")
            else:
                print(f"   ⚠️ {subfolder_path} not found, skipping...")
        
        print(f"📁 Total dataset: {len(self.image_files)} images")
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {folder_path}. Please check your dataset structure.")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            if min(image.size) > self.patch_size:
                left = np.random.randint(0, image.size[0] - self.patch_size + 1)
                top = np.random.randint(0, image.size[1] - self.patch_size + 1)
                image = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            else:
                image = image.resize((self.patch_size, self.patch_size), Image.LANCZOS)
            
            image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
            
            if self.image_range == [-1, 1]:
                image_tensor = image_tensor * 2.0 - 1.0
                
            image_tensor = image_tensor.permute(2, 0, 1)
            return image_tensor
            
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            if self.image_range == [-1, 1]:
                return torch.rand(3, self.patch_size, self.patch_size) * 2.0 - 1.0
            else:
                return torch.rand(3, self.patch_size, self.patch_size)

def get_paper_lr_lambda(total_epochs=150, decay_start=75, decay_every=10, decay_factor=0.5):
    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        else:
            decay_steps = (epoch - decay_start) // decay_every
            return decay_factor ** decay_steps
    return lr_lambda

def safe_get_value(x):
    if hasattr(x, 'item'):
        return x.item()
    elif isinstance(x, (int, float)):
        return float(x)
    else:
        return float(x)

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, config):
    """Training with gradient accumulation"""
    
    model.train()
    running_metrics = {
        'total_loss': 0.0,
        'distortion': 0.0,
        'rate': 0.0,
        'estimated_bpp': 0.0,
        'pqf_loss': 0.0,
        'sparsity_ratio': 0.0
    }
    
    num_batches = len(dataloader)
    current_iteration = epoch * num_batches
    
    # PQF activation logic
    if epoch <= config.PQF_ACTIVE_EPOCHS:
        criterion.lambda_pqf = config.LAMBDA_PQF_ACTIVE
        pqf_status = "ACTIVE"
    else:
        criterion.lambda_pqf = config.LAMBDA_PQF_INACTIVE
        pqf_status = "INACTIVE"
    
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(dataloader), total=num_batches,
                desc=f'Epoch {epoch:03d}/{config.NUM_EPOCHS} [PQF: {pqf_status}] [BS: {config.BATCH_SIZE}×{config.ACCUMULATION_STEPS}]')
    
    for batch_idx, data in pbar:
        data = data.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=config.USE_MIXED_PRECISION):
            outputs = model(data)
            loss_dict = criterion(outputs, data, iteration=current_iteration + batch_idx, epoch=epoch)
            loss = loss_dict['total_loss'] / config.ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0 or (batch_idx + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update metrics
        for key in running_metrics:
            if key in loss_dict:
                running_metrics[key] += safe_get_value(loss_dict[key])
        
        # Enhanced progress bar
        current_bpp = safe_get_value(loss_dict['estimated_bpp'])
        pqf_loss = safe_get_value(loss_dict['pqf_loss'])
        sparsity = safe_get_value(loss_dict.get('sparsity_ratio', 0.0))
        
        step_indicator = "📈" if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0 else "🔄"
        
        pbar.set_postfix({
            'Loss': f'{safe_get_value(loss_dict["total_loss"]):.4f}',
            'BPP': f'{current_bpp:.3f}',
            'PQF': f'{pqf_loss:.4f}' if pqf_loss > 0 else 'OFF',
            'Sparsity': f'{sparsity:.2f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'Step': step_indicator
        })
    
    # Average metrics
    for key in running_metrics:
        running_metrics[key] /= num_batches
    
    return running_metrics

def validate_model(model, dataloader, criterion, device, max_batches=50):
    """🔧 UPDATED: Validation with PSNR and SSIM computation"""
    
    model.eval()
    val_metrics = {
        'total_loss': 0.0,
        'distortion': 0.0,
        'estimated_bpp': 0.0,
        'psnr': 0.0,      # NEW
        'ssim': 0.0       # NEW
    }
    
    num_batches = 0
    
    with torch.no_grad():
        for data in dataloader:
            if num_batches >= max_batches:
                break
                
            data = data.to(device, non_blocking=True)
            outputs = model(data)
            loss_dict = criterion(outputs, data, iteration=0, epoch=0)
            
            # Compute PSNR and SSIM
            x_hat = outputs['x_hat']
            psnr_val = compute_psnr(data, x_hat)
            ssim_val = compute_ssim(data, x_hat)
            
            # Update metrics
            val_metrics['psnr'] += psnr_val
            val_metrics['ssim'] += ssim_val
            
            for key in ['total_loss', 'distortion', 'estimated_bpp']:
                if key in loss_dict:
                    val_metrics[key] += safe_get_value(loss_dict[key])
            
            num_batches += 1
    
    # Average metrics
    for key in val_metrics:
        val_metrics[key] /= max(num_batches, 1)
    
    return val_metrics

def save_models_dual(model, base_path, additional_data=None):
    """🔧 NEW: Save both model types as requested"""
    
    # 1. Save weights only (recommended for portability)
    weights_path = base_path.replace('.pth', '_weights.pth')
    weights_data = {
        'model_state_dict': model.state_dict(),
        'model_class': type(model).__name__,
        'model_config': {
            'N': model.N,
            'M': model.M
        }
    }
    if additional_data:
        weights_data.update(additional_data)
    
    torch.save(weights_data, weights_path)
    print(f"💾 Model weights saved: {weights_path}")
    
    # 2. Save full model for deployment (less portable)
    full_path = base_path.replace('.pth', '_full.pth')
    full_data = {
        'model': model,  # Complete model with architecture
        'model_class': type(model).__name__,
        'model_config': {
            'N': model.N,
            'M': model.M
        }
    }
    if additional_data:
        full_data.update(additional_data)
    
    torch.save(full_data, full_path)
    print(f"💾 Full model saved: {full_path}")
    
    return weights_path, full_path

def main():
    config = PaperConfig()
    
    # Enhanced logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.LOG_DIR, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f'🚀 GPU: {torch.cuda.get_device_name()}')
        logger.info(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        logger.info(f'   Mixed Precision: {config.USE_MIXED_PRECISION}')
    else:
        logger.info('🔧 Using CPU')
        config.USE_MIXED_PRECISION = False
    
    # Dataset loading
    try:
        dataset = PaperDataset(
            config.DATASET_FOLDER,
            patch_size=config.PATCH_SIZE,
            image_range=config.IMAGE_RANGE
        )
        
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_dataloader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=True, prefetch_factor=2
        )
        
        val_dataloader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True
        )
        
        logger.info(f'📊 Dataset Statistics:')
        logger.info(f'   Total images: {len(dataset):,}')
        logger.info(f'   Training: {len(train_dataset):,}')
        logger.info(f'   Validation: {len(val_dataset):,}')
        logger.info(f'   Batches per epoch: {len(train_dataloader):,}')
        logger.info(f'🚀 Effective batch size: {config.BATCH_SIZE * config.ACCUMULATION_STEPS}')
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return
    
    # Model setup
    try:
        model = PaperCompliantModel(N=config.N, M=config.M).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f'🏗️ Model Architecture:')
        logger.info(f'   Total parameters: {total_params:,}')
        logger.info(f'   Trainable parameters: {trainable_params:,}')
        logger.info(f'   N (channels): {config.N}')
        logger.info(f'   M (hyperprior): {config.M}')
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return
    
    # Training setup
    criterion = PaperCompliantLoss(
        lambda_rd=config.LAMBDA_RD,
        lambda_pqf=config.LAMBDA_PQF_ACTIVE,
        distortion_metric=config.DISTORTION_METRIC
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.BASE_LR,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )
    
    lr_lambda = get_paper_lr_lambda(
        total_epochs=config.NUM_EPOCHS,
        decay_start=config.DECAY_START_EPOCH,
        decay_every=config.DECAY_EVERY_EPOCH,
        decay_factor=config.DECAY_FACTOR
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    scaler = GradScaler(device='cuda', enabled=config.USE_MIXED_PRECISION)
    
    # Multi-objective tracking
    training_history = {
        'train_metrics': [],
        'val_metrics': [],
        'config': config.__dict__,
        'best_bpp': float('inf'),
        'best_psnr': 0.0,
        'best_ssim': 0.0,
        'best_combined_score': 0.0
    }
    
    logger.info('🎯 Multi-Objective Training Started:')
    logger.info(f'   Epochs: {config.NUM_EPOCHS}')
    logger.info(f'   Batch size: {config.BATCH_SIZE} (Effective: {config.BATCH_SIZE * config.ACCUMULATION_STEPS})')
    logger.info(f'   Lambda (λ): {config.LAMBDA_RD}')
    logger.info(f'   Learning rate: {config.BASE_LR:.2e}')
    logger.info(f'   Score weights (BPP, PSNR, SSIM): {config.SCORE_WEIGHTS}')
    
    best_combined_score = 0.0
    best_metrics = {'bpp': float('inf'), 'psnr': 0.0, 'ssim': 0.0}
    
    start_time = time.time()
    
    try:
        for epoch in range(1, config.NUM_EPOCHS + 1):
            epoch_start = time.time()
            
            train_metrics = train_one_epoch(
                model, train_dataloader, optimizer, criterion, scaler,
                device, epoch, config
            )
            
            val_metrics = validate_model(model, val_dataloader, criterion, device)
            
            scheduler.step()
            
            # Compute combined score
            current_bpp = val_metrics['estimated_bpp']
            current_psnr = val_metrics['psnr']
            current_ssim = val_metrics['ssim']
            
            combined_score = compute_combined_score(
                current_bpp, current_psnr, current_ssim, config.SCORE_WEIGHTS
            )
            
            # Update history
            training_history['train_metrics'].append(train_metrics)
            training_history['val_metrics'].append(val_metrics)
            
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            # 🔧 ENHANCED: Print all metrics clearly for progress tracking
            logger.info(f'Epoch {epoch:03d}/{config.NUM_EPOCHS} ({epoch_time:.1f}s):')
            logger.info(f'   📊 Training  - Loss: {train_metrics["total_loss"]:.6f} | BPP: {train_metrics["estimated_bpp"]:.4f}')
            logger.info(f'   📊 Validation - BPP: {current_bpp:.4f} | PSNR: {current_psnr:.2f} dB | SSIM: {current_ssim:.4f}')
            logger.info(f'   🎯 Combined Score: {combined_score:.4f} | LR: {current_lr:.2e}')
            
            # Multi-objective model selection
            save_model = False
            save_reason = ""
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_metrics = {'bpp': current_bpp, 'psnr': current_psnr, 'ssim': current_ssim}
                save_model = True
                save_reason = f"Best Combined Score: {combined_score:.4f}"
                
                # Update individual bests
                training_history['best_combined_score'] = combined_score
                if current_bpp < training_history['best_bpp']:
                    training_history['best_bpp'] = current_bpp
                if current_psnr > training_history['best_psnr']:
                    training_history['best_psnr'] = current_psnr
                if current_ssim > training_history['best_ssim']:
                    training_history['best_ssim'] = current_ssim
            
            if save_model:
                # Save both model types
                best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
                save_models_dual(model, best_model_path, {
                    'epoch': epoch,
                    'combined_score': combined_score,
                    'metrics': {
                        'bpp': current_bpp,
                        'psnr': current_psnr,
                        'ssim': current_ssim
                    },
                    'config': config.__dict__,
                    'training_history': training_history,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                })
                logger.info(f'💾 {save_reason} - Both model types saved!')
                logger.info(f'   📊 Best Metrics: BPP={best_metrics["bpp"]:.4f}, PSNR={best_metrics["psnr"]:.2f} dB, SSIM={best_metrics["ssim"]:.4f}')
            
            # Regular checkpoints
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch:03d}.pth')
                save_models_dual(model, checkpoint_path, {
                    'epoch': epoch,
                    'combined_score': combined_score,
                    'metrics': {
                        'bpp': current_bpp,
                        'psnr': current_psnr,
                        'ssim': current_ssim
                    },
                    'config': config.__dict__,
                    'training_history': training_history,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict()
                })
                logger.info(f'💾 Checkpoint saved: epoch {epoch} (both types)')
            
            # Status updates
            if epoch == config.PQF_ACTIVE_EPOCHS:
                logger.info(f'🎯 Epoch {epoch}: PQF regulation deactivated (λ₁ = 0)')
            if epoch == config.DECAY_START_EPOCH:
                logger.info(f'📉 Epoch {epoch}: Learning rate decay started')
    
    except KeyboardInterrupt:
        logger.info('⏹️ Training interrupted by user')
    except Exception as e:
        logger.error(f'❌ Training failed: {e}')
        raise
    
    # Final model save
    final_model_path = os.path.join(config.CHECKPOINT_DIR, f'final_model_lambda_{config.LAMBDA_RD}.pth')
    save_models_dual(model, final_model_path, {
        'config': config.__dict__,
        'training_complete': True,
        'best_combined_score': best_combined_score,
        'best_metrics': best_metrics,
        'total_epochs': config.NUM_EPOCHS,
        'training_history': training_history
    })
    
    # Save training history
    history_path = os.path.join(config.LOG_DIR, f'training_history_lambda_{config.LAMBDA_RD}.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Final statistics
    total_time = time.time() - start_time
    final_train_bpp = training_history['train_metrics'][-1]['estimated_bpp']
    final_val_metrics = training_history['val_metrics'][-1]
    
    logger.info('🏁 Multi-Objective Training Complete!')
    logger.info(f'   Total time: {total_time/3600:.2f} hours')
    logger.info(f'   Best Combined Score: {best_combined_score:.4f}')
    logger.info(f'🏆 Best Metrics Achieved:')
    logger.info(f'   📊 BPP: {best_metrics["bpp"]:.4f} (Target: ≤{config.TARGET_BPP})')
    logger.info(f'   📊 PSNR: {best_metrics["psnr"]:.2f} dB (Target: ≥{config.TARGET_PSNR} dB)')
    logger.info(f'   📊 SSIM: {best_metrics["ssim"]:.4f} (Target: ≥{config.TARGET_SSIM})')
    logger.info(f'   Final Train BPP: {final_train_bpp:.4f}')
    logger.info(f'   Final Val BPP: {final_val_metrics["estimated_bpp"]:.4f}')
    
    # Multi-objective success assessment
    success_bpp = best_metrics["bpp"] <= config.TARGET_BPP
    success_psnr = best_metrics["psnr"] >= config.TARGET_PSNR
    success_ssim = best_metrics["ssim"] >= config.TARGET_SSIM
    
    if success_bpp and success_psnr and success_ssim:
        logger.info('🎉 SUCCESS: All targets achieved! Excellent compression with high quality!')
    elif success_psnr and success_ssim:
        logger.info('✅ GOOD: Quality targets achieved! Consider improving compression efficiency.')
    elif success_bpp:
        logger.info('⚠️ MODERATE: Compression target achieved! Consider improving image quality.')
    else:
        logger.info('🔧 NEEDS IMPROVEMENT: Consider adjusting lambda or training longer.')

if __name__ == '__main__':
    main()
