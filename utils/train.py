import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
import gc
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.dit_model import DiTModel
from utils.diffusion_utils import NoiseScheduler, DiffusionLoss, EMAModel
from utils.dataset import create_data_loaders, get_data_stats

class Trainer:
    def __init__(self, config):
        self.config = config
        
        if torch.cuda.is_available():
            if hasattr(config, 'gpu_id') and config.gpu_id is not None:
                if config.gpu_id < torch.cuda.device_count():
                    self.device = torch.device(f'cuda:{config.gpu_id}')
                    print(f"Using specified GPU: cuda:{config.gpu_id}")
                else:
                    print(f"Warning: GPU {config.gpu_id} does not exist, total GPUs: {torch.cuda.device_count()}")
                    self.device = torch.device('cuda:0')
                    print("Fallback to default GPU: cuda:0")
            else:
                self.device = torch.device('cuda')
                print("Using default GPU: cuda")
        else:
            self.device = torch.device('cpu')
            print("CUDA not available, using CPU.")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            print(f"GPU device: {gpu_name}")
            print(f"GPU memory: {gpu_memory:.1f} GB")
        
        if config.memory_efficient:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        self.train_loader, self.val_loader, self.dataset = create_data_loaders(
            data_file=config.data_file,
            batch_size=config.batch_size,
            train_ratio=config.train_ratio,
            normalize=True,
            num_workers=config.num_workers,
            target_type=config.mode
        )
        
        model_kwargs = {
            'input_dim': 3,
            'output_dim': 12 if config.mode == 'points' else 15,
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'dropout': config.dropout
        }
        
        if hasattr(config, 'mlp_ratio'):
            model_kwargs['mlp_ratio'] = config.mlp_ratio
        if hasattr(config, 'use_adaptive_norm'):
            model_kwargs['use_adaptive_norm'] = config.use_adaptive_norm
        if hasattr(config, 'use_cross_attention'):
            model_kwargs['use_cross_attention'] = config.use_cross_attention
            
        self.model = DiTModel(**model_kwargs).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        self.use_mixed_precision = config.mixed_precision
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            print("✅ Mixed precision training (FP16) enabled.")
        
        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.effective_batch_size = config.batch_size * self.gradient_accumulation_steps
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.effective_batch_size}")
        
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=self.device
        )
        self.criterion = DiffusionLoss(self.noise_scheduler)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        self.ema_model = EMAModel(self.model, decay=config.ema_decay)
        
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(config.log_dir)
        
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        self.memory_efficient = config.memory_efficient
    
    def train_epoch(self):
        """Train for one epoch (supports gradient accumulation and mixed precision)."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        accumulated_loss = 0
        
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (condition, trajectory) in enumerate(pbar):
            condition = condition.to(self.device, non_blocking=True)
            trajectory = trajectory.to(self.device, non_blocking=True)
            
            if self.use_mixed_precision:
                with autocast():
                    loss = self.criterion(self.model, trajectory, condition)
                    loss = loss / self.gradient_accumulation_steps
            else:
                loss = self.criterion(self.model, trajectory, condition)
                loss = loss / self.gradient_accumulation_steps
            
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                self.ema_model.update()
                
                step_loss = accumulated_loss * self.gradient_accumulation_steps
                total_loss += step_loss
                
                avg_loss = total_loss / ((batch_idx // self.gradient_accumulation_steps) + 1)
                pbar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                    'mem': f'{torch.cuda.memory_reserved()/1e9:.1f}GB' if torch.cuda.is_available() else 'N/A'
                })
                
                if self.global_step % self.config.log_interval == 0:
                    self.writer.add_scalar('Train/Loss', step_loss, self.global_step)
                    self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
                    if torch.cuda.is_available():
                        self.writer.add_scalar('Train/GPU_Memory_GB', torch.cuda.memory_reserved()/1e9, self.global_step)
                
                self.global_step += 1
                accumulated_loss = 0
                
                if self.memory_efficient and torch.cuda.is_available() and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
        
        if accumulated_loss > 0:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.ema_model.update()
        
        return total_loss / max(1, (num_batches + self.gradient_accumulation_steps - 1) // self.gradient_accumulation_steps)
    
    def validate(self):
        """Validation loop (supports mixed precision and memory optimizations)."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_idx, (condition, trajectory) in enumerate(tqdm(self.val_loader, desc="Validating")):
                condition = condition.to(self.device, non_blocking=True)
                trajectory = trajectory.to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        loss = self.criterion(self.model, trajectory, condition)
                else:
                    loss = self.criterion(self.model, trajectory, condition)
                
                total_loss += loss.item()
                
                if self.memory_efficient and torch.cuda.is_available() and batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    

    
    def save_checkpoint(self, is_best=False):
        """Save checkpoint (memory-efficient)."""
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.shadow,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config
        }
        
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(self.config.save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ Best model saved to: {best_path}")
        
        if self.memory_efficient and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint (including mixed-precision state if available)."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.ema_model.shadow = checkpoint['ema_state_dict']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['best_val_loss']
            self.global_step = checkpoint['global_step']
            
            print(f"✅ Loaded checkpoint from {checkpoint_path}, epoch {self.epoch}")
        else:
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
    
    def train(self):
        """Full training loop."""
        print("Start training...")
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config.epochs):
            self.epoch = epoch
            
            train_loss = self.train_epoch()
            
            val_loss = self.validate()
            
            self.scheduler.step()
            
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"time={elapsed/3600:.2f}h")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(is_best)
        
        self.save_checkpoint()
        
        self.writer.close()
        print(f"Training completed, total time: {(time.time() - start_time)/3600:.2f} hours")

def get_config():
    """Build argparse configuration for training."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size (optimized for 4GB VRAM).')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--num_workers', type=int, default=0, help='Use 0 to minimize memory.')
    
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID to use (e.g., 0, 1, 2...), default auto.')
    
    parser.add_argument('--d_model', type=int, default=128, help='Transformer hidden dimension.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of Transformer layers.')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP expansion ratio.')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_adaptive_norm', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to use adaptive LayerNorm.')
    parser.add_argument('--use_cross_attention', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether to use cross-attention blocks.')
    
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--beta_start', type=float, default=5e-5)
    parser.add_argument('--beta_end', type=float, default=2e-2)
    
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    
    parser.add_argument('--mixed_precision', type=lambda x: x.lower() == 'true', default=True,
                        help='Enable mixed precision training.')
    parser.add_argument('--memory_efficient', type=lambda x: x.lower() == 'true', default=False,
                        help='Enable additional memory optimizations.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps.')
    
    parser.add_argument('--save_dir', type=str, default='checkpoints/Polynomial-DiT-local')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)
    
    parser.add_argument('--mode', type=str, default='polynomial', help='Training mode: polynomial or points.')
    
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume training from.')
    
    return parser.parse_args()

def main():
    # Show available GPU information
    print("=== GPU devices ===")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU device(s):")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("No CUDA support detected, using CPU.")
    print("==================")
    
    config = get_config()
    
    # Print dataset statistics
    print("=== Dataset statistics ===")
    get_data_stats(config.data_file)
    
    trainer = Trainer(config)
    
    if config.resume:
        trainer.load_checkpoint(config.resume)
    
    trainer.train()

if __name__ == "__main__":
    main() 