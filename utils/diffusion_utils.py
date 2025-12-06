import torch
import torch.nn as nn
import numpy as np
import math

class NoiseScheduler:
    """Noise scheduler implementing the DDPM forward (noising) process."""
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=2e-2, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(self, x_start, noise, timesteps):
        """
        Add noise to clean data.
        Args:
            x_start: [batch_size, dim] original clean data.
            noise: [batch_size, dim] sampled Gaussian noise.
            timesteps: [batch_size] time steps.
        Returns:
            x_noisy: [batch_size, dim] noised data at the given time steps.
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def sample_timesteps(self, batch_size):
        """Randomly sample diffusion time steps."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
    
    def denoise_step(self, model, x_t, t, condition):
        """
        One reverse denoising step.
        Args:
            model: trained denoising model.
            x_t: [batch_size, dim] current noised data.
            t: [batch_size] time step.
            condition: [batch_size, input_dim] conditioning vector.
        Returns:
            x_prev: [batch_size, dim] data at previous time step.
        """
        predicted_noise = model(x_t, t, condition)
        
        alpha_t = self.alphas[t].reshape(-1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].reshape(-1, 1)
        
        pred_original_sample = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        pred_prev_sample = sqrt_recip_alpha_t * (x_t - self.betas[t].reshape(-1, 1) * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
        variance = 0
        if t.min() > 0:
            noise = torch.randn_like(x_t)
            variance_t = self.posterior_variance[t].reshape(-1, 1)
            variance = torch.sqrt(variance_t) * noise
        
        return pred_prev_sample + variance
    
    @torch.no_grad()
    def sample(self, model, shape, condition, device):
        """
        Full DDPM sampling process.
        Args:
            model: trained denoising model.
            shape: sampling shape (batch_size, dim).
            condition: [batch_size, input_dim] conditioning vector.
            device: torch device.
        Returns:
            samples: [batch_size, dim] generated samples.
        """
        batch_size = shape[0]
        
        x = torch.randn(shape, device=device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.denoise_step(model, x, t, condition)
        
        return x

class DiffusionLoss(nn.Module):
    """Loss function for the diffusion model (predicting added noise)."""
    def __init__(self, noise_scheduler):
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.mse_loss = nn.MSELoss()
    
    def forward(self, model, x_start, condition):
        """
        Compute diffusion loss.
        Args:
            model: DiT model.
            x_start: [batch_size, dim] original trajectory data.
            condition: [batch_size, input_dim] condition (x, y, theta).
        Returns:
            loss: scalar loss value.
        """
        batch_size = x_start.shape[0]
        
        timesteps = self.noise_scheduler.sample_timesteps(batch_size)
        noise = torch.randn_like(x_start)
        
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, timesteps)
        
        predicted_noise = model(x_noisy, timesteps, condition)
        
        loss = self.mse_loss(predicted_noise, noise)
        
        return loss

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """Linear beta schedule helper."""
    return torch.linspace(start, end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule helper."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)

class EMAModel:
    """Exponential moving average (EMA) wrapper for model parameters."""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters from the current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1 - self.decay) * (self.shadow[name] - param.data)
    
    def apply_shadow(self):
        """Apply EMA (shadow) parameters to the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original (non-EMA) model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = NoiseScheduler(num_timesteps=1000, device=device)
    
    batch_size = 4
    dim = 18
    x_start = torch.randn(batch_size, dim, device=device)
    noise = torch.randn(batch_size, dim, device=device)
    timesteps = scheduler.sample_timesteps(batch_size)
    
    x_noisy = scheduler.add_noise(x_start, noise, timesteps)
    
    print(f"Clean data shape: {x_start.shape}")
    print(f"Noisy data shape: {x_noisy.shape}")
    print(f"Timesteps: {timesteps}")
    print(f"Device: {device}") 