import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for diffusion time steps."""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, timesteps):
        return self.pe[timesteps]

class MLP(nn.Module):
    """Enhanced multi-layer perceptron module."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1, use_residual=False):
        super().__init__()
        self.use_residual = use_residual and (in_dim == out_dim)
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        out = self.net(x)
        if self.use_residual:
            out = out + x
        return out

class MultiHeadAttention(nn.Module):
    """Enhanced multi-head self-attention."""
    def __init__(self, d_model, n_heads, dropout=0.1, use_flash_attention=False):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_flash_attention = use_flash_attention
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        max_relative_position = 32
        self.relative_position_bias = nn.Parameter(torch.randn(2 * max_relative_position - 1, 2 * max_relative_position - 1, n_heads))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if seq_len <= self.relative_position_bias.shape[0] and seq_len <= self.relative_position_bias.shape[1]:
            relative_position_bias = self.relative_position_bias[:seq_len, :seq_len, :].permute(2, 0, 1).unsqueeze(0)
            scores = scores + relative_position_bias
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attn_output)

class TransformerBlock(nn.Module):
    """Enhanced Transformer block."""
    def __init__(self, d_model, n_heads, mlp_ratio=4, dropout=0.1, layer_scale_init=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), d_model, dropout, use_residual=False)
        
        self.layer_scale_1 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        self.layer_scale_2 = nn.Parameter(torch.ones(d_model) * layer_scale_init)
        
    def forward(self, x):
        x = x + self.layer_scale_1 * self.attn(self.norm1(x))
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))
        return x

class AdaptiveLayerNorm(nn.Module):
    """Adaptive LayerNorm; normalization parameters are conditioned on time embeddings."""
    def __init__(self, d_model, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.time_proj = nn.Linear(time_dim, d_model * 2)
        
    def forward(self, x, time_emb):
        x_norm = self.norm(x)
        time_params = self.time_proj(time_emb).unsqueeze(1)  # [batch_size, 1, d_model * 2]
        
        scale, shift = time_params.chunk(2, dim=-1)
        return x_norm * (1 + scale) + shift

class CrossAttentionBlock(nn.Module):
    """Cross-attention block to better fuse conditional information."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4, d_model, dropout)
        
    def forward(self, x, context):
        x = x + self.self_attn(self.norm1(x))
        # Simplified cross attention: in practice, query=x and key=value=context.
        x = x + self.cross_attn(self.norm2(x))
        x = x + self.mlp(self.norm3(x))
        return x

class DiTModel(nn.Module):
    """Enhanced DiT (Diffusion Transformer) model (~20M parameters)."""
    def __init__(
        self,
        input_dim=3,        # input dim (x, y, theta)
        output_dim=15,      # output dim (15-D polynomial coeffs or 12-D points)
        d_model=128,
        n_heads=8,
        n_layers=6,
        mlp_ratio=4,
        dropout=0.1,
        max_timesteps=1000,
        use_adaptive_norm=True,
        use_cross_attention=True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim  # 15 (polynomial) or 12 (points)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.use_adaptive_norm = use_adaptive_norm
        self.use_cross_attention = use_cross_attention
        
        self.time_embedding = PositionalEncoding(d_model // 2, max_timesteps)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.condition_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.noise_embedding = nn.Sequential(
            nn.Linear(output_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.feature_mixer = MLP(d_model, d_model * 2, d_model, dropout, use_residual=True)
        
        if use_cross_attention:
            self.transformer_blocks = nn.ModuleList([
                CrossAttentionBlock(d_model, n_heads, dropout) if i % 3 == 0 
                else TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
                for i in range(n_layers)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
                for _ in range(n_layers)
            ])
        
        if use_adaptive_norm:
            self.adaptive_norms = nn.ModuleList([
                AdaptiveLayerNorm(d_model, d_model) for _ in range(n_layers)
            ])
        
        self.norm_out = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
        
        # Learnable positional embeddings for the 4-token sequence.
        self.pos_embedding = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)
        
        # Extra global context token.
        self.global_context = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        self._init_weights()
    
    def _init_weights(self):
        """Enhanced weight initialization for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module, std=0.02)
    
    def forward(self, x_noisy, timesteps, condition):
        """
        Forward pass of the diffusion transformer.
        Args:
            x_noisy: [batch_size, output_dim] noised trajectory.
            timesteps: [batch_size] diffusion time steps.
            condition: [batch_size, input_dim] condition (x, y, theta).
        Returns:
            predicted_noise: [batch_size, output_dim] predicted noise.
        """
        batch_size = x_noisy.shape[0]
        
        time_emb = self.time_embedding(timesteps)  # [batch_size, d_model//2]
        time_emb = self.time_mlp(time_emb)  # [batch_size, d_model]
        
        cond_emb = self.condition_embedding(condition)  # [batch_size, d_model]
        
        noise_emb = self.noise_embedding(x_noisy)  # [batch_size, d_model]
        
        # Feature mixing with a small time injection.
        mixed_features = self.feature_mixer(noise_emb + time_emb * 0.1)
        
        # Build sequence: [global_context, condition, time, mixed_features]
        global_ctx = self.global_context.expand(batch_size, -1, -1)
        sequence = torch.cat([
            global_ctx,
            cond_emb.unsqueeze(1), 
            time_emb.unsqueeze(1), 
            mixed_features.unsqueeze(1)
        ], dim=1)  # [batch_size, 4, d_model]
        
        sequence = sequence + self.pos_embedding
        
        for i, block in enumerate(self.transformer_blocks):
            if self.use_adaptive_norm and i < len(self.adaptive_norms):
                sequence = self.adaptive_norms[i](sequence, time_emb)
            
            if self.use_cross_attention and isinstance(block, CrossAttentionBlock):
                sequence = block(sequence, sequence)
            else:
                sequence = block(sequence)
        
        output = sequence[:, 3, :]  # [batch_size, d_model]
        
        output = self.norm_out(output)
        predicted_noise = self.output_projection(output)
        
        return predicted_noise
    
    def get_num_params(self):
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def count_parameters(model):
    """Return total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == "__main__":
    model = DiTModel(
        input_dim=3,
        output_dim=15,
        d_model=128,
        n_heads=8,
        n_layers=6,
        mlp_ratio=4,
        dropout=0.1,
        use_adaptive_norm=True,
        use_cross_attention=True
    )
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {total / 1e6:.2f}M")
    
    batch_size = 4
    x_noisy = torch.randn(batch_size, 15)
    timesteps = torch.randint(0, 1000, (batch_size,))
    condition = torch.randn(batch_size, 3)
    
    with torch.no_grad():
        output = model(x_noisy, timesteps, condition)
        print(f"Input shape: {x_noisy.shape}")
        print(f"Output shape: {output.shape}")
    
    print("\n=== Model architecture ===")
    print(f"d_model: {model.d_model}")
    print(f"n_heads: {model.n_heads}")
    print(f"n_layers: {model.n_layers}")
    print(f"mlp_ratio: {model.mlp_ratio}")
    print(f"use_adaptive_norm: {model.use_adaptive_norm}")
    print(f"use_cross_attention: {model.use_cross_attention}")
    