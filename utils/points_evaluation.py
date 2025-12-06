import torch
import numpy as np
from utils.dataset import TrajectoryDataset
from utils.dit_model import DiTModel
from utils.diffusion_utils import NoiseScheduler
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class PointEvaluator:
    """4-point trajectory (12D) prediction evaluator"""

    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
            device=device
        )

    def _load_model(self, model_path):
        """Load trained model weights. The model outputs 12D flattened xyz coordinates of 4 points."""
        model = DiTModel(
            input_dim=3,
            output_dim=12,  # 18-dim flattened 6×3 points
            d_model=128,
            n_heads=8,
            n_layers=6,
            mlp_ratio=4,
            dropout=0.1,
            use_adaptive_norm=True,
            use_cross_attention=True
        )

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from: {model_path}")
        else:
            print(f"Warning: model file not found at {model_path}, using randomly initialized weights.")

        model.to(self.device)
        model.eval()
        return model

    def predict_points(self, conditions, num_samples=1):
        """Predict 4×3 trajectory points (flattened 12D) for given conditions.

        Args:
            conditions (Tensor): [batch, 3] tensor containing (x, y, θ) after the same preprocessing as the training stage.
            num_samples (int): number of trajectories to sample per condition (via stochastic DDPM sampling).
        Returns:
            ndarray: [batch*num_samples, 12] predicted point coordinates.
        """
        conditions = conditions.to(self.device)
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)

        with torch.no_grad():
            pred = self.noise_scheduler.sample(
                model=self.model,
                shape=(conditions.shape[0], 12),
                condition=conditions,
                device=self.device
            )
        return pred  # tensor on device

    @staticmethod
    def calculate_trajectory_rmse(pred_points, true_points):
        """Compute RMSE between predicted and true 6-point trajectories.

        Args:
            pred_points: [..., 6, 3]
            true_points: same shape
        """
        if isinstance(pred_points, torch.Tensor):
            pred_points = pred_points.cpu().numpy()
        if isinstance(true_points, torch.Tensor):
            true_points = true_points.cpu().numpy()

        diff = pred_points - true_points
        mse = np.mean(np.sum(diff ** 2, axis=-1))
        return np.sqrt(mse) 