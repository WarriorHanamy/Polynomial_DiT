#!/usr/bin/env python3
"""
Polynomial Trajectory Predictor Interface
Provides a clean interface for predicting polynomial trajectories using a trained model.
"""

import torch
import numpy as np
import os
import sys
import random
from pathlib import Path

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from utils.polynomial_evaluation import PolynomialEvaluator
from utils.dataset import TrajectoryDataset
from utils.points_evaluation import PointEvaluator 

class TrajectoryPredictor:
    def __init__(self, model_path: str, 
             data_path: str, 
             base_dir: str = None, 
             mode: str = "points", 
             device: str = "cuda" if torch.cuda.is_available() else "cpu",
             seed: int = 42):
        """
        Initialize the polynomial trajectory predictor.
        
        Args:
            model_path (str): Path to the model checkpoint file
            data_path (str): Path to the processed data file
            base_dir (str, optional): Base directory for resolving relative paths. 
                                    If None, uses the directory of this file.
            device (str, optional): Device to use for prediction. Defaults to "cuda" if available, otherwise "cpu".
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        
        # Set up base directory
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = Path(base_dir)
        
        self.mode = mode
        
        # Resolve paths
        model_path = self._resolve_path(model_path)
        data_path = self._resolve_path(data_path)
        
        # Initialize the polynomial evaluator
        if self.mode == "polynomial":
            self.evaluator = PolynomialEvaluator(str(model_path))
            self.dataset = TrajectoryDataset(str(data_path), normalize=True, target_type="polynomial")
            self.condition_scaler, self.coeff_scaler = self.dataset.get_scalers()
        elif self.mode=="points":
            self.evaluator = PointEvaluator(str(model_path))
            self.dataset = TrajectoryDataset(str(data_path), normalize=True, target_type="points")
            self.condition_scaler, self.point_scaler = self.dataset.get_scalers()
        self.device = device
        

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path relative to the base directory.
        
        Args:
            path (str): Path to resolve
            
        Returns:
            Path: Resolved absolute path
        """
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_dir / path
    
    def predict_trajectory(self, condition: np.ndarray, t: np.ndarray = None) -> np.ndarray:
        """
        Predict trajectory for given condition and time points.
        
        Args:
            condition (np.ndarray): Input condition array [x, y, theta]
            t (np.ndarray): Time points array (should be between 0 and 1)
            
        Returns:
            np.ndarray: Predicted trajectory points of shape (n, 3) where n is the length of t
        """
        if self.mode=="polynomial":
            # Normalize condition
            condition_norm = self.condition_scaler.transform(condition.reshape(1, -1))
            condition_tensor = torch.from_numpy(condition_norm).float().to(self.device)
            
            # Predict polynomial coefficients
            pred_coeffs = self.evaluator.predict_polynomial_coeffs(condition_tensor)
            # 确保在转换为numpy之前将张量移到CPU
            pred_coeffs = pred_coeffs.cpu()
            pred_coeffs_denorm = self.coeff_scaler.inverse_transform(pred_coeffs.numpy())[0]
            
            # Reshape coefficients to (3, 5) for x, y, z
            pred_coeffs_reshaped = pred_coeffs_denorm.reshape(3, 5)
            
            # Generate trajectory points
            trajectory = self.dataset.polynomial_to_trajectory(pred_coeffs_reshaped, t)
            
            return trajectory
        elif self.mode=="points":
            condition_norm = self.condition_scaler.transform(condition.reshape(1, -1))
            condition_tensor = torch.from_numpy(condition_norm).float().to(self.device)
            trajectory = self.evaluator.predict_points(condition_tensor).cpu().numpy()
            trajectory_denorm = self.point_scaler.inverse_transform(trajectory)
            trajectory = trajectory_denorm.reshape(-1, 3)
            return trajectory
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

# Example usage:
if __name__ == "__main__":
    # Initialize predictor with relative paths
    polynomial_predictor = TrajectoryPredictor(
        model_path="checkpoints/Polynomial-DiT-local/strike_checkpoint_latest.pth",
        data_path="datasets/strike_processed.txt",
        mode="polynomial"
    )
    
    points_predictor = TrajectoryPredictor(
        model_path="checkpoints/Polynomial-DiT-local/strike_checkpoint_latest_4_points.pth",
        data_path="datasets/strike_processed.txt",
        mode="points"
    )
    
    # Example condition [x, y, theta]
    condition = np.array([-2, 0.4, 0.78])
    
    # Example time points
    t = np.linspace(0, 1, 100)
    
    # Predict trajectory
    trajectory = polynomial_predictor.predict_trajectory(condition, t)
    
    print(f"Predicted trajectory shape: {trajectory.shape}")
    print(f"First few points:\n{trajectory[:5]}") 
    
    trajectory_points = points_predictor.predict_trajectory(condition)
    print(f"Predicted trajectory points shape: {trajectory_points.shape}")
    print(f"First few points:\n{trajectory_points}") 