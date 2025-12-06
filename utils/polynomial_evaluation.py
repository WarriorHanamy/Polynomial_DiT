import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from utils.dataset import TrajectoryDataset, create_data_loaders
from utils.dit_model import DiTModel
from utils.diffusion_utils import NoiseScheduler
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

class PolynomialEvaluator:
    """Evaluator for polynomial-parameterized trajectory prediction."""
    
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
        """Load the trained model checkpoint."""
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
    
    def predict_polynomial_coeffs(self, conditions, num_samples=1):
        """
        Predict polynomial coefficients.
        Args:
            conditions: [batch_size, 3] conditions (x, y, theta).
            num_samples: number of samples per condition.
        Returns:
            polynomial_coeffs: [batch_size * num_samples, 15] coefficients.
        """
        batch_size = conditions.shape[0]
        conditions = conditions.to(self.device)
        import time
        np.random.seed(int(1))
        if num_samples > 1:
            conditions = conditions.repeat_interleave(num_samples, dim=0)
        
        with torch.no_grad():
            predicted_coeffs = self.noise_scheduler.sample(
                model=self.model,
                shape=(conditions.shape[0], 15),
                condition=conditions,
                device=self.device
            )
        
        return predicted_coeffs
    
    def polynomial_to_trajectory(self, polynomial_coeffs, t_values=None):
        """
        Convert polynomial coefficients to trajectory points.
        Args:
            polynomial_coeffs: [batch_size, 15] coefficients.
            t_values: [n_points] parameter values in [0, 1], default 6 evenly-spaced points.
        Returns:
            trajectories: [batch_size, n_points, 3] xyz points.
        """
        if t_values is None:
            t_values = np.linspace(0, 1, 6)
        
        if isinstance(polynomial_coeffs, torch.Tensor):
            polynomial_coeffs = polynomial_coeffs.cpu().numpy()
        
        batch_size = polynomial_coeffs.shape[0]
        n_points = len(t_values)
        
        coeffs = polynomial_coeffs.reshape(batch_size, 3, 5)
        
        trajectories = np.zeros((batch_size, n_points, 3))
        
        for i, t in enumerate(t_values):
            for coord in range(3):  # x, y, z
                for batch_idx in range(batch_size):
                    trajectories[batch_idx, i, coord] = np.polyval(coeffs[batch_idx, coord, :], t)
        
        return trajectories
    
    def calculate_trajectory_rmse(self, pred_trajectories, true_trajectories):
        """
        Compute trajectory RMSE.
        Args:
            pred_trajectories: [batch_size, n_points, 3] predicted trajectories.
            true_trajectories: [batch_size, n_points, 3] ground-truth trajectories.
        Returns:
            rmse: root-mean-square error.
        """
        if isinstance(pred_trajectories, torch.Tensor):
            pred_trajectories = pred_trajectories.cpu().numpy()
        if isinstance(true_trajectories, torch.Tensor):
            true_trajectories = true_trajectories.cpu().numpy()
        
        diff = pred_trajectories - true_trajectories
        squared_distances = np.sum(diff ** 2, axis=2)  # [batch_size, n_points]
        mse = np.mean(squared_distances)
        rmse = np.sqrt(mse)
        
        return rmse
    
    def calculate_polynomial_distance(self, pred_coeffs, true_coeffs):
        """
        Compute L2 distance between predicted and ground-truth polynomial coefficients.
        Args:
            pred_coeffs: [batch_size, 15] predicted coefficients.
            true_coeffs: [batch_size, 15] ground-truth coefficients.
        Returns:
            l2_distance: mean L2 distance across batch.
        """
        if isinstance(pred_coeffs, torch.Tensor):
            pred_coeffs = pred_coeffs.cpu().numpy()
        if isinstance(true_coeffs, torch.Tensor):
            true_coeffs = true_coeffs.cpu().numpy()
        
        diff = pred_coeffs - true_coeffs
        l2_distances = np.sqrt(np.sum(diff ** 2, axis=1))  # [batch_size]
        return np.mean(l2_distances)
    
    def evaluate_dataset(self, data_file="datas_processed.txt", num_samples=100):
        """
        Evaluate model performance on a dataset.
        Args:
            data_file: path to processed dataset.
            num_samples: number of samples used for evaluation.
        Returns:
            evaluation_results: dictionary of metrics and raw arrays.
        """
        print("Starting model evaluation...")
        print(f"Data file: {data_file}")
        print(f"Number of evaluation samples: {num_samples}")
        
        dataset = TrajectoryDataset(data_file, normalize=True, target_type="polynomial")
        condition_scaler, coeff_scaler = dataset.get_scalers()
        point_scaler = dataset.get_point_scaler()
        
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        pred_coeffs_list = []
        true_coeffs_list = []
        pred_trajectories_list = []
        true_trajectories_list = []
        conditions_list = []
        
        batch_size = 32
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            
            batch_conditions = []
            batch_true_coeffs = []
            batch_true_trajectories = []
            
            for idx in batch_indices:
                condition, true_coeffs = dataset[idx]
                true_trajectory = dataset.get_trajectory_points(idx)
                
                batch_conditions.append(condition)
                batch_true_coeffs.append(true_coeffs)
                batch_true_trajectories.append(true_trajectory)
            
            batch_conditions = torch.stack(batch_conditions)
            batch_true_coeffs = torch.stack(batch_true_coeffs)
            batch_true_trajectories = torch.stack(batch_true_trajectories)
            
            pred_coeffs = self.predict_polynomial_coeffs(batch_conditions)
            
            pred_coeffs_denorm = coeff_scaler.inverse_transform(pred_coeffs.cpu().numpy())
            true_coeffs_denorm = coeff_scaler.inverse_transform(batch_true_coeffs.cpu().numpy())
            
            pred_trajectories = self.polynomial_to_trajectory(pred_coeffs_denorm)
            true_trajectories_denorm = point_scaler.inverse_transform(batch_true_trajectories.cpu().numpy().reshape(-1, 18))
            true_trajectories = true_trajectories_denorm.reshape(-1, 6, 3)
            
            pred_coeffs_list.append(pred_coeffs_denorm)
            true_coeffs_list.append(true_coeffs_denorm)
            pred_trajectories_list.append(pred_trajectories)
            true_trajectories_list.append(true_trajectories)
            conditions_list.append(condition_scaler.inverse_transform(batch_conditions.cpu().numpy()))
        
        all_pred_coeffs = np.concatenate(pred_coeffs_list, axis=0)
        all_true_coeffs = np.concatenate(true_coeffs_list, axis=0)
        all_pred_trajectories = np.concatenate(pred_trajectories_list, axis=0)
        all_true_trajectories = np.concatenate(true_trajectories_list, axis=0)
        all_conditions = np.concatenate(conditions_list, axis=0)
        
        coeff_l2_distance = self.calculate_polynomial_distance(all_pred_coeffs, all_true_coeffs)
        trajectory_rmse = self.calculate_trajectory_rmse(all_pred_trajectories, all_true_trajectories)
        
        x_rmse = self.calculate_trajectory_rmse(all_pred_trajectories[:, :, 0:1], all_true_trajectories[:, :, 0:1])
        y_rmse = self.calculate_trajectory_rmse(all_pred_trajectories[:, :, 1:2], all_true_trajectories[:, :, 1:2])
        z_rmse = self.calculate_trajectory_rmse(all_pred_trajectories[:, :, 2:3], all_true_trajectories[:, :, 2:3])
        
        results = {
            'num_samples': len(all_pred_coeffs),
            'polynomial_l2_distance': coeff_l2_distance,
            'trajectory_rmse': trajectory_rmse,
            'x_rmse': x_rmse,
            'y_rmse': y_rmse,
            'z_rmse': z_rmse,
            'pred_coeffs': all_pred_coeffs,
            'true_coeffs': all_true_coeffs,
            'pred_trajectories': all_pred_trajectories,
            'true_trajectories': all_true_trajectories,
            'conditions': all_conditions
        }
        
        print("\n=== Evaluation results ===")
        print(f"Number of evaluated samples: {results['num_samples']}")
        print(f"Polynomial coefficient L2 distance: {coeff_l2_distance:.6f}")
        print(f"Overall trajectory RMSE: {trajectory_rmse:.6f}")
        print(f"X RMSE: {x_rmse:.6f}")
        print(f"Y RMSE: {y_rmse:.6f}")
        print(f"Z RMSE: {z_rmse:.6f}")
        
        return results
    
    def visualize_predictions(self, results, num_examples=6, save_path="logs/polynomial_evaluation.png"):
        """
        Visualize predicted vs ground-truth trajectories.
        Args:
            results: evaluation results dictionary.
            num_examples: number of examples to show.
            save_path: path to save the figure.
        """
        pred_trajectories = results['pred_trajectories']
        true_trajectories = results['true_trajectories']
        conditions = results['conditions']
        
        fig = plt.figure(figsize=(20, 12))
        
        indices = np.random.choice(len(pred_trajectories), num_examples, replace=False)
        
        for i, idx in enumerate(indices):
            pred_traj = pred_trajectories[idx]
            true_traj = true_trajectories[idx]
            condition = conditions[idx]
            
            ax = fig.add_subplot(2, num_examples, i+1, projection='3d')
            
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
                   'b-o', linewidth=2, markersize=8, label='Predicted', alpha=0.8)
            
            ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 
                   'r-s', linewidth=2, markersize=6, label='Ground Truth', alpha=0.8)
            
            ax.scatter([pred_traj[0, 0]], [pred_traj[0, 1]], [pred_traj[0, 2]], 
                      c='blue', s=100, marker='^', label='Pred Start')
            ax.scatter([true_traj[0, 0]], [true_traj[0, 1]], [true_traj[0, 2]], 
                      c='red', s=100, marker='^', label='True Start')
            
            rmse = self.calculate_trajectory_rmse(pred_traj[None], true_traj[None])
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Sample {i+1}\nCondition: ({condition[0]:.2f}, {condition[1]:.2f}, {condition[2]:.2f})\nRMSE: {rmse:.4f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(2, num_examples, i+1+num_examples)
            ax2.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-o', linewidth=2, markersize=6, label='Predicted')
            ax2.plot(true_traj[:, 0], true_traj[:, 1], 'r-s', linewidth=2, markersize=4, label='Ground Truth')
            ax2.scatter([pred_traj[0, 0]], [pred_traj[0, 1]], c='blue', s=80, marker='^')
            ax2.scatter([true_traj[0, 0]], [true_traj[0, 1]], c='red', s=80, marker='^')
            
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title(f'XY Projection - Sample {i+1}')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
        
        plt.tight_layout()
        plt.suptitle('Polynomial Trajectory Prediction Results\n(Top: 3D View, Bottom: XY Projection)', 
                     fontsize=16, y=0.98)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {save_path}")

    def visualize_polynomial_coefficients(self, results, num_examples=6, save_path="logs/polynomial_coefficients.png"):
        """
        Visualize polynomial coefficient predictions.
        Args:
            results: evaluation results dictionary.
            num_examples: number of examples to show.
            save_path: path to save the figure.
        """
        pred_coeffs = results['pred_coeffs']
        true_coeffs = results['true_coeffs']
        conditions = results['conditions']
        
        fig = plt.figure(figsize=(20, 15))
        
        indices = np.random.choice(len(pred_coeffs), num_examples, replace=False)
        
        for i, idx in enumerate(indices):
            pred_coeff = pred_coeffs[idx]
            true_coeff = true_coeffs[idx]
            condition = conditions[idx]
            
            pred_coeff_reshaped = pred_coeff.reshape(3, 5)
            true_coeff_reshaped = true_coeff.reshape(3, 5)
            
            ax = fig.add_subplot(3, num_examples, i+1)
            
            x_pos = np.arange(15)
            width = 0.35
            
            ax.bar(x_pos - width/2, pred_coeff, width, label='Predicted', alpha=0.8, color='blue')
            ax.bar(x_pos + width/2, true_coeff, width, label='Ground Truth', alpha=0.8, color='red')
            
            coeff_l2 = np.sqrt(np.sum((pred_coeff - true_coeff) ** 2))
            
            ax.set_xlabel('Coefficient Index')
            ax.set_ylabel('Value')
            ax.set_title(f'Sample {i+1}\nCondition: ({condition[0]:.2f}, {condition[1]:.2f}, {condition[2]:.2f})\nL2 Distance: {coeff_l2:.4f}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=9.5, color='gray', linestyle='--', alpha=0.5)
            ax.text(2, ax.get_ylim()[1]*0.9, 'X', ha='center', fontweight='bold')
            ax.text(7, ax.get_ylim()[1]*0.9, 'Y', ha='center', fontweight='bold')
            ax.text(12, ax.get_ylim()[1]*0.9, 'Z', ha='center', fontweight='bold')
        
        coord_names = ['X', 'Y', 'Z']
        colors = ['red', 'green', 'blue']
        
        for coord in range(3):
            ax = fig.add_subplot(3, 3, 6 + coord + 1)
            
            for coeff_idx in range(5):
                global_idx = coord * 5 + coeff_idx
                ax.scatter(true_coeffs[:100, global_idx], pred_coeffs[:100, global_idx], 
                          alpha=0.6, s=30, label=f'c{coeff_idx}')
            
            min_val = min(np.min(true_coeffs[:100, coord*5:(coord+1)*5]), 
                         np.min(pred_coeffs[:100, coord*5:(coord+1)*5]))
            max_val = max(np.max(true_coeffs[:100, coord*5:(coord+1)*5]), 
                         np.max(pred_coeffs[:100, coord*5:(coord+1)*5]))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect')
            
            ax.set_xlabel(f'True {coord_names[coord]} Coefficients')
            ax.set_ylabel(f'Predicted {coord_names[coord]} Coefficients')
            ax.set_title(f'{coord_names[coord]} Coordinate Coefficients')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        ax = fig.add_subplot(3, 3, 9)
        
        coeff_errors = np.abs(pred_coeffs - true_coeffs)
        mean_errors = np.mean(coeff_errors, axis=0)
        std_errors = np.std(coeff_errors, axis=0)
        
        x_pos = np.arange(15)
        ax.bar(x_pos, mean_errors, yerr=std_errors, capsize=3, alpha=0.7, color='orange')
        ax.set_xlabel('Coefficient Index')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Mean Coefficient Errors (±std)')
        ax.grid(True, alpha=0.3)
        
        ax.axvline(x=4.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=9.5, color='gray', linestyle='--', alpha=0.5)
        ax.text(2, ax.get_ylim()[1]*0.9, 'X', ha='center', fontweight='bold')
        ax.text(7, ax.get_ylim()[1]*0.9, 'Y', ha='center', fontweight='bold')
        ax.text(12, ax.get_ylim()[1]*0.9, 'Z', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle('Polynomial Coefficients Prediction Analysis', fontsize=16, y=0.98)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Polynomial coefficient visualization saved to: {save_path}")

    def visualize_polynomial_trajectories(self, results, num_examples=6, save_path="logs/polynomial_trajectories.png"):
        """
        Visualize continuous polynomial trajectories vs discrete GT points.
        Args:
            results: evaluation results dictionary.
            num_examples: number of examples to show.
            save_path: path to save the figure.
        """
        pred_coeffs = results['pred_coeffs']
        true_coeffs = results['true_coeffs']
        conditions = results['conditions']
        
        fig = plt.figure(figsize=(24, 16))
        
        indices = np.random.choice(len(pred_coeffs), num_examples, replace=False)
        
        t_dense = np.linspace(0, 1, 100)  # 100 points for smooth curves
        t_discrete = np.linspace(0, 1, 6)   # 6 discrete GT points
        
        for i, idx in enumerate(indices):
            pred_coeff = pred_coeffs[idx]
            true_coeff = true_coeffs[idx]
            condition = conditions[idx]
            
            pred_trajectory_continuous = self.polynomial_to_trajectory(pred_coeff[None], t_dense)[0]
            
            true_trajectory_discrete = self.polynomial_to_trajectory(true_coeff[None], t_discrete)[0]
            
            ax = fig.add_subplot(3, num_examples, i+1, projection='3d')
            
            ax.plot(pred_trajectory_continuous[:, 0], pred_trajectory_continuous[:, 1], pred_trajectory_continuous[:, 2], 
                   'b-', linewidth=3, label='Predicted Polynomial', alpha=0.8)
            
            ax.scatter(true_trajectory_discrete[:, 0], true_trajectory_discrete[:, 1], true_trajectory_discrete[:, 2], 
                      c='red', s=80, marker='o', label='GT Discrete Points', alpha=0.9, edgecolors='darkred', linewidth=1)
            
            ax.plot(true_trajectory_discrete[:, 0], true_trajectory_discrete[:, 1], true_trajectory_discrete[:, 2], 
                   'r--', linewidth=1.5, alpha=0.6, label='GT Connection')
            
            ax.scatter([pred_trajectory_continuous[0, 0]], [pred_trajectory_continuous[0, 1]], [pred_trajectory_continuous[0, 2]], 
                      c='blue', s=100, marker='^', label='Pred Start', edgecolors='darkblue')
            ax.scatter([pred_trajectory_continuous[-1, 0]], [pred_trajectory_continuous[-1, 1]], [pred_trajectory_continuous[-1, 2]], 
                      c='blue', s=100, marker='s', label='Pred End', edgecolors='darkblue')
            
            coeff_l2 = np.sqrt(np.sum((pred_coeff - true_coeff) ** 2))
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Sample {i+1} - Continuous vs Discrete\nCondition: ({condition[0]:.2f}, {condition[1]:.2f}, {condition[2]:.2f})\nCoeff L2: {coeff_l2:.4f}')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(3, num_examples, i+1+num_examples)
            
            ax2.plot(pred_trajectory_continuous[:, 0], pred_trajectory_continuous[:, 1], 'b-', linewidth=3, label='Predicted Polynomial')
            
            ax2.scatter(true_trajectory_discrete[:, 0], true_trajectory_discrete[:, 1], c='red', s=80, marker='o', 
                       label='GT Points', alpha=0.9, edgecolors='darkred', linewidth=1)
            ax2.plot(true_trajectory_discrete[:, 0], true_trajectory_discrete[:, 1], 'r--', linewidth=1.5, alpha=0.6)
            
            ax2.scatter([pred_trajectory_continuous[0, 0]], [pred_trajectory_continuous[0, 1]], c='blue', s=80, marker='^')
            ax2.scatter([pred_trajectory_continuous[-1, 0]], [pred_trajectory_continuous[-1, 1]], c='blue', s=80, marker='s')
            
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title(f'XY Projection - Sample {i+1}')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.axis('equal')
            
            ax3 = fig.add_subplot(3, num_examples, i+1+2*num_examples)
            
            ax3.plot(t_dense, pred_trajectory_continuous[:, 2], 'b-', linewidth=3, label='Predicted Z')
            
            ax3.scatter(t_discrete, true_trajectory_discrete[:, 2], c='red', s=80, marker='o', 
                       label='GT Z Points', alpha=0.9, edgecolors='darkred', linewidth=1)
            ax3.plot(t_discrete, true_trajectory_discrete[:, 2], 'r--', linewidth=1.5, alpha=0.6)
            
            ax3.set_xlabel('Parameter t')
            ax3.set_ylabel('Z Coordinate')
            ax3.set_title(f'Z vs Time - Sample {i+1}')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Polynomial Trajectory Generation: Continuous Prediction vs Discrete Ground Truth\n(Blue: Continuous Polynomial, Red: Original Discrete Points)', 
                     fontsize=18, y=0.98)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Polynomial trajectory visualization saved to: {save_path}")
        print("Note: red dots are original 6 GT points; blue line is the 100-point continuous polynomial trajectory.")

def main():
    # Evaluation configuration
    model_path = "checkpoints/Polynomial-DiT-local/strike_checkpiont_latest.pth"
    data_file = "datas_processed.txt"
    num_samples = 200
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    evaluator = PolynomialEvaluator(model_path, device)
    
    results = evaluator.evaluate_dataset(data_file, num_samples)
    
    print("\n=== Polynomial coefficient prediction analysis ===")
    pred_coeffs = results['pred_coeffs']
    true_coeffs = results['true_coeffs']
    
    for coord, coord_name in enumerate(['X', 'Y', 'Z']):
        print(f"\n{coord_name} polynomial coefficients (c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0):")
        for i in range(5):
            coeff_idx = coord * 5 + i
            pred_mean = np.mean(pred_coeffs[:, coeff_idx])
            true_mean = np.mean(true_coeffs[:, coeff_idx])
            mae = np.mean(np.abs(pred_coeffs[:, coeff_idx] - true_coeffs[:, coeff_idx]))
            print(f"  c{4-i} - pred mean: {pred_mean:8.4f}, true mean: {true_mean:8.4f}, MAE: {mae:.4f}")
    
    print("\n=== Polynomial trajectory generation visualization ===")
    evaluator.visualize_polynomial_trajectories(results, num_examples=6)
    
    evaluator.visualize_polynomial_coefficients(results, num_examples=6)
    
    results_summary = {
        'polynomial_l2_distance': float(results['polynomial_l2_distance']),
        'trajectory_rmse': float(results['trajectory_rmse']),
        'x_rmse': float(results['x_rmse']),
        'y_rmse': float(results['y_rmse']),
        'z_rmse': float(results['z_rmse']),
        'coefficient_statistics': {
            coord_name: {
                f'c{4-i}_mae': float(np.mean(np.abs(pred_coeffs[:, coord*5+i] - true_coeffs[:, coord*5+i])))
                for i in range(5)
            }
            for coord, coord_name in enumerate(['x', 'y', 'z'])
        }
    }
    
    import json
    with open('logs/polynomial_evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\nDetailed evaluation results saved to: logs/polynomial_evaluation_results.json")

if __name__ == "__main__":
    main() 