import torch
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class TrajectoryDataset(data.Dataset):
    """Trajectory dataset supporting both polynomial coefficients and discrete points."""
    def __init__(self, data_file=None, normalize=True, target_type="polynomial"):
        """
        Args:
            data_file: Path to the data file.
            normalize: Whether to apply standardization.
            target_type: Target type, "polynomial" (15-D coeffs) or "points" (12-D points).
        """
        self.data_file = data_file
        self.normalize = normalize
        self.target_type = target_type
        
        self.data = self._load_data()
        
        self.conditions, self.trajectory_points, self.polynomial_coeffs = self._process_data()
        
        if target_type == "polynomial":
            self.targets = self.polynomial_coeffs
            target_dim = 15
            print("Target type: 15D polynomial coefficients")
        elif target_type == "points":
            self.targets = self.trajectory_points
            target_dim = 12
            print("Target type: 12D trajectory points")
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        if self.normalize:
            self.condition_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
            self.point_scaler = StandardScaler()
            
            self.conditions = self.condition_scaler.fit_transform(self.conditions)
            self.targets = self.target_scaler.fit_transform(self.targets)
            self.trajectory_points = self.point_scaler.fit_transform(self.trajectory_points)
        
        self.conditions = torch.FloatTensor(self.conditions)
        self.targets = torch.FloatTensor(self.targets)
        self.trajectory_points = torch.FloatTensor(self.trajectory_points)
        
        print(f"Dataset size: {len(self)}")
        print(f"Condition dimension: {self.conditions.shape[1]}")
        print(f"Target dimension: {self.targets.shape[1]} ({target_type})")
    
    def _load_data(self):
        """Load raw data from file."""
        try:
            data = pd.read_csv(self.data_file, header=None, sep=',')
            return data.values
        except Exception as e:
            print(f"Failed to read data file with pandas: {e}, falling back to manual parsing.")
            data = []
            with open(self.data_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        values = [float(x) for x in line.split(',')]
                        data.append(values)
            return np.array(data)
    
    def _process_data(self):
        """Process data into conditions, trajectory points and polynomial coefficients (indexed 38-D format)."""
        actual_dim = self.data.shape[1] if len(self.data) > 0 else 0
        
        if actual_dim != 38:
            raise ValueError(f"Data dimension mismatch. Expected 38-D (indexed format), got {actual_dim}. "
                             f"Please use an indexed data file.")
        
        print("Detected indexed data format (38-D).")
        
        conditions = []
        trajectory_points = []
        polynomial_coeffs = []
        
        for row in self.data:
            data_row = row[1:]
            
            x, y = data_row[0], data_row[1]
            theta = data_row[3]  # 4th dim is theta
            condition = [x, y, theta]
            
            trajectory = data_row[7:19]
            
            poly_coeffs = data_row[22:37]
            
            conditions.append(condition)
            trajectory_points.append(trajectory)
            polynomial_coeffs.append(poly_coeffs)
        
        return np.array(conditions), np.array(trajectory_points), np.array(polynomial_coeffs)
    
    
    def __len__(self):
        return len(self.conditions)
    
    def __getitem__(self, idx):
        condition = self.conditions[idx]
        target = self.targets[idx]
        return condition, target
    
    def get_trajectory_points(self, idx):
        """Get trajectory point coordinates (for visualization)."""
        return self.trajectory_points[idx]
    
    def get_scalers(self):
        """Get fitted scalers for denormalization."""
        if self.target_type == "polynomial":
            if hasattr(self, 'condition_scaler') and hasattr(self, 'target_scaler'):
                return self.condition_scaler, self.target_scaler
            else:
                return None, None
        elif self.target_type == "points":
            if hasattr(self, 'condition_scaler') and hasattr(self, 'point_scaler'):
                return self.condition_scaler, self.point_scaler
            else:
                return None, None
        else:
            return None, None
    
    def get_point_scaler(self):
        """Get point-coordinate scaler."""
        if hasattr(self, 'point_scaler'):
            return self.point_scaler
        else:
            return None
    
    def denormalize_target(self, normalized_target):
        """Denormalize target (polynomial coefficients or point coordinates)."""
        if hasattr(self, 'target_scaler'):
            if isinstance(normalized_target, torch.Tensor):
                normalized_target = normalized_target.cpu().numpy()
            return self.target_scaler.inverse_transform(normalized_target)
        return normalized_target
    
    def denormalize_points(self, normalized_points):
        """Denormalize point coordinates."""
        if hasattr(self, 'point_scaler'):
            if isinstance(normalized_points, torch.Tensor):
                normalized_points = normalized_points.cpu().numpy()
            return self.point_scaler.inverse_transform(normalized_points)
        return normalized_points
    
    def denormalize_condition(self, normalized_condition):
        """Denormalize conditions."""
        if hasattr(self, 'condition_scaler'):
            if isinstance(normalized_condition, torch.Tensor):
                normalized_condition = normalized_condition.cpu().numpy()
            return self.condition_scaler.inverse_transform(normalized_condition)
        return normalized_condition
    

    def polynomial_to_trajectory(self, coeffs: np.ndarray, t_values: np.ndarray=None):
        if t_values is None:
            t_values = np.linspace(0, 1, 6)
        
        n_points = len(t_values)
        trajectory = np.zeros((n_points, 3))
        
        for i, t in enumerate(t_values):
            for coord in range(3):  # x, y, z
                # coeffs[coord] is from high to low order
                trajectory[i, coord] = np.polyval(coeffs[coord], t)
        return trajectory

def create_data_loaders(data_file=None, 
                       batch_size=32, 
                       train_ratio=0.8, 
                       normalize=True,
                       num_workers=4,
                       target_type="polynomial"):
    """
    Create training and validation data loaders.

    Args:
        data_file: Path to the data file.
        batch_size: Batch size.
        train_ratio: Train/val split ratio.
        normalize: Whether to standardize data.
        num_workers: Number of data-loading workers.
        target_type: "polynomial" (15-D coeffs) or "points" (12-D points).

    Returns:
        train_loader, val_loader, dataset.
    """
    dataset = TrajectoryDataset(data_file, normalize=normalize, target_type=target_type)
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Target type: {target_type}")
    
    return train_loader, val_loader, dataset

def get_data_stats(data_file=None):
    """Print basic statistics of the dataset."""
    dataset = TrajectoryDataset(data_file, normalize=False)
    
    conditions = dataset.conditions.numpy()
    trajectories = dataset.trajectory_points.numpy()
    
    print("=== Dataset statistics ===")
    print(f"Number of samples: {len(dataset)}")
    print(f"Condition dimension: {conditions.shape[1]}")
    print(f"Trajectory point dimension: {trajectories.shape[1]}")
    
    print("\nCondition stats (x, y, theta):")
    for i, name in enumerate(['x', 'y', 'theta']):
        values = conditions[:, i]
        print(f"  {name}: min={values.min():.4f}, max={values.max():.4f}, "
              f"mean={values.mean():.4f}, std={values.std():.4f}")


if __name__ == "__main__":
    get_data_stats(data_file="datasets/strike_processed.txt")
    
    train_loader, val_loader, dataset = create_data_loaders(
        batch_size=16, 
        train_ratio=0.8
    )
    
    for condition, target in train_loader:
        print("\nBatch example:")
        print(f"Condition shape: {condition.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Condition sample: {condition[0]}")
        print(f"Target sample (first 6 values): {target[0][:6]}...")
        break 