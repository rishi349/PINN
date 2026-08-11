import json
import os
from typing import Dict, Any, Optional
import numpy as np

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    torch = None
    Data = None

class DisplacementNormalizer:
    """
    Z-score normalization for displacement targets.
    
    Predictions should be made on normalized targets ((y - mean) / std),
    and then denormalized (pred * std + mean) before adding to positions.
    """
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = np.zeros(3, dtype=np.float32) if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = np.ones(3, dtype=np.float32) if std is None else np.asarray(std, dtype=np.float32)
        
        # Prevent division by zero if std is somehow exactly 0
        self.std = np.clip(self.std, a_min=1e-8, a_max=None)
        
    def fit(self, data_list) -> None:
        """
        Compute mean and std from a list of PyG Data objects.
        
        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            List containing the training graphs with .y attribute (displacements).
        """
        if torch is None:
            raise ImportError("PyTorch is required for fitting.")
            
        all_y = []
        for data in data_list:
            all_y.append(data.y.detach().cpu().numpy())
            
        if not all_y:
            return
            
        concat_y = np.concatenate(all_y, axis=0)
        self.mean = np.mean(concat_y, axis=0).astype(np.float32)
        self.std = np.std(concat_y, axis=0).astype(np.float32)
        self.std = np.clip(self.std, a_min=1e-8, a_max=None)
        
    def normalize(self, y: Any) -> Any:
        """Normalize targets: (y - mean) / std."""
        is_tensor = False
        if torch is not None and isinstance(y, torch.Tensor):
            is_tensor = True
            mean = torch.tensor(self.mean, device=y.device, dtype=y.dtype)
            std = torch.tensor(self.std, device=y.device, dtype=y.dtype)
        else:
            mean = self.mean
            std = self.std
            
        return (y - mean) / std
        
    def denormalize(self, y_norm: Any) -> Any:
        """Denormalize predictions: y_norm * std + mean."""
        is_tensor = False
        if torch is not None and isinstance(y_norm, torch.Tensor):
            is_tensor = True
            mean = torch.tensor(self.mean, device=y_norm.device, dtype=y_norm.dtype)
            std = torch.tensor(self.std, device=y_norm.device, dtype=y_norm.dtype)
        else:
            mean = self.mean
            std = self.std
            
        return y_norm * std + mean
        
    def save(self, filepath: str) -> None:
        """Save statistics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({
                "mean": self.mean.tolist(),
                "std": self.std.tolist()
            }, f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> "DisplacementNormalizer":
        """Load statistics from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(mean=np.array(data["mean"]), std=np.array(data["std"]))
