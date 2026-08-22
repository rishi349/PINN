import os
import glob
import numpy as np
from typing import Tuple, List, Optional, Callable, Any, Dict

try:
    import torch
    from torch_geometric.data import InMemoryDataset, Data
    from torch_geometric.loader import DataLoader
except ImportError:
    torch = None
    InMemoryDataset = object
    Data = None
    DataLoader = None

from src.data.graph_construction import trajectory_to_pyg_data_list


def assert_no_leakage(
    splits: Dict[str, List[str]],
) -> None:
    """Assert zero data leakage across train/val/test splits.

    Checks that no trajectory file appears in more than one split.
    Raises AssertionError with a clear message if leakage is found.

    Parameters
    ----------
    splits : dict mapping split name -> list of file paths
        e.g. {'train': [...], 'val': [...], 'test': [...]}

    Raises
    ------
    AssertionError
        If any file appears in more than one split.
    """
    seen: Dict[str, str] = {}  # filepath -> split name
    leaks = []

    for split_name, file_list in splits.items():
        for fp in file_list:
            norm_fp = os.path.abspath(fp)
            if norm_fp in seen:
                leaks.append(
                    f"'{norm_fp}' appears in both '{seen[norm_fp]}' and '{split_name}'"
                )
            else:
                seen[norm_fp] = split_name

    assert len(leaks) == 0, (
        f"Data leakage detected! {len(leaks)} file(s) appear in multiple splits:\n"
        + "\n".join(leaks)
    )


class PolymerDataset(InMemoryDataset):
    """
    PyG InMemoryDataset for polymer chain trajectories.
    Splits are performed by trajectory, never by frame, to prevent data leakage.
    """
    def __init__(self, root: str, trajectories_dir: str, neighbor_cutoff: float = 2.5, 
                 split: str = 'train', split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), 
                 seed: int = 42, transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None):
        if torch is None or InMemoryDataset is object:
            raise ImportError("PyTorch and PyTorch Geometric are required for this module.")

        self.trajectories_dir = trajectories_dir
        self.neighbor_cutoff = neighbor_cutoff
        self.split = split
        self.split_ratios = split_ratios
        self.seed = seed

        super().__init__(root, transform, pre_transform)

        # Load the requested split
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        # We handle scanning dynamically in process()
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def process(self):
        # Dynamically import I/O functions to avoid circular deps if they exist
        try:
            from src.data.trajectory import load_trajectory_json, load_trajectory_npz
        except ImportError:
            import json
            def load_trajectory_json(path):
                with open(path, 'r') as f:
                    return json.load(f)
            def load_trajectory_npz(path):
                data = np.load(path, allow_pickle=True)
                return dict(data)

        # Find all trajectory files
        json_files = sorted(glob.glob(os.path.join(self.trajectories_dir, "*.json")))
        npz_files = sorted(glob.glob(os.path.join(self.trajectories_dir, "*.npz")))

        # Prefer JSON if available, otherwise NPZ
        traj_files = json_files if len(json_files) > 0 else npz_files

        if not traj_files:
            raise FileNotFoundError(f"No trajectory files found in {self.trajectories_dir}")

        # Deterministic shuffle for splitting by trajectory
        rng = np.random.RandomState(self.seed)
        traj_files_shuffled = traj_files.copy()
        rng.shuffle(traj_files_shuffled)

        n_trajs = len(traj_files_shuffled)
        n_train = int(n_trajs * self.split_ratios[0])
        n_val = int(n_trajs * self.split_ratios[1])

        splits = {
            'train': traj_files_shuffled[:n_train],
            'val': traj_files_shuffled[n_train:n_train+n_val],
            'test': traj_files_shuffled[n_train+n_val:]
        }

        # Automated leakage check — required by §4 Month 4 gate
        assert_no_leakage(splits)

        # Process and save each split
        for i, split_name in enumerate(['train', 'val', 'test']):
            data_list = []
            for filepath in splits[split_name]:
                if filepath.endswith('.json'):
                    traj = load_trajectory_json(filepath)
                else:
                    traj = load_trajectory_npz(filepath)

                traj_data_list = trajectory_to_pyg_data_list(traj, self.neighbor_cutoff)
                data_list.extend(traj_data_list)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])

def create_dataloaders(
    root: str, trajectories_dir: str, batch_size: int = 32,
    neighbor_cutoff: float = 2.5, seed: int = 42,
) -> Tuple[Any, Any, Any]:
    """
    Creates DataLoaders for train, validation, and test splits.
    """
    if DataLoader is None:
        raise ImportError("PyTorch Geometric is required for DataLoaders.")

    train_dataset = PolymerDataset(root, trajectories_dir, neighbor_cutoff=neighbor_cutoff, split='train', seed=seed)
    val_dataset = PolymerDataset(root, trajectories_dir, neighbor_cutoff=neighbor_cutoff, split='val', seed=seed)
    test_dataset = PolymerDataset(root, trajectories_dir, neighbor_cutoff=neighbor_cutoff, split='test', seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
