import numpy as np
from typing import List, Dict, Any

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    torch = None
    Data = None

def frame_to_pyg_data(
    positions_t: np.ndarray,
    positions_t1: np.ndarray,
    neighbor_cutoff: float = 2.5,
    n_beads: int = 30,
) -> Any:
    """
    Converts a pair of consecutive trajectory frames into a PyTorch Geometric Data object.
    
    Node features (per bead):
    - position: (3,) float
    - bead_index: (1,) float (normalized i/N)
    
    Edge features (per edge, bidirectional):
    - displacement: (3,) float (r_j - r_i)
    - distance: (1,) float (|r_j - r_i|)
    - bond_type: (1,) float (1.0 for bonded, 0.0 for nonbonded)
    
    Target:
    - y: (N, 3) float (displacement = positions_t1 - positions_t)
    """
    if torch is None or Data is None:
        raise ImportError("PyTorch and PyTorch Geometric are required for this module.")
        
    positions_t = np.asarray(positions_t, dtype=np.float32)
    positions_t1 = np.asarray(positions_t1, dtype=np.float32)
    
    # Node features: position and normalized index
    node_pos = positions_t
    node_idx = np.arange(n_beads, dtype=np.float32) / float(n_beads)
    x = np.concatenate([node_pos, node_idx[:, None]], axis=1) # (N, 4)
    x = torch.tensor(x, dtype=torch.float)
    
    # Target (next step displacement)
    y = torch.tensor(positions_t1 - positions_t, dtype=torch.float)
    
    # Edges
    src_list = []
    dst_list = []
    edge_attr_list = []
    
    for i in range(n_beads):
        for j in range(n_beads):
            if i == j:
                continue
                
            dist_vec = positions_t[j] - positions_t[i]
            dist = np.linalg.norm(dist_vec)
            
            is_bond = abs(i - j) == 1
            is_nonbond = abs(i - j) > 1 and dist < neighbor_cutoff
            
            if is_bond or is_nonbond:
                src_list.append(i)
                dst_list.append(j)
                bond_type = 1.0 if is_bond else 0.0
                edge_attr_list.append([dist_vec[0], dist_vec[1], dist_vec[2], dist, bond_type])
                
    if len(src_list) > 0:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.float)
        
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.pos = torch.tensor(positions_t, dtype=torch.float)
    return data

def trajectory_to_pyg_data_list(
    trajectory: Dict[str, Any],
    neighbor_cutoff: float = 2.5,
) -> List[Any]:
    """
    Processes all consecutive frame pairs from a trajectory into a list of Data objects.
    """
    if torch is None or Data is None:
        raise ImportError("PyTorch and PyTorch Geometric are required for this module.")
        
    # Extract frames based on trajectory format
    if "frames" in trajectory:
        # JSON format
        frames = trajectory["frames"]
        positions = [np.array(f["positions"]) for f in frames]
    elif "positions" in trajectory:
        # NPZ format
        positions = trajectory["positions"]
    else:
        raise ValueError("Unknown trajectory format: neither 'frames' nor 'positions' key found.")
        
    n_frames = len(positions)
    if n_frames == 0:
        return []
        
    n_beads = len(positions[0])
    
    data_list = []
    for t in range(n_frames - 1):
        data = frame_to_pyg_data(
            positions[t],
            positions[t+1],
            neighbor_cutoff=neighbor_cutoff,
            n_beads=n_beads
        )
        data_list.append(data)
        
    return data_list
