"""
Graph construction for polymer chain frames.

Implements §8.3 pseudocode:

    Input: polymer frame r[1..N]
    nodes = []; edges = []
    for i in 1..N:
        node_feature = [position r[i], velocity v[i] if available,
                        force F[i] if available, bead_index, i/N,
                        local_coordination, bead_type, chain_index]
        nodes.append(node_feature)
    for i in 1..N-1:
        edge_feature = [distance(r[i], r[i+1]), relative_vector,
                        bond_flag=1, edge_type=1]
        edges.append((i, i+1, edge_feature))
    for all pairs (i,j) with |i-j| > 1:
        if distance(r[i], r[j]) < neighbor_cutoff:
            edge_feature = [distance(r[i], r[j]), relative_vector,
                            bond_flag=0, edge_type=2, cutoff_flag=1]
            edges.append((i, j, edge_feature))
    return graph(nodes, edges)

This module will be fully implemented in Month 4 (dataset engineering).
The structure is defined here for completeness.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def build_graph(
    positions: np.ndarray,
    forces: Optional[np.ndarray] = None,
    velocities: Optional[np.ndarray] = None,
    neighbor_cutoff: float = 2.5,
    chain_index: int = 0,
    bead_type: int = 0,
) -> Dict[str, Any]:
    """
    Build a graph representation of a polymer frame.

    Implements §8.3 pseudocode and §5 node/edge feature schemas.

    Parameters
    ----------
    positions : np.ndarray, shape (N, dim)
        Bead positions.
    forces : np.ndarray, optional, shape (N, dim)
        Per-bead forces.
    velocities : np.ndarray, optional, shape (N, dim)
        Per-bead velocities.
    neighbor_cutoff : float
        Cutoff distance for nonbonded edges.
    chain_index : int
        Chain index (future-proofing for multi-chain systems per §5).
    bead_type : int
        Bead/monomer type (future-proofing for heterogeneous chains per §5).

    Returns
    -------
    graph : dict
        Graph data with node_features, edge_index, edge_features.
    """
    N, dim = positions.shape

    # === Node features (§5, §8.3) ===
    node_features = []
    for i in range(N):
        feature = {
            "position": positions[i].tolist(),
            "bead_index": i,
            "normalized_chain_position": i / N,  # §5: "i/N"
            "local_coordination": _local_coordination(i, N),  # bonded-neighbor count
            "bead_type": bead_type,  # §5: future-proofing
            "chain_index": chain_index,  # §5: future-proofing
        }
        if forces is not None:
            feature["force"] = forces[i].tolist()
        if velocities is not None:
            feature["velocity"] = velocities[i].tolist()
        node_features.append(feature)

    # === Edge construction (§8.3) ===
    edge_index = []  # (source, target) pairs
    edge_features = []

    # Bonded edges: i--(i+1) for linear chain
    for i in range(N - 1):
        r_ij = positions[i + 1] - positions[i]
        dist = float(np.linalg.norm(r_ij))
        edge_feat = {
            "relative_vector": r_ij.tolist(),
            "distance": dist,
            "bond_flag": 1,
            "edge_type": 1,  # §5: 1=bonded
            "cutoff_flag": 0,
            "bond_order": 1,
            "spring_type": "harmonic",  # §5: future-proofing
        }
        # Add both directions for message passing
        edge_index.append((i, i + 1))
        edge_features.append(edge_feat)
        edge_index.append((i + 1, i))
        edge_features.append({
            **edge_feat,
            "relative_vector": (-r_ij).tolist(),
        })

    # Nonbonded edges: pairs with |i-j| > 1 within cutoff
    for i in range(N):
        for j in range(i + 2, N):  # |i-j| > 1
            r_ij = positions[j] - positions[i]
            dist = float(np.linalg.norm(r_ij))
            if dist < neighbor_cutoff:
                edge_feat = {
                    "relative_vector": r_ij.tolist(),
                    "distance": dist,
                    "bond_flag": 0,
                    "edge_type": 2,  # §5: 2=nonbonded-within-cutoff
                    "cutoff_flag": 1,
                    "bond_order": 0,
                    "spring_type": "none",
                }
                edge_index.append((i, j))
                edge_features.append(edge_feat)
                edge_index.append((j, i))
                edge_features.append({
                    **edge_feat,
                    "relative_vector": (-r_ij).tolist(),
                })

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "n_nodes": N,
        "n_edges": len(edge_index),
    }


def _local_coordination(i: int, N: int) -> int:
    """
    Compute bonded-neighbor count for bead i in a linear chain.

    End beads (i=0, i=N-1) have 1 bonded neighbor.
    Interior beads have 2 bonded neighbors.
    """
    if i == 0 or i == N - 1:
        return 1
    return 2
