"""
Anti-Money Laundering (AML) Project: Dynamic GNNs vs. GBDT
Phase 1: Data Ingestion, Temporal Splitting & Preprocessing
Dataset: EllipticBitcoinDataset (torch_geometric.datasets)
"""

import os
from typing import Dict, Tuple, Optional
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset


def load_elliptic_data(data_dir: str = "./data") -> Data:
    """
    Downloads and loads the Elliptic Bitcoin Dataset via PyTorch Geometric,
    enriches the Data object with raw transaction IDs and explicit time-steps,
    and constructs strict temporal masks for train (steps 1-34) and test (steps 35-49)
    excluding unlabeled nodes (class 2).

    Parameters:
    -----------
    data_dir : str
        Directory to download/cache the raw and processed Elliptic dataset.

    Returns:
    --------
    data : torch_geometric.data.Data
        PyG Data object containing:
            - x: Node features tensor [num_nodes, 165] (float32)
            - edge_index: Graph connectivity tensor [2, num_edges] (int64)
            - y: Ground-truth binary labels [num_nodes] (0: Licit, 1: Illicit, 2: Unknown)
            - timestep: Time-step for each node [num_nodes] (int64, values 1..49)
            - tx_id: Transaction IDs [num_nodes] (int64)
            - labeled_mask: Boolean mask indicating known nodes (y != 2)
            - train_mask: Boolean mask for training (labeled & timestep <= 34)
            - test_mask: Boolean mask for testing / OOT (labeled & timestep >= 35)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Download & load raw PyG dataset
    print(f"[DataLoader] Loading EllipticBitcoinDataset from '{data_dir}'...")
    dataset = EllipticBitcoinDataset(root=data_dir)
    data = dataset[0]

    # 2. Extract transaction IDs and time-steps from raw features CSV
    raw_features_path = os.path.join(data_dir, "raw", "elliptic_txs_features.csv")
    if not os.path.exists(raw_features_path):
        raise FileNotFoundError(
            f"Raw features file not found at '{raw_features_path}'. Ensure dataset downloaded correctly."
        )

    print(f"[DataLoader] Reading metadata (txId, timestep) from '{raw_features_path}'...")
    meta_df = pd.read_csv(raw_features_path, header=None, usecols=[0, 1])
    tx_ids = torch.tensor(meta_df[0].values, dtype=torch.long)
    timesteps = torch.tensor(meta_df[1].values, dtype=torch.long)

    # Attach enriched attributes to PyG Data object
    data.tx_id = tx_ids
    data.timestep = timesteps

    # 3. Create strict temporal and labeled masks
    # y encoding: 0 = Licit, 1 = Illicit, 2 = Unknown/Unlabeled
    labeled_mask = (data.y != 2)
    train_mask = labeled_mask & (data.timestep <= 34)
    test_mask = labeled_mask & (data.timestep >= 35)

    data.labeled_mask = labeled_mask
    data.train_mask = train_mask
    data.test_mask = test_mask

    return data


def get_tabular_data(
    data_dir: str = "./data"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function for Phase 2 (LightGBM baseline). Extracts tabular node feature
    matrices (X) and binary labels (y) based on the temporal train/test split,
    excluding unlabeled nodes.

    Returns:
    --------
    X_train : np.ndarray
        Features for labeled nodes in time-steps 1-34. Shape: [num_train, 165]
    y_train : np.ndarray
        Labels for labeled nodes in time-steps 1-34. Shape: [num_train] (0 or 1)
    X_test : np.ndarray
        Features for labeled nodes in time-steps 35-49. Shape: [num_test, 165]
    y_test : np.ndarray
        Labels for labeled nodes in time-steps 35-49. Shape: [num_test] (0 or 1)
    train_indices : np.ndarray
        Original node indices for train set.
    test_indices : np.ndarray
        Original node indices for test set.
    """
    data = load_elliptic_data(data_dir=data_dir)

    train_idx = torch.where(data.train_mask)[0].numpy()
    test_idx = torch.where(data.test_mask)[0].numpy()

    X = data.x.numpy()
    y = data.y.numpy()

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_test, y_test, train_idx, test_idx


def get_dataset_summary(data: Data) -> Dict[str, any]:
    """
    Computes summary statistics for the dataset, splits, and class distributions.
    """
    num_total_nodes = data.num_nodes
    num_total_edges = data.num_edges
    num_labeled = int(data.labeled_mask.sum().item())
    num_unlabeled = num_total_nodes - num_labeled

    train_total = int(data.train_mask.sum().item())
    train_illicit = int((data.y[data.train_mask] == 1).sum().item())
    train_licit = int((data.y[data.train_mask] == 0).sum().item())

    test_total = int(data.test_mask.sum().item())
    test_illicit = int((data.y[data.test_mask] == 1).sum().item())
    test_licit = int((data.y[data.test_mask] == 0).sum().item())

    return {
        "total_nodes": num_total_nodes,
        "total_edges": num_total_edges,
        "features_dim": data.num_features,
        "unlabeled_nodes": num_unlabeled,
        "labeled_nodes": num_labeled,
        "train_total": train_total,
        "train_licit": train_licit,
        "train_illicit": train_illicit,
        "train_illicit_pct": (train_illicit / train_total * 100) if train_total > 0 else 0,
        "test_total": test_total,
        "test_licit": test_licit,
        "test_illicit": test_illicit,
        "test_illicit_pct": (test_illicit / test_total * 100) if test_total > 0 else 0,
    }


if __name__ == "__main__":
    print("=" * 70)
    print(" Phase 1: Data Ingestion & Temporal Split Verification ")
    print("=" * 70)

    data = load_elliptic_data(data_dir="./data")
    summary = get_dataset_summary(data)

    print(f"\n--- Dataset Topology & Dimensions ---")
    print(f"Total Nodes (Transactions):    {summary['total_nodes']:,}")
    print(f"Total Directed Edges (Flows):   {summary['total_edges']:,}")
    print(f"Node Feature Dimension:        {summary['features_dim']}")
    print(f"Unlabeled Nodes (Class 2):     {summary['unlabeled_nodes']:,} ({(summary['unlabeled_nodes'] / summary['total_nodes'] * 100):.2f}%)")
    print(f"Labeled Nodes (Class 0 & 1):   {summary['labeled_nodes']:,} ({(summary['labeled_nodes'] / summary['total_nodes'] * 100):.2f}%)")

    print(f"\n--- Strict Temporal Split Breakdown ---")
    print(f"Train Set (Time-steps 1 to 34):")
    print(f"  - Total Labeled Nodes:       {summary['train_total']:,}")
    print(f"  - Licit Nodes (Class 0):     {summary['train_licit']:,} ({(100 - summary['train_illicit_pct']):.2f}%)")
    print(f"  - Illicit Nodes (Class 1):   {summary['train_illicit']:,} ({summary['train_illicit_pct']:.2f}%)")

    print(f"\nTest Set - Out-of-Time (Time-steps 35 to 49):")
    print(f"  - Total Labeled Nodes:       {summary['test_total']:,}")
    print(f"  - Licit Nodes (Class 0):     {summary['test_licit']:,} ({(100 - summary['test_illicit_pct']):.2f}%)")
    print(f"  - Illicit Nodes (Class 1):   {summary['test_illicit']:,} ({summary['test_illicit_pct']:.2f}%)")

    # Sanity checks
    assert data.x.shape == (203769, 165), "Node feature matrix shape mismatch!"
    assert data.edge_index.shape[0] == 2, "Edge index dimension mismatch!"
    assert summary['train_total'] + summary['test_total'] == summary['labeled_nodes'], "Mask sum mismatch!"
    assert data.train_mask.dtype == torch.bool, "train_mask must be boolean!"
    assert data.test_mask.dtype == torch.bool, "test_mask must be boolean!"

    print("\n[OK] Phase 1 sanity checks passed successfully!")
    print("=" * 70)
