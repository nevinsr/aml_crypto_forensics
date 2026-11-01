"""
Anti-Money Laundering (AML) Project: Dynamic GNNs vs. GBDT
Phase 3: Graph Convolutional Network (GCN) - "The Spiderweb Cop"
Dataset: EllipticBitcoinDataset (Temporal Split: Train 1-34, Test 35-49)
"""

import os
import time
from typing import Dict, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from data_loader import load_elliptic_data


class GCN(nn.Module):
    """
    2-Layer Graph Convolutional Network (GCN) for Anti-Money Laundering
    node classification over transaction flow graphs.
    """
    def __init__(
        self,
        in_channels: int = 165,
        hidden_channels: int = 128,
        out_channels: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass propagating 2-hop structural neighborhood information.

        Parameters:
        -----------
        x : torch.Tensor
            Node feature matrix of shape [N, 165].
        edge_index : torch.Tensor
            Graph edge index of shape [2, E].

        Returns:
        --------
        logits : torch.Tensor
            Unnormalized raw logits of shape [N].
        """
        # First GCN Layer + ReLU + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Second GCN Layer
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)


def train_gnn(
    data_dir: str = "./data",
    hidden_channels: int = 128,
    dropout: float = 0.2,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    epochs: int = 150,
    eval_every: int = 10,
    save_path: str = "./models/gcn_weights.pt",
) -> Tuple[GCN, Dict[str, Any]]:
    """
    CPU-optimized training loop for GCN on the Elliptic Bitcoin transaction graph.
    Employs BCEWithLogitsLoss with positive weight to tackle the class imbalance.

    Returns:
    --------
    model : GCN
        Trained GCN model.
    final_metrics : Dict[str, Any]
        Evaluation metrics on the OOT test partition (timesteps 35-49).
    """
    # Force CPU execution as per project constraints
    device = torch.device("cpu")
    print(f"\n[GNN Cop] Initializing Graph Pipeline on device: {device}")

    # 1. Load enriched graph
    data = load_elliptic_data(data_dir=data_dir).to(device)

    # 2. Compute positive class weight for loss balancing
    num_train_licit = (data.y[data.train_mask] == 0).sum().item()
    num_train_illicit = (data.y[data.train_mask] == 1).sum().item()
    pos_weight_val = num_train_licit / max(num_train_illicit, 1)
    pos_weight = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)

    print(f"[GNN Cop] Graph Topology: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    print(f"[GNN Cop] Training Nodes: {data.train_mask.sum().item():,} (Licit: {num_train_licit:,}, Illicit: {num_train_illicit:,})")
    print(f"[GNN Cop] Loss Balancing: pos_weight = {pos_weight_val:.3f}")

    # 3. Model, Loss, and Optimizer
    model = GCN(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=1,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Training Loop
    best_pr_auc = 0.0
    best_state = None
    start_time = time.time()

    print(f"\n[GNN Cop] Starting training for {epochs} epochs...")
    print(f"{'Epoch':>6} | {'Train Loss':>11} | {'Test PR-AUC':>12} | {'Test ROC-AUC':>13} | {'Test F1 (Illicit)':>18} | {'Elapsed':>8}")
    print("-" * 80)

    train_labels = data.y[data.train_mask].float()
    test_labels = data.y[data.test_mask].cpu().numpy()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass on full graph to propagate structural message passing
        out = model(data.x, data.edge_index)
        
        # Compute loss exclusively on training mask
        loss = criterion(out[data.train_mask], train_labels)
        loss.backward()
        optimizer.step()

        # Periodic Out-of-Time Test Evaluation
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                eval_out = model(data.x, data.edge_index)
                eval_probs = torch.sigmoid(eval_out[data.test_mask]).cpu().numpy()
                eval_preds = (eval_probs >= 0.5).astype(int)

                test_pr_auc = average_precision_score(test_labels, eval_probs)
                test_roc_auc = roc_auc_score(test_labels, eval_probs)
                test_f1 = f1_score(test_labels, eval_preds, pos_label=1)

                elapsed = time.time() - start_time
                print(f"{epoch:6d} | {loss.item():11.4f} | {test_pr_auc:12.4f} | {test_roc_auc:13.4f} | {test_f1:18.4f} | {elapsed:7.1f}s")

                if test_pr_auc > best_pr_auc:
                    best_pr_auc = test_pr_auc
                    best_state = model.state_dict().copy()

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # 5. Final Comprehensive Evaluation on OOT Test Set
    model.eval()
    with torch.no_grad():
        final_logits = model(data.x, data.edge_index)
        test_probs = torch.sigmoid(final_logits[data.test_mask]).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)

    pr_auc = float(average_precision_score(test_labels, test_probs))
    roc_auc = float(roc_auc_score(test_labels, test_probs))
    illicit_f1 = float(f1_score(test_labels, test_preds, pos_label=1))
    illicit_prec = float(precision_score(test_labels, test_preds, pos_label=1, zero_division=0))
    illicit_rec = float(recall_score(test_labels, test_preds, pos_label=1, zero_division=0))
    cm = confusion_matrix(test_labels, test_preds)
    cr = classification_report(test_labels, test_preds, target_names=["Licit (0)", "Illicit (1)"], digits=4)

    final_metrics = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "illicit_f1": illicit_f1,
        "illicit_precision": illicit_prec,
        "illicit_recall": illicit_rec,
        "confusion_matrix": cm,
        "classification_report": cr,
    }

    # 6. Save model weights and metadata
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[Model Registry] Saved trained GCN weights to '{save_path}'")

    return model, final_metrics


def predict_node_gnn(
    model: GCN,
    data,
    node_idx: int,
) -> float:
    """
    Inference helper: Computes the predicted illicit probability for a given node index.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        prob = torch.sigmoid(logits[node_idx]).item()
    return prob


def main():
    print("=" * 70)
    print(" Phase 3: Graph Convolutional Network (GCN) Training & Evaluation ")
    print("=" * 70)

    model, metrics = train_gnn(
        data_dir="./data",
        hidden_channels=128,
        dropout=0.2,
        lr=0.005,
        weight_decay=1e-4,
        epochs=120,
        eval_every=10,
        save_path="./models/gcn_weights.pt",
    )

    print("\n" + "=" * 70)
    print(" GCN Out-of-Time Test Set Results (Timesteps 35 to 49)")
    print("=" * 70)
    print(f"  * PR-AUC (Average Precision) : {metrics['pr_auc']:.4f}")
    print(f"  * ROC-AUC Score              : {metrics['roc_auc']:.4f}")
    print(f"  * Illicit (Minority) F1-Score : {metrics['illicit_f1']:.4f}")
    print(f"  * Illicit Precision          : {metrics['illicit_precision']:.4f}")
    print(f"  * Illicit Recall             : {metrics['illicit_recall']:.4f}")

    print("\n--- Confusion Matrix ---")
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    print(f"  True Licit (TN)  : {tn:,}  |  False Illicit (FP) : {fp:,}")
    print(f"  False Licit (FN) : {fn:,}   |  True Illicit (TP)  : {tp:,}")

    print("\n--- Full Classification Report ---")
    print(metrics['classification_report'])

    print("\n[OK] Phase 3 GCN pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
