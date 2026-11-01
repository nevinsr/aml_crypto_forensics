"""
Anti-Money Laundering (AML) Project: Dynamic GNNs vs. GBDT
Phase 4: Streamlit Forensic Decision-Support Dashboard
Interactive Subgraph Visualizer (NetworkX + PyVis) & Real-time Dual-Model Scoring
"""

import os
from typing import Tuple, Dict, Any, Optional
import streamlit as st
import streamlit.components.v1 as components
import torch
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network
from torch_geometric.utils import k_hop_subgraph, to_undirected

from data_loader import load_elliptic_data
from train_gnn import GCN

# Page configuration
st.set_page_config(
    page_title="AML Graph Forensics: GNN vs. GBDT",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    /* Keep header visible for sidebar toggle and top-right options/theme menu */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 100 !important;
    }
    
    /* Clean, balanced top spacing for main dashboard */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 96% !important;
    }

    /* Clean sidebar spacing while keeping collapse/expand controls fully accessible */
    [data-testid="stSidebarHeader"] {
        padding-top: 0.8rem !important;
        padding-bottom: 0rem !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 0.8rem !important;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text-color, inherit) !important;
        margin-top: 0rem;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: var(--text-color, inherit) !important;
        opacity: 0.75;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.08)) !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        color: var(--text-color, inherit) !important;
    }
    .risk-badge-critical {
        background-color: rgba(239, 68, 68, 0.18) !important;
        color: #EF4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.35) !important;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: rgba(245, 158, 11, 0.18) !important;
        color: #F59E0B !important;
        border: 1px solid rgba(245, 158, 11, 0.35) !important;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: rgba(16, 185, 129, 0.18) !important;
        color: #10B981 !important;
        border: 1px solid rgba(16, 185, 129, 0.35) !important;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .legend-box {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        padding: 10px 14px;
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.08)) !important;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: var(--text-color, inherit) !important;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .color-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Elliptic Bitcoin Graph & Metadata...")
def get_cached_graph_data():
    """Caches the full PyG graph data in memory for real-time dashboard interactions."""
    data = load_elliptic_data(data_dir="./data")
    # Pre-build undirected edge index for bidirectional neighborhood discovery
    undirected_edge_index = to_undirected(data.edge_index)
    
    # Precompute fast lookups
    node_indices = np.arange(data.num_nodes)
    tx_ids = data.tx_id.numpy()
    timesteps = data.timestep.numpy()
    labels = data.y.numpy()
    test_mask = data.test_mask.numpy()

    df_nodes = pd.DataFrame({
        "node_idx": node_indices,
        "tx_id": tx_ids,
        "timestep": timesteps,
        "label": labels,
        "is_test": test_mask,
    })

    return data, undirected_edge_index, df_nodes


@st.cache_resource(show_spinner="Loading LightGBM Tabular Baseline Model...")
def get_cached_lgbm_model():
    """Loads the pre-trained LightGBM model from disk."""
    model_path = "./models/lgbm_baseline.pkl"
    if not os.path.exists(model_path):
        st.error(f"LightGBM model not found at '{model_path}'. Run 'python train_ml.py' first.")
        st.stop()
    return joblib.load(model_path)


@st.cache_resource(show_spinner="Loading GCN Spiderweb Model...")
def get_cached_gcn_model():
    """Loads the pre-trained 2-layer GCN weights from disk."""
    weights_path = "./models/gcn_weights.pt"
    if not os.path.exists(weights_path):
        st.error(f"GCN weights not found at '{weights_path}'. Run 'python train_gnn.py' first.")
        st.stop()
    
    model = GCN(in_channels=165, hidden_channels=128, out_channels=1, dropout=0.2)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def generate_subgraph_html(
    data,
    undirected_edge_index,
    target_node_idx: int,
    num_hops: int = 2,
    max_nodes: int = 80,
) -> Tuple[str, Dict[str, int]]:
    """
    Extracts the k-hop ego subgraph around the target node and renders an
    interactive PyVis HTML visualization.
    """
    # 1. Extract k-hop neighborhood using PyG utility
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=target_node_idx,
        num_hops=num_hops,
        edge_index=undirected_edge_index,
        relabel_nodes=False,
    )

    subset_list = subset.tolist()
    # Cap size if ego network is excessively dense to ensure snappy browser rendering
    if len(subset_list) > max_nodes:
        # Prioritize target node and immediate 1-hop neighbors
        subset_1hop, _, _, _ = k_hop_subgraph(
            node_idx=target_node_idx,
            num_hops=1,
            edge_index=undirected_edge_index,
            relabel_nodes=False,
        )
        subset_1hop_set = set(subset_1hop.tolist())
        remaining = [idx for idx in subset_list if idx not in subset_1hop_set]
        subset_list = list(subset_1hop_set) + remaining[:(max_nodes - len(subset_1hop_set))]

    subset_set = set(subset_list)

    # 2. Extract induced original directed edges for payment flow visualization
    src, dst = data.edge_index[0], data.edge_index[1]
    induced_mask = torch.tensor([s in subset_set and d in subset_set for s, d in zip(src.tolist(), dst.tolist())], dtype=torch.bool)
    induced_src = src[induced_mask].tolist()
    induced_dst = dst[induced_mask].tolist()

    # 3. Create interactive PyVis Network
    net = Network(
        height="540px",
        width="100%",
        directed=True,
        bgcolor="#1E293B",
        font_color="#F8FAFC",
    )

    # Physics configuration for clean aesthetic layout
    net.set_options(
        """
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -45,
              "centralGravity": 0.015,
              "springLength": 85,
              "springConstant": 0.08,
              "damping": 0.8
            },
            "solver": "forceAtlas2Based",
            "stabilization": { "iterations": 80 }
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
            "color": { "color": "#475569", "highlight": "#38BDF8" },
            "smooth": { "type": "continuous" }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true
          }
        }
        """
    )

    stats = {"total": len(subset_list), "target": 1, "illicit": 0, "licit": 0, "unknown": 0, "edges": len(induced_src)}

    # Add Nodes
    for orig_idx in subset_list:
        is_target = (orig_idx == target_node_idx)
        label_val = data.y[orig_idx].item()
        tx_id_val = data.tx_id[orig_idx].item()
        t_val = data.timestep[orig_idx].item()

        if is_target:
            color = "#FFD700"  # Target: Vibrant Gold / Yellow
            border_color = "#FFFFFF"
            size = 28
            shape = "diamond"
            label_text = f"TARGET: {tx_id_val}"
        elif label_val == 1:
            color = "#EF4444"  # Illicit: Vibrant Red
            border_color = "#B91C1C"
            size = 18
            shape = "dot"
            label_text = f"Illicit: {tx_id_val}"
            stats["illicit"] += 1
        elif label_val == 0:
            color = "#10B981"  # Licit: Vibrant Green
            border_color = "#047857"
            size = 14
            shape = "dot"
            label_text = f"Licit: {tx_id_val}"
            stats["licit"] += 1
        else:
            color = "#64748B"  # Unknown: Slate Gray
            border_color = "#334155"
            size = 11
            shape = "dot"
            label_text = f"Unlabeled"
            stats["unknown"] += 1

        label_name = "Target (Selected)" if is_target else ("Illicit (Fraud)" if label_val == 1 else ("Licit (Legit)" if label_val == 0 else "Unknown"))
        title_tooltip = f"TxID: {tx_id_val}\nType: {label_name}\nTime-step: {t_val}\nNode Index: {orig_idx}"

        net.add_node(
            orig_idx,
            label=label_text,
            title=title_tooltip,
            color={"background": color, "border": border_color, "highlight": {"background": "#F59E0B", "border": "#FFFFFF"}},
            size=size,
            shape=shape,
            borderWidth=2 if is_target else 1,
            font={"size": 11, "face": "Helvetica", "color": "#F1F5F9"},
        )

    # Add Directed Payment Flow Edges
    for s, d in zip(induced_src, induced_dst):
        net.add_edge(s, d)

    html_code = net.generate_html()
    return html_code, stats


def main():
    # Load backend resources
    data, undirected_edge_index, df_nodes = get_cached_graph_data()
    lgbm_model = get_cached_lgbm_model()
    gcn_model = get_cached_gcn_model()

    # Pre-filter Out-of-Time Test Set (Timesteps 35 to 49)
    df_test = df_nodes[df_nodes["is_test"]].copy()

    # -------------------------------------------------------------
    # SIDEBAR: Transaction Selection & Investigation Filters
    # -------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; padding: 6px 0;">
                <span style="font-size: 2.2rem; line-height: 1;">🛡️</span>
                <div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: var(--text-color, inherit); line-height: 1.2;">AML Forensics</div>
                    <div style="font-size: 0.8rem; color: var(--text-color, inherit); opacity: 0.75;">Decision Support System</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(r"Select a transaction from the **Out-of-Time Test Set** ($t \in [35, 49]$) to investigate.")

        # Timestep selector (35 to 49)
        available_timesteps = sorted(df_test["timestep"].unique().tolist())
        selected_timestep = st.selectbox(
            "📅 Select Time-step (OOT Test):",
            options=available_timesteps,
            index=0,
            help="Out-of-Time evaluation strictly uses unseen time-steps 35 to 49.",
        )

        df_t = df_test[df_test["timestep"] == selected_timestep]

        # Filter by class category
        filter_option = st.radio(
            "🎯 Filter Transactions By:",
            options=["All Labeled", "Illicit Only (Suspicious)", "Licit Only (Legit)"],
            index=1,
        )

        if filter_option == "Illicit Only (Suspicious)":
            df_filtered = df_t[df_t["label"] == 1]
        elif filter_option == "Licit Only (Legit)":
            df_filtered = df_t[df_t["label"] == 0]
        else:
            df_filtered = df_t

        if len(df_filtered) == 0:
            st.warning(f"No transactions found with selected filter in time-step {selected_timestep}. Showing all labeled.")
            df_filtered = df_t

        # Random illicit sample quick action button
        if st.button("🎲 Sample Random Suspicious Tx"):
            illicit_samples = df_test[df_test["label"] == 1]
            random_row = illicit_samples.sample(1).iloc[0]
            st.session_state["selected_node_idx"] = int(random_row["node_idx"])
            st.session_state["selected_timestep"] = int(random_row["timestep"])

        # Format selectbox options
        tx_options = df_filtered["node_idx"].tolist()
        tx_labels = {
            idx: f"Tx #{df_nodes.loc[idx, 'tx_id']} ({'Illicit' if df_nodes.loc[idx, 'label'] == 1 else 'Licit'})"
            for idx in tx_options
        }

        default_idx = tx_options[0]
        if "selected_node_idx" in st.session_state and st.session_state["selected_node_idx"] in tx_options:
            default_idx = st.session_state["selected_node_idx"]

        selected_node_idx = st.selectbox(
            "🔍 Target Transaction ID:",
            options=tx_options,
            format_func=lambda x: tx_labels[x],
            index=tx_options.index(default_idx) if default_idx in tx_options else 0,
        )

        # Subgraph parameter controls
        st.markdown("---")
        st.markdown("#### **Subgraph Visualization Settings**")
        num_hops = st.slider("Neighborhood Radius (Hops):", min_value=1, max_value=2, value=2)
        max_subgraph_nodes = st.slider("Max Subgraph Display Nodes:", min_value=20, max_value=120, value=75)

    # -------------------------------------------------------------
    # MAIN PANEL: Header & Intelligence Summary
    # -------------------------------------------------------------
    st.markdown('<div class="main-title">Dynamic Graph Neural Networks vs. Gradient Boosted Decision Trees</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Forensic transaction analysis, ego-network visualization, and dual-model fraud probability scoring.</div>', unsafe_allow_html=True)

    target_tx_id = int(data.tx_id[selected_node_idx].item())
    target_timestep = int(data.timestep[selected_node_idx].item())
    target_ground_truth = int(data.y[selected_node_idx].item())

    # Target metadata summary banner
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric("Target Transaction ID", f"{target_tx_id}")
    with col_t2:
        st.metric("Time-step Snapshot", f"t = {target_timestep} (OOT)")
    with col_t3:
        gt_label_str = "ILLICIT (FRAUD)" if target_ground_truth == 1 else "LICIT (LEGIT)"
        st.metric("Ground Truth Label", gt_label_str)
    with col_t4:
        # Degree calculation
        in_deg = int((data.edge_index[1] == selected_node_idx).sum().item())
        out_deg = int((data.edge_index[0] == selected_node_idx).sum().item())
        st.metric("Transaction Degree", f"{in_deg + out_deg} (In: {in_deg}, Out: {out_deg})")

    st.markdown("---")

    # -------------------------------------------------------------
    # DUAL-MODEL SCORING PANEL (Decision Support Metric Cards)
    # -------------------------------------------------------------
    st.markdown("### ⚡ **Dual-Model Forensic Scoring Panel**")

    # 1. Run LightGBM tabular inference
    target_features = data.x[selected_node_idx].unsqueeze(0).numpy()
    lgbm_prob = float(lgbm_model.predict_proba(target_features)[0, 1])

    # 2. Run GCN structural inference
    with torch.no_grad():
        all_logits = gcn_model(data.x, data.edge_index)
        gcn_prob = float(torch.sigmoid(all_logits[selected_node_idx]).item())

    # Display side-by-side metric cards
    score_col1, score_col2, score_col3 = st.columns([1.2, 1.2, 1.6])

    with score_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; font-weight:600; color:var(--text-color, inherit); opacity:0.75;">TABULAR BASELINE (LightGBM)</div>
                <div style="font-size:2rem; font-weight:700; color:var(--text-color, inherit); margin: 6px 0;">{lgbm_prob * 100:.1f}%</div>
                <div style="font-size:0.85rem; color:var(--text-color, inherit); opacity:0.8;">Predicted Illicit Probability</div>
                <div style="margin-top:10px;">
                    <span class="{'risk-badge-critical' if lgbm_prob >= 0.75 else ('risk-badge-high' if lgbm_prob >= 0.50 else 'risk-badge-low')}">
                        {'CRITICAL RISK' if lgbm_prob >= 0.75 else ('HIGH RISK' if lgbm_prob >= 0.50 else 'LOW RISK')}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with score_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; font-weight:600; color:var(--text-color, inherit); opacity:0.75;">SPIDERWEB GNN (2-Layer GCN)</div>
                <div style="font-size:2rem; font-weight:700; color:var(--text-color, inherit); margin: 6px 0;">{gcn_prob * 100:.1f}%</div>
                <div style="font-size:0.85rem; color:var(--text-color, inherit); opacity:0.8;">Predicted Illicit Probability</div>
                <div style="margin-top:10px;">
                    <span class="{'risk-badge-critical' if gcn_prob >= 0.75 else ('risk-badge-high' if gcn_prob >= 0.50 else 'risk-badge-low')}">
                        {'CRITICAL RISK' if gcn_prob >= 0.75 else ('HIGH RISK' if gcn_prob >= 0.50 else 'LOW RISK')}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with score_col3:
        # Model consensus and intelligence diagnostic
        both_flag = (lgbm_prob >= 0.5) and (gcn_prob >= 0.5)
        both_clear = (lgbm_prob < 0.5) and (gcn_prob < 0.5)
        gnn_elevated = (gcn_prob >= 0.5) and (lgbm_prob < 0.5)
        lgbm_elevated = (lgbm_prob >= 0.5) and (gcn_prob < 0.5)

        if both_flag:
            consensus_badge = "🔴 STRONG CONSENSUS: ILLICIT"
            consensus_desc = "Both the tabular decision trees and graph neural network flag this transaction as highly suspicious."
        elif both_clear:
            consensus_badge = "🟢 STRONG CONSENSUS: LICIT"
            consensus_desc = "Both models concordantly classify this transaction as standard legitimate commercial activity."
        elif gnn_elevated:
            consensus_badge = "🟡 GRAPH-TRIGGERED RISK"
            consensus_desc = "GNN elevates risk above baseline, detecting structural contamination in the surrounding 2-hop transaction neighborhood."
        else:
            consensus_badge = "🟡 TABULAR-TRIGGERED RISK"
            consensus_desc = "LightGBM flags anomalous local transaction features (e.g. abnormal volume or fee distribution)."

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; font-weight:600; color:var(--text-color, inherit); opacity:0.75;">MODEL CONSENSUS & DIAGNOSTIC</div>
                <div style="font-size:1.15rem; font-weight:700; color:var(--text-color, inherit); margin: 8px 0;">{consensus_badge}</div>
                <div style="font-size:0.85rem; color:var(--text-color, inherit); opacity:0.85; line-height: 1.4;">{consensus_desc}</div>
                <div style="margin-top:10px; font-size:0.8rem; color:var(--text-color, inherit); opacity:0.75;">
                    Probability Delta |GBDT - GNN|: <b>{abs(lgbm_prob - gcn_prob) * 100:.1f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -------------------------------------------------------------
    # INTERACTIVE SUBGRAPH VISUALIZER (NetworkX + PyVis)
    # -------------------------------------------------------------
    st.markdown("### 🕸️ **Interactive Subgraph Visualizer (2-Hop Ego-Network)**")
    
    # Legend
    st.markdown(
        """
        <div class="legend-box">
            <div class="legend-item"><span class="color-dot" style="background:#FFD700;"></span> <b>Target Transaction (Gold Diamond)</b></div>
            <div class="legend-item"><span class="color-dot" style="background:#EF4444;"></span> <b>Known Illicit Flow (Red)</b></div>
            <div class="legend-item"><span class="color-dot" style="background:#10B981;"></span> <b>Known Licit Flow (Green)</b></div>
            <div class="legend-item"><span class="color-dot" style="background:#64748B;"></span> <b>Unlabeled Transaction (Gray)</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Generate PyVis HTML
    with st.spinner("Generating 2-hop transaction network graph..."):
        subgraph_html, stats = generate_subgraph_html(
            data=data,
            undirected_edge_index=undirected_edge_index,
            target_node_idx=selected_node_idx,
            num_hops=num_hops,
            max_nodes=max_subgraph_nodes,
        )

    # Render HTML in Streamlit
    components.html(subgraph_html, height=560)

    # Subgraph statistics summary
    sub_c1, sub_c2, sub_c3, sub_c4, sub_c5 = st.columns(5)
    with sub_c1:
        st.caption(f"**Total Subgraph Nodes:** {stats['total']}")
    with sub_c2:
        st.caption(f"**Connected Flow Edges:** {stats['edges']}")
    with sub_c3:
        st.caption(f"**🔴 Illicit Neighbors:** {stats['illicit']}")
    with sub_c4:
        st.caption(f"**🟢 Licit Neighbors:** {stats['licit']}")
    with sub_c5:
        st.caption(f"**⚪ Unlabeled Neighbors:** {stats['unknown']}")

    st.markdown("---")

    # -------------------------------------------------------------
    # BENCHMARK COMPARISON & MODEL INSIGHTS EXPANDER
    # -------------------------------------------------------------
    with st.expander("📊 Global Benchmark Comparison on Out-of-Time Test Partition (16,670 Transactions)"):
        st.markdown(
            """
            This table compares the Out-of-Time (OOT) generalization metrics on time-steps 35 to 49:
            """
        )
        benchmark_df = pd.DataFrame({
            "Metric": ["PR-AUC (Average Precision)", "ROC-AUC", "Minority F1-Score (Illicit)", "Illicit Precision", "Illicit Recall"],
            "LightGBM (Tabular GBDT)": ["0.7999", "0.9301", "0.7720", "0.8075", "0.7396"],
            "GCN (Spiderweb Cop)": ["0.4790", "0.8425", "0.4735", "0.4238", "0.5365"],
            "Forensic Advantage": [
                "GBDT excels on local transaction feature aggregates",
                "Strong overall ranking across licit/illicit transactions",
                "GBDT exhibits higher precision with fewer false alarms",
                "GBDT avoids false positives on high-degree mixing hubs",
                "GCN detects flow contagion across interconnected wallets",
            ]
        })
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
