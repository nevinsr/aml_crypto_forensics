# Dynamic Graph Neural Networks vs. Gradient Boosted Decision Trees for Anti-Money Laundering

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amlcryptoforensics-3nxfdpjywgcaurysttcmhc.streamlit.app/)

An MSc-level machine learning and financial forensics system comparing **Graph Convolutional Networks (GCNs)** with **Gradient Boosted Decision Trees (LightGBM)** on the **Elliptic Bitcoin Dataset**.

---

## 🏛️ Project Architecture

```
mini-project (Anti-monet laundring sys)/
├── data/                             # Raw and processed Elliptic graph dataset
├── models/                           # Serialized trained model weights
│   ├── lgbm_baseline.pkl             # LightGBM trained model
│   └── gcn_weights.pt                # 2-layer PyG GCN model weights
├── requirements.txt                  # Python dependencies
├── data_loader.py                    # Phase 1: Ingestion, temporal split & masks
├── train_ml.py                       # Phase 2: LightGBM tabular baseline
├── train_gnn.py                      # Phase 3: 2-layer GCN ("Spiderweb Cop")
├── app.py                            # Phase 4: Streamlit forensic dashboard
└── README.md                         # Project documentation
```

---

## ⚙️ Hardware Optimization & Constraints

- **Execution Profile:** CPU-optimized execution designed for Intel Core i5 with 16GB RAM.
- **Multithreading:** LightGBM runs with `n_jobs=-1`.
- **PyTorch/PyG:** Utilizes CPU-optimized message passing and tensor operations.

---

## 🔬 Dataset & Strict Temporal Split Protocol

- **Dataset:** `EllipticBitcoinDataset` from `torch_geometric.datasets`.
- **Graph Dimensions:** 203,769 transactions (nodes), 234,355 payment flows (directed edges), 165 node features.
- **Label Filtering:** Unlabeled transactions (Class `2`) are masked out during training and evaluation.
- **Temporal Splitting (Out-of-Time / OOT):**
  - **Train Partition ($1 \le t \le 34$):** 29,894 labeled nodes (26,432 Licit, 3,462 Illicit).
  - **Test Partition ($35 \le t \le 49$):** 16,670 labeled nodes (15,587 Licit, 1,083 Illicit).

---

## 📊 Benchmark Results on Out-of-Time Test Set ($t \in [35, 49]$)

| Metric | LightGBM Tabular Baseline | GCN "Spiderweb Cop" |
| :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | **0.7999** | **0.4790** |
| **ROC-AUC Score** | **0.9301** | **0.8425** |
| **Minority-Class (Illicit) F1** | **0.7720** | **0.4735** |
| **Illicit Precision** | **0.8075** | **0.4238** |
| **Illicit Recall** | **0.7396** | **0.5365** |

---

## 🚀 How to Run

### 1. Ingestion & Split Verification (Phase 1)
```bash
python data_loader.py
```

### 2. Train Tabular Baseline (Phase 2)
```bash
python train_ml.py
```

### 3. Train Graph Neural Network (Phase 3)
```bash
python train_gnn.py
```

### 4. Launch Interactive Streamlit Dashboard (Phase 4)
```bash
streamlit run app.py
```
Open your browser at: `http://localhost:8501`

---

## 🛡️ Streamlit Dashboard Features

1. **Time-step Explorer:** Select discrete temporal snapshots ($35 \le t \le 49$).
2. **Transaction Filtering:** Filter transactions by category or use the **"🎲 Sample Random Suspicious Tx"** feature.
3. **Dual-Model Forensic Scoring:** Side-by-side metric cards displaying predicted illicit probabilities and risk levels for LightGBM vs. GCN.
4. **2-Hop Ego-Network Visualization:** Interactive physics-driven PyVis graph coloring the target node (Gold Diamond), known illicit nodes (Red), licit nodes (Green), and unlabeled nodes (Gray).
