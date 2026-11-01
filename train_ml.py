"""
Anti-Money Laundering (AML) Project: Dynamic GNNs vs. GBDT
Phase 2: Tabular ML Baseline (LightGBM)
Dataset: EllipticBitcoinDataset (Temporal Split: Train 1-34, Test 35-49)
"""

import os
import joblib
from typing import Dict, Tuple, Any
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from data_loader import get_tabular_data


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
) -> lgb.LGBMClassifier:
    """
    Initializes and trains a CPU-optimized LightGBM Classifier with balanced class weights
    to combat the severe class imbalance in illicit transaction detection.

    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix of shape [N_train, 165].
    y_train : np.ndarray
        Training labels (0: Licit, 1: Illicit) of shape [N_train].
    random_state : int
        Seed for reproducibility.
    n_estimators : int
        Number of boosted decision trees.
    learning_rate : float
        Boosting learning rate.
    max_depth : int
        Maximum tree depth for regularized decision trees.
    num_leaves : int
        Maximum tree leaves for base learners.

    Returns:
    --------
    model : lgb.LGBMClassifier
        Fitted LightGBM model.
    """
    # Calculate empirical class ratio for transparency
    num_licit = np.sum(y_train == 0)
    num_illicit = np.sum(y_train == 1)
    imbalance_ratio = num_licit / max(num_illicit, 1)

    print(f"\n[LightGBM] Training Configuration:")
    print(f"  - Total Training Samples: {len(y_train):,}")
    print(f"  - Licit (Class 0):        {num_licit:,} ({(num_licit / len(y_train) * 100):.2f}%)")
    print(f"  - Illicit (Class 1):      {num_illicit:,} ({(num_illicit / len(y_train) * 100):.2f}%)")
    print(f"  - Imbalance Ratio (0:1):  {imbalance_ratio:.2f}:1")
    print(f"  - Class Weight Strategy:  'balanced' (auto-weighted gradient boosting)")
    print(f"  - Hyperparameters:        n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}, num_leaves={num_leaves}")

    # Build CPU-optimized classifier
    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,  # Utilize all available CPU cores
        random_state=random_state,
        verbose=-1,
    )

    print("\n[LightGBM] Fitting model on CPU...")
    model.fit(X_train, y_train)
    print("[LightGBM] Training complete!")
    return model


def evaluate_tabular_model(
    model: lgb.LGBMClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates model performance on the Out-of-Time (OOT) test set (time-steps 35-49).
    Focuses on imbalanced classification metrics: PR-AUC, ROC-AUC, and Illicit F1-Score.

    Parameters:
    -----------
    model : lgb.LGBMClassifier
        Trained classifier.
    X_test : np.ndarray
        Test features of shape [N_test, 165].
    y_test : np.ndarray
        Ground-truth test labels (0: Licit, 1: Illicit).
    threshold : float
        Decision threshold for positive class (default: 0.5).

    Returns:
    --------
    metrics : Dict[str, Any]
        Dictionary of computed performance metrics.
    """
    # Predict probabilities for positive class (Class 1: Illicit)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Core MSc Evaluation Metrics
    pr_auc = float(average_precision_score(y_test, y_prob))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    illicit_f1 = float(f1_score(y_test, y_pred, pos_label=1))
    illicit_prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    illicit_rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Licit (0)", "Illicit (1)"], digits=4)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "illicit_f1": illicit_f1,
        "illicit_precision": illicit_prec,
        "illicit_recall": illicit_rec,
        "confusion_matrix": cm,
        "classification_report": cr,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def save_model(model: Any, filepath: str = "./models/lgbm_baseline.pkl") -> None:
    """
    Serializes and saves the trained model to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[Model Registry] Saved trained model to '{filepath}'")


def main():
    print("=" * 70)
    print(" Phase 2: Tabular ML Baseline Training & Out-of-Time Evaluation ")
    print("=" * 70)

    # 1. Load tabular data partition
    X_train, y_train, X_test, y_test, train_idx, test_idx = get_tabular_data(data_dir="./data")

    # 2. Train LightGBM model
    model = train_lightgbm(X_train, y_train)

    # 3. Evaluate on OOT Test Set
    metrics = evaluate_tabular_model(model, X_test, y_test, threshold=0.5)

    print("\n" + "=" * 70)
    print(" Out-of-Time Test Set Evaluation Results (Timesteps 35 to 49)")
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

    # 4. Save trained model
    save_model(model, filepath="./models/lgbm_baseline.pkl")

    print("\n[OK] Phase 2 tabular ML pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
