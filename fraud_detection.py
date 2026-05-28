# =============================================================================
# FINANCIAL FRAUD DETECTION PIPELINE USING XGBOOST
# =============================================================================
# Dataset   : Fraud.csv (~6.3M rows, 11 columns)
# Target    : isFraud  (0 = Normal, 1 = Fraudulent)
# Author    : Senior ML Engineer
# =============================================================================
#
# WHY XGBOOST FOR FRAUD DETECTION?
# ---------------------------------
# 1. Handles class imbalance via scale_pos_weight
# 2. Robust to outliers (common in transaction amounts)
# 3. Built-in feature importance — critical for interpretability in finance
# 4. Gradient boosting captures complex nonlinear patterns fraudsters exploit
# 5. Fast training even on millions of rows with tree_method='hist'
# 6. Works well on tabular data without heavy preprocessing
#
# WHY CLASS IMBALANCE MATTERS IN FRAUD DETECTION?
# ------------------------------------------------
# In real fraud datasets, fraud cases are < 0.5% of all transactions.
# A naive model predicting "all normal" gets 99.5% accuracy but catches 0 fraud.
# Imbalance-aware training forces the model to learn from rare fraud patterns.
#
# WHY RECALL IS THE CRITICAL METRIC?
# ------------------------------------
# FALSE NEGATIVE (missed fraud) = Bank loses money, customer harmed
# FALSE POSITIVE (flagged legitimate tx) = Customer inconvenienced, support cost
# Missing real fraud (false negative) is far more costly than a false alarm.
# Therefore, we MAXIMIZE RECALL even at some cost to precision.
#
# WHY ACCURACY IS MISLEADING?
# ----------------------------
# If 99.9% of transactions are legitimate, a model that always predicts 0
# achieves 99.9% accuracy but is completely useless for detecting fraud.
# We use ROC-AUC, F1-score, and Precision-Recall AUC instead.
# =============================================================================

import os
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for VS Code / server use
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "csv_path": "Fraud.csv",
    "target_col": "isFraud",
    "test_size": 0.20,
    "random_state": 42,
    "output_dir": "fraud_output",
    "sample_size": 500000,

    "xgb_params": {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 1,
    "min_child_weight": 5,
    "tree_method": "hist",
    "eval_metric": "aucpr",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "n_jobs": -1
}
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

os.makedirs(CONFIG["output_dir"], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(CONFIG["output_dir"], "pipeline.log")),
    ]
)
log = logging.getLogger(__name__)

# =============================================================================
# STEP 1 — DATA LOADING  EXPLORATION
# =============================================================================

# =============================================================================
# STEP 1 — DATA LOADING
# =============================================================================

def load_data(csv_path, sample_size=None):

    print("Loading dataset...")

    if sample_size:
        df = pd.read_csv(csv_path, nrows=sample_size)
    else:
        df = pd.read_csv(csv_path)

    print("Dataset Loaded Successfully!")

    return df


def explore_data(df: pd.DataFrame, target_col: str) -> None:
    """Print key dataset statistics to console."""
    log.info("=" * 60)
    log.info("DATASET EXPLORATION")
    log.info("=" * 60)
    log.info("Shape          : %s", df.shape)
    log.info("Columns        : %s", list(df.columns))
    log.info("\nData Types:\n%s", df.dtypes.to_string())
    log.info("\nMissing Values:\n%s", df.isnull().sum().to_string())

    fraud_counts = df[target_col].value_counts()
    fraud_pct    = df[target_col].value_counts(normalize=True) * 100
    log.info(
        "\nClass Distribution:\n  Normal (0): %d  (%.4f%%)\n  Fraud  (1): %d  (%.4f%%)",
        fraud_counts.get(0, 0), fraud_pct.get(0, 0),
        fraud_counts.get(1, 0), fraud_pct.get(1, 0),
    )
    log.info("\nDescriptive Statistics:\n%s", df.describe().to_string())


# =============================================================================
# STEP 2 — EDA VISUALIZATIONS
# =============================================================================

def generate_eda_plots(df: pd.DataFrame, target_col: str, output_dir: str) -> None:
    """
    Generate and save EDA plots:
      - Fraud class countplot
      - Correlation heatmap
      - Transaction amount distribution
    """
    log.info("Generating EDA visualizations ...")

    # --- 1. Fraud vs Non-Fraud Countplot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df[target_col].value_counts()
    bars   = ax.bar(
        ["Normal (0)", "Fraud (1)"],
        counts.values,
        color=["steelblue", "crimson"],
        edgecolor="black"
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{int(bar.get_height()):,}",
            ha="center", fontsize=11
        )
    ax.set_title("Fraud vs Normal Transactions", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_class_distribution.png"), dpi=150)
    plt.close()

    # --- 2. Transaction Amount Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, label, color in zip(axes, [0, 1], ["steelblue", "crimson"]):
        subset = df[df[target_col] == label]["amount"]
        ax.hist(subset.clip(upper=subset.quantile(0.99)), bins=60, color=color,
                edgecolor="black", alpha=0.85)
        ax.set_title(f"Amount Distribution — {'Fraud' if label else 'Normal'}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Transaction Amount")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_amount_distribution.png"), dpi=150)
    plt.close()

    # --- 3. Correlation Heatmap (numeric columns only) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix  = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_correlation_heatmap.png"), dpi=150)
    plt.close()

    log.info("EDA plots saved to %s/", output_dir)


# =============================================================================
# STEP 3 — DATA PREPROCESSING
# =============================================================================

def preprocess(df: pd.DataFrame, target_col: str):
    """
    Clean and transform the dataset:
      - Drop identifier columns (no predictive value)
      - Encode categorical 'type' column
      - Fill missing values
      - Separate features / target
      - Stratified train-test split
      - Scale numerical features
    """
    log.info("Preprocessing data ...")

    # Drop high-cardinality identifier columns — they are unique per transaction
    # and would cause massive overfitting if kept
    drop_cols = ["nameOrig", "nameDest"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode the transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.)
    if "type" in df.columns:
        le = LabelEncoder()
        df["type"] = le.fit_transform(df["type"].astype(str))
        log.info("Encoded 'type' categories: %s", list(le.classes_))

    # Handle missing values — fill numeric with median (robust to outliers)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    log.info("Features shape: %s  |  Target distribution:\n%s",
             X.shape, y.value_counts().to_string())

    # Stratified train-test split preserves fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y     # CRITICAL for imbalanced datasets
    )

    # Scale numerical columns — tree models don't strictly need scaling,
    # but it improves convergence when combined with linear models or NN.
    # We scale here for pipeline completeness and future model swapping.
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler   = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    log.info(
        "Train size: %d  |  Test size: %d  |  Fraud in train: %d  |  Fraud in test: %d",
        len(X_train), len(X_test), y_train.sum(), y_test.sum()
    )

    return X_train, X_test, y_train, y_test, scaler, list(X.columns)


# =============================================================================
# STEP 4 — HANDLE CLASS IMBALANCE + TRAIN XGBOOST
# =============================================================================

def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Compute scale_pos_weight = count(negatives) / count(positives).
    
    This is XGBoost's built-in imbalance handler.
    It up-weights the minority (fraud) class during gradient computation,
    equivalent to SMOTE but without synthetic sample generation.
    Preferred for large datasets where SMOTE would be very slow.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    ratio = neg / pos
    log.info("scale_pos_weight = %.2f  (neg=%d, pos=%d)", ratio, neg, pos)
    return ratio


def train_model(X_train, X_test, y_train, y_test) -> XGBClassifier:
    """
    Train XGBoost with early stopping on the validation set.
    Early stopping prevents overfitting by halting when the eval metric
    stops improving for `early_stopping_rounds` consecutive rounds.
    """
    log.info("Training XGBoost model ...")

    spw    = compute_scale_pos_weight(y_train)
    params = CONFIG["xgb_params"].copy()
    params["scale_pos_weight"] = spw

    # eval_metric='aucpr' = Area Under Precision-Recall Curve
    # Better than ROC-AUC for highly imbalanced datasets because it focuses
    # on the minority class performance directly
    model = XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    if hasattr(model, "best_iteration"):
        log.info("Training complete. Best iteration: %d", model.best_iteration)
    else:
        log.info("Training complete.")
    return model


# =============================================================================
# STEP 5 — MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, feature_names, output_dir):
    """
    Full evaluation suite:
      - Classification metrics
      - Confusion matrix
      - ROC Curve
      - Precision-Recall Curve
    
    WHY ACCURACY IS MISLEADING:
    If fraud is 0.1% of data, predicting all 0s gives 99.9% accuracy.
    The model appears excellent but catches zero fraud. Always use
    precision, recall, F1, and ROC-AUC for imbalanced problems.
    """
    log.info("Evaluating model ...")

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, zero_division=0)
    rec       = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)
    pr_auc    = average_precision_score(y_test, y_prob)

    metrics = {
        "accuracy":   acc,
        "precision":  prec,
        "recall":     rec,
        "f1_score":   f1,
        "roc_auc":    roc_auc,
        "pr_auc":     pr_auc,
    }

    log.info("=" * 50)
    log.info("EVALUATION RESULTS")
    log.info("=" * 50)
    for k, v in metrics.items():
        log.info("  %-15s : %.4f", k, v)
    log.info("\nClassification Report:\n%s",
             classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"], ax=ax)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # Annotate TP/TN/FP/FN directly
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, labels[i][j],
                    ha='center', color='red', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_confusion_matrix.png"), dpi=150)
    plt.close()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_roc_curve.png"), dpi=150)
    plt.close()

    # --- Precision-Recall Curve ---
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall_vals, precision_vals, color='teal', lw=2,
            label=f"PR Curve (AP = {pr_auc:.4f})")
    baseline = y_test.sum() / len(y_test)
    ax.axhline(baseline, color='red', linestyle='--',
               label=f"Baseline (fraud rate = {baseline:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_precision_recall_curve.png"), dpi=150)
    plt.close()

    log.info("Evaluation plots saved to %s/", output_dir)
    return metrics, y_pred, y_prob


# =============================================================================
# STEP 6 — HYPERPARAMETER OPTIMIZATION
# =============================================================================

def hyperparameter_tuning(X_train, y_train) -> dict:
    """
    RandomizedSearchCV over key XGBoost hyperparameters.
    Uses StratifiedKFold to preserve fraud ratio in each fold.
    Scoring on 'roc_auc' — a threshold-free metric suitable for fraud.
    """
    log.info("Starting hyperparameter tuning (RandomizedSearchCV) ...")

    param_dist = {
        "max_depth":        [3, 4, 5, 6, 7, 8],
        "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
        "n_estimators":     [200, 300, 400, 500],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
        "gamma":            [0, 0.1, 0.5, 1, 2],
        "min_child_weight": [1, 3, 5, 7, 10],
    }

    spw = compute_scale_pos_weight(y_train)
    base_model = XGBClassifier(
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=CONFIG["random_state"],
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True,
                         random_state=CONFIG["random_state"])

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,              # Increase to 50+ for production
        scoring="roc_auc",
        cv=cv,
        verbose=2,
        random_state=CONFIG["random_state"],
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    log.info("Best Parameters : %s", search.best_params_)
    log.info("Best CV ROC-AUC : %.4f", search.best_score_)

    return search.best_params_, search.best_score_, search.best_estimator_


# =============================================================================
# STEP 7 — FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(model, feature_names, output_dir):
    """
    Plot and save XGBoost feature importance.
    'weight' = number of times a feature is used in splits (default)
    'gain'   = average gain when feature is used (more meaningful)
    """
    log.info("Plotting feature importance ...")

    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(list(importance.items()), columns=["feature", "gain"])
        .sort_values("gain", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(imp_df["feature"], imp_df["gain"],
                   color="steelblue", edgecolor="black")
    ax.invert_yaxis()
    ax.set_title("XGBoost Feature Importance (Gain)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Gain")
    for bar in bars:
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()

    log.info("Top features by gain:\n%s", imp_df.to_string(index=False))
    return imp_df


# =============================================================================
# STEP 8 — FRAUD RISK SCORING
# =============================================================================

def fraud_risk_score(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - fraud_probability : raw model probability (0.0 – 1.0)
      - risk_score        : scaled to 0–100
      - risk_category     : Low / Medium / High
    
    Thresholds (tunable based on business risk appetite):
      Low Risk    : score < 30
      Medium Risk : 30 <= score < 70
      High Risk   : score >= 70
    """
    probs      = model.predict_proba(X)[:, 1]
    scores     = np.round(probs * 100, 2)
    categories = pd.cut(
        scores,
        bins=[-np.inf, 30, 70, np.inf],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    result = pd.DataFrame({
        "fraud_probability": np.round(probs, 4),
        "risk_score":        scores,
        "risk_category":     categories,
    })
    return result


# =============================================================================
# STEP 9 — SAVE ARTIFACTS
# =============================================================================

def save_artifacts(model, scaler, metrics, y_test, y_pred, y_prob,
                   output_dir, X_test_index):
    """
    Save trained model, scaler, metrics, and prediction CSV.
    """
    log.info("Saving artifacts to %s/ ...", output_dir)

    # Model and scaler
    joblib.dump(model,  os.path.join(output_dir, "xgb_fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    # Metrics JSON
    import json
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: round(float(v), 4) for k, v in metrics.items()}, f, indent=2)

    # Predictions CSV
    preds_df = pd.DataFrame({
        "index":            X_test_index,
        "actual":           y_test.values,
        "predicted":        y_pred,
        "fraud_probability": np.round(y_prob, 4),
        "risk_score":       np.round(y_prob * 100, 2),
        "risk_category": pd.cut(
            y_prob * 100,
            bins=[-np.inf, 30, 70, np.inf],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        ),
    })
    preds_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    log.info("Saved: xgb_fraud_model.pkl, scaler.pkl, "
             "evaluation_metrics.json, predictions.csv")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("FRAUD DETECTION PIPELINE — START")
    log.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load and Explore

    # ------------------------------------------------------------------ #
    df = load_data(
    CONFIG["csv_path"],
    sample_size=CONFIG["sample_size"]
)

    # ------------------------------------------------------------------ #
    # 2. EDA Visualizations
    # ------------------------------------------------------------------ #
    generate_eda_plots(df, CONFIG["target_col"], CONFIG["output_dir"])

    # ------------------------------------------------------------------ #
    # 3. Preprocessing
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(
        df, CONFIG["target_col"]
    )

    # ------------------------------------------------------------------ #
    # 4. Train Baseline XGBoost
    # ------------------------------------------------------------------ #
    model = train_model(X_train, X_test, y_train, y_test)

    # ------------------------------------------------------------------ #
    # 5. Evaluate Baseline Model
    # ------------------------------------------------------------------ #
    metrics, y_pred, y_prob = evaluate_model(
        model, X_test, y_test, feature_names, CONFIG["output_dir"]
    )

    # ------------------------------------------------------------------ #
    # 6. Hyperparameter Tuning (on a smaller subset for speed)
    # ------------------------------------------------------------------ #
    log.info("Running hyperparameter tuning on a 20%% subsample for speed ...")
    from sklearn.utils import resample
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train, test_size=0.80,
        stratify=y_train, random_state=CONFIG["random_state"]
    )
    best_params, best_score, tuned_model = hyperparameter_tuning(X_tune, y_tune)
    log.info("Tuning done. Re-evaluating tuned model on full test set ...")
    tuned_metrics, tuned_pred, tuned_prob = evaluate_model(
        tuned_model, X_test, y_test, feature_names,
        CONFIG["output_dir"]   # Plots will overwrite baseline with tuned results
    )

    # ------------------------------------------------------------------ #
    # 7. Feature Importance
    # ------------------------------------------------------------------ #
    imp_df = plot_feature_importance(tuned_model, feature_names, CONFIG["output_dir"])

    # ------------------------------------------------------------------ #
    # 8. Fraud Risk Scoring (on first 10 test rows as demo)
    # ------------------------------------------------------------------ #
    log.info("Fraud Risk Scoring — sample of 10 test transactions:")
    sample_risk = fraud_risk_score(tuned_model, X_test.head(10))
    sample_risk["actual"] = y_test.head(10).values
    log.info("\n%s", sample_risk.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 9. Save All Artifacts
    # ------------------------------------------------------------------ #
    save_artifacts(
        tuned_model, scaler, tuned_metrics,
        y_test, tuned_pred, tuned_prob,
        CONFIG["output_dir"], X_test.index
    )

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE — All outputs saved to: %s/", CONFIG["output_dir"])
    log.info("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()