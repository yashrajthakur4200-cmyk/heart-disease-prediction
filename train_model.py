# =============================================================================
# Heart Disease Prediction - Model Training Script
# Author: ML Engineering Team
# Description: Complete EDA, preprocessing, model training, evaluation & saving
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_PATH  = "dataset/heart.csv"
MODELS_DIR    = "models"
SCREENSHOTS   = "screenshots"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2

os.makedirs(MODELS_DIR,   exist_ok=True)
os.makedirs(SCREENSHOTS,  exist_ok=True)

# Consistent color palette
PALETTE   = ["#E74C3C", "#2ECC71"]
BG_COLOR  = "#F8F9FA"
GRID_CLR  = "#DEE2E6"
ACCENT    = "#2C3E50"

def set_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor":   BG_COLOR,
        "axes.edgecolor":   GRID_CLR,
        "axes.grid":        True,
        "grid.color":       GRID_CLR,
        "grid.linestyle":   "--",
        "grid.alpha":       0.7,
        "font.family":      "DejaVu Sans",
        "axes.titlesize":   14,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
    })

set_style()

# =============================================================================
# STEP 1 — LOAD DATASET
# =============================================================================
print("\n" + "="*60)
print("  HEART DISEASE PREDICTION — MODEL TRAINING")
print("="*60)

def load_data(path):
    """Load the heart disease CSV and return a DataFrame."""
    if not os.path.exists(path):
        print(f"\n[INFO] '{path}' not found — generating synthetic dataset …")
        np.random.seed(RANDOM_STATE)
        n = 1025
        df = pd.DataFrame({
            "age":      np.random.randint(29, 77, n),
            "sex":      np.random.randint(0, 2, n),
            "cp":       np.random.randint(0, 4, n),
            "trestbps": np.clip(np.random.normal(131, 17, n), 94, 200).astype(int),
            "chol":     np.clip(np.random.normal(246, 52, n), 126, 564).astype(int),
            "fbs":      np.random.randint(0, 2, n),
            "restecg":  np.random.randint(0, 3, n),
            "thalach":  np.clip(np.random.normal(149, 23, n), 71, 202).astype(int),
            "exang":    np.random.randint(0, 2, n),
            "oldpeak":  np.round(np.clip(np.random.exponential(1.0, n), 0, 6.2), 1),
            "slope":    np.random.randint(0, 3, n),
            "ca":       np.random.randint(0, 4, n),
            "thal":     np.random.randint(0, 4, n),
            "target":   np.random.randint(0, 2, n),
        })
        os.makedirs("dataset", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[INFO] Synthetic dataset saved to '{path}'")
    df = pd.read_csv(path)
    print(f"\n[✓] Dataset loaded — {df.shape[0]} rows × {df.shape[1]} columns")
    return df

df_raw = load_data(DATASET_PATH)

# =============================================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n[STEP 2] Exploratory Data Analysis …")

print("\n── Column Descriptions ──────────────────────────────────────")
col_info = {
    "age":      "Age of patient (years)",
    "sex":      "Sex (1 = Male, 0 = Female)",
    "cp":       "Chest pain type (0-3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol":     "Serum cholesterol (mg/dl)",
    "fbs":      "Fasting blood sugar > 120 mg/dl (1=True, 0=False)",
    "restecg":  "Resting ECG results (0,1,2)",
    "thalach":  "Maximum heart rate achieved",
    "exang":    "Exercise induced angina (1=Yes, 0=No)",
    "oldpeak":  "ST depression induced by exercise",
    "slope":    "Slope of peak exercise ST segment",
    "ca":       "Number of major vessels (0-3) colored by fluoroscopy",
    "thal":     "Thalassemia (0=Normal, 1=Fixed defect, 2=Reversible defect, 3=Unknown)",
    "target":   "Heart disease (1=Disease, 0=No disease) — TARGET",
}
for col, desc in col_info.items():
    print(f"  {col:10s} → {desc}")

print("\n── Dataset Info ─────────────────────────────────────────────")
print(df_raw.dtypes)
print("\n── Missing Values ───────────────────────────────────────────")
print(df_raw.isnull().sum())
print("\n── Descriptive Statistics ───────────────────────────────────")
print(df_raw.describe().round(2))

# =============================================================================
# STEP 3 — DATA PREPROCESSING
# =============================================================================
print("\n[STEP 3] Data Preprocessing …")

df = df_raw.copy()

# 3a. Handle missing values
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  [FIX] '{col}' — filled missing with median")

# 3b. Remove duplicate rows
n_dup = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"  [FIX] Removed {n_dup} duplicate row(s)")

# 3c. Outlier capping via IQR (numerical columns only, excluding target/binary)
continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
for col in continuous:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR     = Q3 - Q1
    lo, hi  = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    before  = df.shape[0]
    df[col] = df[col].clip(lower=lo, upper=hi)
    print(f"  [FIX] '{col}' — capped outliers to [{lo:.1f}, {hi:.1f}]")

# 3d. Ensure valid target
df = df[df["target"].isin([0, 1])]
print(f"  [✓] After cleaning: {df.shape[0]} rows remain")

# =============================================================================
# STEP 4 — VISUALIZATIONS
# =============================================================================
print("\n[STEP 4] Generating visualizations …")

# ── 4.1 Disease Distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Heart Disease Distribution", fontsize=16, fontweight="bold", color=ACCENT)

counts  = df["target"].value_counts()
labels  = ["No Disease", "Heart Disease"]
colors  = PALETTE

axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
axes[0].set_title("Count")
axes[0].set_ylabel("Number of Patients")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Proportion")

plt.tight_layout()
plt.savefig(f"{SCREENSHOTS}/01_disease_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] 01_disease_distribution.png")

# ── 4.2 Age Distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Age Distribution by Heart Disease Status", fontsize=16, fontweight="bold", color=ACCENT)

for target_val, label, color in zip([0, 1], ["No Disease", "Heart Disease"], PALETTE):
    subset = df[df["target"] == target_val]["age"]
    ax.hist(subset, bins=20, alpha=0.7, label=label, color=color, edgecolor="white")

ax.set_xlabel("Age (years)")
ax.set_ylabel("Count")
ax.legend()
ax.axvline(df["age"].mean(), color=ACCENT, linestyle="--", linewidth=1.5,
           label=f"Mean Age: {df['age'].mean():.1f}")

plt.tight_layout()
plt.savefig(f"{SCREENSHOTS}/02_age_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] 02_age_distribution.png")

# ── 4.3 Cholesterol Distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Cholesterol Analysis", fontsize=16, fontweight="bold", color=ACCENT)

axes[0].hist(df["chol"], bins=30, color="#3498DB", edgecolor="white", alpha=0.85)
axes[0].set_title("Overall Cholesterol Distribution")
axes[0].set_xlabel("Cholesterol (mg/dl)")
axes[0].set_ylabel("Count")
axes[0].axvline(df["chol"].mean(), color="#E74C3C", linestyle="--",
                label=f"Mean: {df['chol'].mean():.1f}")
axes[0].legend()

df.boxplot(column="chol", by="target", ax=axes[1],
           boxprops=dict(color=ACCENT), medianprops=dict(color="#E74C3C", linewidth=2))
axes[1].set_title("Cholesterol by Disease Status")
axes[1].set_xlabel("Target (0=No Disease, 1=Disease)")
axes[1].set_ylabel("Cholesterol (mg/dl)")
plt.suptitle("")

plt.tight_layout()
plt.savefig(f"{SCREENSHOTS}/03_cholesterol_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] 03_cholesterol_distribution.png")

# ── 4.4 Correlation Heatmap ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.5, linecolor="white",
            ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", color=ACCENT, pad=15)
plt.tight_layout()
plt.savefig(f"{SCREENSHOTS}/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] 04_correlation_heatmap.png")

# ── 4.5 (placeholder) Feature Importance — computed after training ─────────────
# (saved later in Step 6)

# =============================================================================
# STEP 5 — FEATURE ENGINEERING & SPLITTING
# =============================================================================
print("\n[STEP 5] Feature Engineering & Train/Test Split …")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# =============================================================================
# STEP 6 — MODEL TRAINING & EVALUATION
# =============================================================================
print("\n[STEP 6] Training & Evaluating Models …\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "SVM":                 SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE),
}

results   = {}
roc_data  = {}
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred   = model.predict(X_test_s)
    y_prob   = model.predict_proba(X_test_s)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cv_s = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy").mean()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)

    results[name]  = {"Accuracy": acc, "Precision": prec, "Recall": rec,
                      "F1 Score": f1, "CV Score": cv_s, "AUC": roc_auc,
                      "y_pred": y_pred, "model": model}
    roc_data[name] = (fpr, tpr, roc_auc)

    print(f"  {name:<22} Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  AUC={roc_auc:.3f}")

# ── 4.5 Feature Importance ────────────────────────────────────────────────────
rf_model   = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(importances.index, importances.values,
               color=plt.cm.RdYlGn(importances.values / importances.values.max()),
               edgecolor="white")
ax.set_title("Feature Importance (Random Forest)", fontsize=15, fontweight="bold", color=ACCENT)
ax.set_xlabel("Importance Score")
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{SCREENSHOTS}/05_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] 05_feature_importance.png")

# ── Confusion Matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold", color=ACCENT)
axes_flat  = axes.flatten()

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                ax=axes_flat[idx], linewidths=0.5, linecolor="white",
                cbar=False)
    axes_flat[idx].set_title(name, fontweight="bold")
    axes_flat[idx].set_xlabel("Predicted")
    axes_flat[idx].set_ylabel("Actual")

axes_flat[-1].set_visible(False)
plt.tight_layout()
plt.savefig(f"{SCREENSHOTS}/06_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] 06_confusion_matrices.png")

# ── ROC Curves ────────────────────────────────────────────────────────────────
roc_colors = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]
fig, ax    = plt.subplots(figsize=(9, 7))

for (name, (fpr, tpr, roc_auc)), color in zip(roc_data.items(), roc_colors):
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")

ax.plot([0,1],[0,1], "k--", lw=1.2, alpha=0.6, label="Random Classifier")
ax.fill_between([0,1],[0,1], alpha=0.04, color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models", fontsize=15, fontweight="bold", color=ACCENT)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{SCREENSHOTS}/07_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] 07_roc_curves.png")

# ── Model Comparison Bar Chart ────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k not in ("y_pred","model")}
    for name, res in results.items()
}).T

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", color=ACCENT)
axes_flat  = axes.flatten()
bar_colors = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]

for idx, metric in enumerate(["Accuracy","Precision","Recall","F1 Score","AUC","CV Score"]):
    vals = metrics_df[metric]
    bars = axes_flat[idx].bar(vals.index, vals.values,
                              color=bar_colors, edgecolor="white", alpha=0.9)
    axes_flat[idx].set_title(metric, fontweight="bold")
    axes_flat[idx].set_ylim(0, 1.05)
    axes_flat[idx].set_xticklabels(vals.index, rotation=20, ha="right", fontsize=8)
    for b, v in zip(bars, vals.values):
        axes_flat[idx].text(b.get_x()+b.get_width()/2, v+0.01,
                            f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{SCREENSHOTS}/08_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [✓] 08_model_comparison.png")

# =============================================================================
# STEP 7 — BEST MODEL SELECTION
# =============================================================================
print("\n[STEP 7] Selecting Best Model …")

metrics_df_clean = metrics_df.drop(columns=["y_pred","model"], errors="ignore")
# Composite score: weighted average of key metrics
metrics_df["composite"] = (
    0.25 * metrics_df["Accuracy"] +
    0.25 * metrics_df["F1 Score"] +
    0.25 * metrics_df["AUC"]      +
    0.25 * metrics_df["CV Score"]
)
best_name  = metrics_df["composite"].idxmax()
best_model = results[best_name]["model"]

print(f"\n  ┌─ Best Model: {best_name}")
print(f"  │  Accuracy  : {results[best_name]['Accuracy']:.4f}")
print(f"  │  F1 Score  : {results[best_name]['F1 Score']:.4f}")
print(f"  │  AUC       : {results[best_name]['AUC']:.4f}")
print(f"  └─ CV Score  : {results[best_name]['CV Score']:.4f}")

print(f"\n  Classification Report — {best_name}:\n")
print(classification_report(y_test, results[best_name]["y_pred"],
                             target_names=["No Disease","Heart Disease"]))

# =============================================================================
# STEP 8 — SAVE MODEL & SCALER
# =============================================================================
print("\n[STEP 8] Saving model and scaler …")

joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")
joblib.dump(scaler,     f"{MODELS_DIR}/scaler.pkl")
joblib.dump(list(X.columns), f"{MODELS_DIR}/feature_names.pkl")

# Save model metadata
import json
meta = {
    "best_model_name":  best_name,
    "feature_columns":  list(X.columns),
    "accuracy":  round(results[best_name]["Accuracy"], 4),
    "f1_score":  round(results[best_name]["F1 Score"], 4),
    "auc":       round(results[best_name]["AUC"],      4),
    "cv_score":  round(results[best_name]["CV Score"], 4),
}
with open(f"{MODELS_DIR}/model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"  [✓] Saved: models/best_model.pkl")
print(f"  [✓] Saved: models/scaler.pkl")
print(f"  [✓] Saved: models/feature_names.pkl")
print(f"  [✓] Saved: models/model_metadata.json")

print("\n" + "="*60)
print("  TRAINING COMPLETE!")
print(f"  Best Model : {best_name}")
print(f"  Accuracy   : {results[best_name]['Accuracy']:.4f}")
print(f"  Plots saved to: {SCREENSHOTS}/")
print("="*60 + "\n")
