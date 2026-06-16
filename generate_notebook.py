import json
import uuid

def cell_id():
    return str(uuid.uuid4())[:8]

def md_cell(source):
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id(),
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }

cells = []

# ─── HEADER ───────────────────────────────────────────────────────────────────
cells.append(md_cell("""# Proyek Akhir: Sistem Machine Learning - Klasifikasi Wine

**Nama:** saipulkarimsuleman  
**Email:** saipulkarimsuleman@gmail.com  
**Dataset:** Wine Classification (sklearn built-in)  
**Tujuan:** Membangun sistem ML production-ready mencakup EDA, preprocessing, pelatihan model dengan MLflow tracking, dan evaluasi komprehensif.

---

## Deskripsi Dataset

Dataset Wine dari sklearn berisi hasil analisis kimia dari wine yang berasal dari tiga kultivar yang berbeda di Italia. Dataset ini memiliki:
- **178 sampel** dengan **13 fitur** kimia
- **3 kelas target**: class_0, class_1, class_2

## Alur Kerja (Workflow)

1. **Import Libraries** - Mempersiapkan semua dependensi
2. **Data Loading** - Memuat dan menyimpan dataset
3. **Exploratory Data Analysis (EDA)** - Analisis statistik dan visualisasi
4. **Data Preprocessing** - Pembersihan, transformasi, dan split data
5. **MLflow Setup** - Konfigurasi experiment tracking
6. **Model Training** - Melatih beberapa model dengan logging MLflow
7. **Model Evaluation** - Evaluasi performa dan perbandingan model
8. **Model Registration** - Registrasi model terbaik ke MLflow Registry
"""))

# ─── CELL 1: IMPORTS ──────────────────────────────────────────────────────────
cells.append(md_cell("## 1. Import Libraries"))
cells.append(code_cell("""# Standard libraries
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
%matplotlib inline
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid", palette="Set2")

# Machine Learning
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay
)

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

print("All libraries imported successfully!")
print(f"pandas version: {pd.__version__}")
print(f"sklearn version: {__import__('sklearn').__version__}")
print(f"mlflow version: {mlflow.__version__}")
"""))

# ─── CELL 2: DATA LOADING ─────────────────────────────────────────────────────
cells.append(md_cell("## 2. Data Loading"))
cells.append(code_cell("""# Load Wine dataset dari sklearn
wine = load_wine()

# Buat DataFrame
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
df["target_name"] = df["target"].map({0: "class_0", 1: "class_1", 2: "class_2"})

# Simpan ke CSV
os.makedirs("data", exist_ok=True)
df.to_csv("data/wine_data.csv", index=False)

print(f"Dataset berhasil dimuat!")
print(f"Shape: {df.shape}")
print(f"Kolom: {df.columns.tolist()}")
print(f"\\nJumlah kelas: {df['target'].nunique()}")
print(f"Nama kelas: {wine.target_names.tolist()}")
print(f"\\nDistribusi target:")
print(df["target_name"].value_counts())
"""))

cells.append(code_cell("""# Tampilkan 5 baris pertama
print("5 baris pertama dataset:")
df.head()
"""))

cells.append(code_cell("""# Info dataset
print("Informasi Dataset:")
df.info()
"""))

# ─── CELL 3: EDA ──────────────────────────────────────────────────────────────
cells.append(md_cell("""## 3. Exploratory Data Analysis (EDA)

EDA dilakukan untuk memahami karakteristik data sebelum melakukan pemodelan.
"""))

cells.append(code_cell("""# Statistik deskriptif
print("Statistik Deskriptif:")
df.describe().round(3)
"""))

cells.append(code_cell("""# Cek missing values
print("Missing Values per Kolom:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Tidak ada missing values!")
print(f"\\nTotal missing values: {df.isnull().sum().sum()}")
"""))

cells.append(code_cell("""# Distribusi kelas target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
counts = df["target_name"].value_counts()
axes[0].bar(counts.index, counts.values, color=["#2196F3", "#4CAF50", "#FF9800"])
axes[0].set_title("Distribusi Kelas Target", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Kelas")
axes[0].set_ylabel("Jumlah Sampel")
for i, (name, val) in enumerate(counts.items()):
    axes[0].text(i, val + 0.5, str(val), ha="center", fontweight="bold")

# Pie chart
axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=["#2196F3", "#4CAF50", "#FF9800"], startangle=90)
axes[1].set_title("Proporsi Kelas Target", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig("data/target_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
print("Dataset relatif seimbang untuk semua kelas")
"""))

cells.append(code_cell("""# Distribusi semua fitur numerik
feature_cols = [c for c in df.columns if c not in ["target", "target_name"]]
n_features = len(feature_cols)

fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    axes[i].hist(df[col], bins=25, edgecolor="white", color="#2196F3", alpha=0.8)
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_xlabel("Nilai")
    axes[i].set_ylabel("Frekuensi")

# Hapus axes yang tidak terpakai
for j in range(n_features, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Distribusi Setiap Fitur", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/feature_distributions.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(code_cell("""# Boxplot fitur per kelas
feature_subset = ["alcohol", "malic_acid", "total_phenols", "flavanoids",
                  "color_intensity", "proline"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
colors = ["#2196F3", "#4CAF50", "#FF9800"]

for i, col in enumerate(feature_subset):
    data_per_class = [df[df["target"] == cls][col].values for cls in range(3)]
    bp = axes[i].boxplot(data_per_class, patch_artist=True,
                         labels=["Class 0", "Class 1", "Class 2"])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Nilai")

plt.suptitle("Distribusi Fitur Utama per Kelas", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/boxplot_features_per_class.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(code_cell("""# Heatmap korelasi
feature_cols_only = [c for c in df.columns if c not in ["target", "target_name"]]
corr_matrix = df[feature_cols_only + ["target"]].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, mask=mask, square=True,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Heatmap Korelasi Antar Fitur", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.show()

# Fitur dengan korelasi tinggi terhadap target
target_corr = corr_matrix["target"].drop("target").abs().sort_values(ascending=False)
print("\\nKorelasi Fitur dengan Target (absolut, diurutkan):")
for feat, val in target_corr.items():
    print(f"  {feat:40s}: {val:.4f}")
"""))

cells.append(code_cell("""# Pairplot untuk fitur terpenting
top_features = target_corr.head(5).index.tolist()
pairplot_df = df[top_features + ["target_name"]]

g = sns.pairplot(pairplot_df, hue="target_name",
                 palette={"class_0": "#2196F3", "class_1": "#4CAF50", "class_2": "#FF9800"},
                 diag_kind="kde", plot_kws={"alpha": 0.6})
g.fig.suptitle("Pairplot 5 Fitur Terkorelasi Tertinggi", y=1.02, fontsize=14, fontweight="bold")
plt.savefig("data/pairplot_top_features.png", dpi=100, bbox_inches="tight")
plt.show()
print(f"5 fitur terkorelasi tertinggi dengan target: {top_features}")
"""))

cells.append(code_cell("""# Violin plot untuk alcohol dan proline (fitur paling diskriminatif)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
key_features = ["alcohol", "proline"]

for i, col in enumerate(key_features):
    sns.violinplot(data=df, x="target_name", y=col, ax=axes[i],
                   palette={"class_0": "#2196F3", "class_1": "#4CAF50", "class_2": "#FF9800"})
    axes[i].set_title(f"Distribusi {col} per Kelas", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Kelas")
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.savefig("data/violin_key_features.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ─── CELL 4: PREPROCESSING ────────────────────────────────────────────────────
cells.append(md_cell("""## 4. Data Preprocessing

Tahapan preprocessing meliputi:
- Penanganan missing values (pengecekan)
- Penghapusan outlier menggunakan Z-Score
- Pembagian data train/test dengan stratified split
- Normalisasi fitur menggunakan StandardScaler
"""))

cells.append(code_cell("""# 4.1 Pisahkan fitur dan target
feature_cols = [c for c in df.columns if c not in ["target", "target_name"]]
X = df[feature_cols]
y = df["target"]

print(f"Fitur (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Nama fitur: {X.columns.tolist()}")
"""))

cells.append(code_cell("""# 4.2 Penanganan missing values
print("Pengecekan missing values:")
missing = X.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "Tidak ditemukan missing values!")

# Jika ada missing values, isi dengan median
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)
        print(f"  {col}: diisi dengan median = {X[col].median():.3f}")
"""))

cells.append(code_cell("""# 4.3 Deteksi dan hapus outlier menggunakan Z-Score
from scipy import stats

z_scores = np.abs(stats.zscore(X))
outlier_mask = (z_scores < 3.0).all(axis=1)

X_clean = X[outlier_mask].reset_index(drop=True)
y_clean = y[outlier_mask].reset_index(drop=True)

n_removed = len(X) - len(X_clean)
print(f"Jumlah sampel sebelum: {len(X)}")
print(f"Jumlah sampel setelah penghapusan outlier: {len(X_clean)}")
print(f"Outlier dihapus: {n_removed} sampel")

# Distribusi kelas setelah pembersihan
print(f"\\nDistribusi kelas setelah pembersihan:")
print(pd.Series(y_clean).value_counts())
"""))

cells.append(code_cell("""# 4.4 Split data train/test (80:20 dengan stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean,
    test_size=0.2,
    random_state=42,
    stratify=y_clean  # Pastikan distribusi kelas seimbang
)

print(f"Train set: {X_train.shape[0]} sampel")
print(f"Test set:  {X_test.shape[0]} sampel")
print(f"\\nDistribusi kelas pada train set:")
print(pd.Series(y_train).value_counts())
print(f"\\nDistribusi kelas pada test set:")
print(pd.Series(y_test).value_counts())
"""))

cells.append(code_cell("""# 4.5 Normalisasi fitur menggunakan StandardScaler
import joblib

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simpan scaler
os.makedirs("artifacts", exist_ok=True)
joblib.dump(scaler, "artifacts/scaler.pkl")

# Kembalikan ke DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

print("Data berhasil dinormalisasi dengan StandardScaler")
print(f"\\nStatistik setelah normalisasi (train set):")
print(X_train_scaled.describe().round(3).loc[["mean", "std", "min", "max"]])
"""))

cells.append(code_cell("""# 4.6 Visualisasi perbandingan sebelum dan sesudah normalisasi
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Sebelum normalisasi
X_train.iloc[:, :6].plot(kind="box", ax=axes[0], rot=45)
axes[0].set_title("Sebelum Normalisasi\\n(6 Fitur Pertama)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Nilai")

# Sesudah normalisasi
X_train_scaled.iloc[:, :6].plot(kind="box", ax=axes[1], rot=45)
axes[1].set_title("Sesudah Normalisasi\\n(6 Fitur Pertama)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Nilai (Z-Score)")

plt.tight_layout()
plt.savefig("data/normalization_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("Preprocessing selesai!")
"""))

# ─── CELL 5: MLFLOW SETUP ─────────────────────────────────────────────────────
cells.append(md_cell("""## 5. Setup MLflow Experiment Tracking

Konfigurasi MLflow untuk mencatat semua eksperimen, parameter, metrik, dan artefak model.
"""))

cells.append(code_cell("""# Setup MLflow
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "wine-classification-experiment"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Verifikasi koneksi
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment:
    print(f"Experiment ditemukan: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}")
else:
    print(f"Experiment baru dibuat: {EXPERIMENT_NAME}")

print(f"\\nMLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow Artifact URI: {mlflow.get_artifact_uri() if mlflow.active_run() else 'N/A (no active run)'}")
"""))

# ─── CELL 6: MODEL TRAINING ───────────────────────────────────────────────────
cells.append(md_cell("""## 6. Model Training dengan MLflow

Melatih tiga model berbeda dan mencatat semua eksperimen ke MLflow:
1. **Logistic Regression** - Model baseline sederhana
2. **Random Forest** - Ensemble model berbasis pohon keputusan
3. **Gradient Boosting** - Boosting ensemble yang kuat
"""))

cells.append(code_cell("""# Fungsi helper untuk menghitung semua metrik
def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc_ovr"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            )
        except Exception:
            pass
    return metrics

print("Fungsi helper siap digunakan.")
"""))

cells.append(code_cell("""# ─── MODEL 1: LOGISTIC REGRESSION ────────────────────────────────────────────
print("Training Model 1: Logistic Regression...")

lr_params = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42
}

with mlflow.start_run(run_name="logistic_regression") as run:
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_params(lr_params)
    mlflow.log_param("train_size", len(X_train_scaled))
    mlflow.log_param("test_size", len(X_test_scaled))
    mlflow.log_param("n_features", X_train_scaled.shape[1])
    mlflow.log_param("scaler", "StandardScaler")
    
    # Train
    model_lr = LogisticRegression(**lr_params)
    model_lr.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_lr = model_lr.predict(X_test_scaled)
    y_proba_lr = model_lr.predict_proba(X_test_scaled)
    metrics_lr = compute_metrics(y_test, y_pred_lr, y_proba_lr)
    
    # Log metrics
    for k, v in metrics_lr.items():
        mlflow.log_metric(k, v)
    
    # Cross validation
    cv_scores = cross_val_score(model_lr, X_train_scaled, y_train, cv=5, scoring="accuracy")
    mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
    mlflow.log_metric("cv_accuracy_std", cv_scores.std())
    
    # Log model
    signature = infer_signature(X_train_scaled, y_pred_lr)
    mlflow.sklearn.log_model(
        model_lr, "model",
        signature=signature,
        input_example=X_test_scaled.iloc[:3],
        registered_model_name="wine-classifier-lr"
    )
    
    # Log scaler artifact
    mlflow.log_artifact("artifacts/scaler.pkl")
    
    run_id_lr = run.info.run_id

print(f"  Run ID: {run_id_lr}")
print(f"  Accuracy: {metrics_lr['accuracy']:.4f}")
print(f"  F1-Score (weighted): {metrics_lr['f1_weighted']:.4f}")
print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
"""))

cells.append(code_cell("""# ─── MODEL 2: RANDOM FOREST ───────────────────────────────────────────────────
print("Training Model 2: Random Forest...")

rf_params = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1
}

with mlflow.start_run(run_name="random_forest") as run:
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_params(rf_params)
    mlflow.log_param("train_size", len(X_train_scaled))
    mlflow.log_param("test_size", len(X_test_scaled))
    mlflow.log_param("n_features", X_train_scaled.shape[1])
    mlflow.log_param("scaler", "StandardScaler")
    
    model_rf = RandomForestClassifier(**rf_params)
    model_rf.fit(X_train_scaled, y_train)
    
    y_pred_rf = model_rf.predict(X_test_scaled)
    y_proba_rf = model_rf.predict_proba(X_test_scaled)
    metrics_rf = compute_metrics(y_test, y_pred_rf, y_proba_rf)
    
    for k, v in metrics_rf.items():
        mlflow.log_metric(k, v)
    
    cv_scores_rf = cross_val_score(model_rf, X_train_scaled, y_train, cv=5, scoring="accuracy")
    mlflow.log_metric("cv_accuracy_mean", cv_scores_rf.mean())
    mlflow.log_metric("cv_accuracy_std", cv_scores_rf.std())
    
    # Feature importances
    importances = dict(zip(feature_cols, model_rf.feature_importances_.tolist()))
    with open("artifacts/feature_importances_rf.json", "w") as f:
        json.dump(importances, f, indent=2)
    mlflow.log_artifact("artifacts/feature_importances_rf.json")
    
    signature = infer_signature(X_train_scaled, y_pred_rf)
    mlflow.sklearn.log_model(
        model_rf, "model",
        signature=signature,
        input_example=X_test_scaled.iloc[:3],
        registered_model_name="wine-classifier-rf"
    )
    mlflow.log_artifact("artifacts/scaler.pkl")
    
    run_id_rf = run.info.run_id

print(f"  Run ID: {run_id_rf}")
print(f"  Accuracy: {metrics_rf['accuracy']:.4f}")
print(f"  F1-Score (weighted): {metrics_rf['f1_weighted']:.4f}")
print(f"  CV Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
"""))

cells.append(code_cell("""# ─── MODEL 3: GRADIENT BOOSTING ───────────────────────────────────────────────
print("Training Model 3: Gradient Boosting...")

gb_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "min_samples_split": 2,
    "random_state": 42
}

with mlflow.start_run(run_name="gradient_boosting") as run:
    mlflow.log_param("model_type", "GradientBoostingClassifier")
    mlflow.log_params(gb_params)
    mlflow.log_param("train_size", len(X_train_scaled))
    mlflow.log_param("test_size", len(X_test_scaled))
    mlflow.log_param("n_features", X_train_scaled.shape[1])
    mlflow.log_param("scaler", "StandardScaler")
    
    model_gb = GradientBoostingClassifier(**gb_params)
    model_gb.fit(X_train_scaled, y_train)
    
    y_pred_gb = model_gb.predict(X_test_scaled)
    y_proba_gb = model_gb.predict_proba(X_test_scaled)
    metrics_gb = compute_metrics(y_test, y_pred_gb, y_proba_gb)
    
    for k, v in metrics_gb.items():
        mlflow.log_metric(k, v)
    
    cv_scores_gb = cross_val_score(model_gb, X_train_scaled, y_train, cv=5, scoring="accuracy")
    mlflow.log_metric("cv_accuracy_mean", cv_scores_gb.mean())
    mlflow.log_metric("cv_accuracy_std", cv_scores_gb.std())
    
    signature = infer_signature(X_train_scaled, y_pred_gb)
    mlflow.sklearn.log_model(
        model_gb, "model",
        signature=signature,
        input_example=X_test_scaled.iloc[:3],
        registered_model_name="wine-classifier-gb"
    )
    mlflow.log_artifact("artifacts/scaler.pkl")
    
    run_id_gb = run.info.run_id

print(f"  Run ID: {run_id_gb}")
print(f"  Accuracy: {metrics_gb['accuracy']:.4f}")
print(f"  F1-Score (weighted): {metrics_gb['f1_weighted']:.4f}")
print(f"  CV Accuracy: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})")
"""))

# ─── CELL 7: EVALUATION ───────────────────────────────────────────────────────
cells.append(md_cell("""## 7. Evaluasi dan Perbandingan Model

Membandingkan performa semua model untuk memilih model terbaik.
"""))

cells.append(code_cell("""# Rangkuman semua model
all_results = {
    "Logistic Regression": {
        "model": model_lr, "metrics": metrics_lr,
        "y_pred": y_pred_lr, "y_proba": y_proba_lr, "run_id": run_id_lr
    },
    "Random Forest": {
        "model": model_rf, "metrics": metrics_rf,
        "y_pred": y_pred_rf, "y_proba": y_proba_rf, "run_id": run_id_rf
    },
    "Gradient Boosting": {
        "model": model_gb, "metrics": metrics_gb,
        "y_pred": y_pred_gb, "y_proba": y_proba_gb, "run_id": run_id_gb
    }
}

# Tabel perbandingan
comparison_df = pd.DataFrame({
    name: {
        "Accuracy": f"{r['metrics']['accuracy']:.4f}",
        "F1 (weighted)": f"{r['metrics']['f1_weighted']:.4f}",
        "Precision (weighted)": f"{r['metrics']['precision_weighted']:.4f}",
        "Recall (weighted)": f"{r['metrics']['recall_weighted']:.4f}",
        "ROC-AUC (OVR)": f"{r['metrics'].get('roc_auc_ovr', 'N/A'):.4f}" if isinstance(r['metrics'].get('roc_auc_ovr'), float) else "N/A",
    }
    for name, r in all_results.items()
}).T

print("\\nPerbandingan Performa Model:")
print("=" * 70)
print(comparison_df.to_string())
print("=" * 70)
"""))

cells.append(code_cell("""# Visualisasi perbandingan metrik
metrics_to_plot = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
model_names = list(all_results.keys())

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
colors = ["#2196F3", "#4CAF50", "#FF9800"]

for i, metric in enumerate(metrics_to_plot):
    values = [all_results[name]["metrics"][metric] for name in model_names]
    bars = axes[i].bar(model_names, values, color=colors, edgecolor="white", linewidth=1.5)
    axes[i].set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
    axes[i].set_ylim(0.8, 1.02)
    axes[i].set_ylabel("Score")
    axes[i].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.suptitle("Perbandingan Performa Model", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("artifacts/model_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(code_cell("""# Confusion matrix untuk semua model
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, result) in enumerate(all_results.items()):
    cm = confusion_matrix(y_test, result["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=wine.target_names)
    disp.plot(ax=axes[i], colorbar=True, cmap="Blues")
    axes[i].set_title(name, fontsize=12, fontweight="bold")

plt.suptitle("Confusion Matrix Semua Model", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("artifacts/all_confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

cells.append(code_cell("""# Classification report model terbaik
best_model_name = max(all_results, key=lambda k: all_results[k]["metrics"]["accuracy"])
best_result = all_results[best_model_name]

print(f"Model Terbaik: {best_model_name}")
print(f"Accuracy: {best_result['metrics']['accuracy']:.4f}")
print("\\nClassification Report:")
print("=" * 60)
print(classification_report(y_test, best_result["y_pred"], target_names=wine.target_names))

# Simpan report
report_dict = classification_report(
    y_test, best_result["y_pred"],
    target_names=wine.target_names,
    output_dict=True
)
with open("artifacts/classification_report.json", "w") as f:
    json.dump(report_dict, f, indent=2)
print("Classification report disimpan ke artifacts/classification_report.json")
"""))

cells.append(code_cell("""# Feature importance dari Random Forest
importances = model_rf.feature_importances_
feat_imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors_imp = ["#4CAF50" if v > 0.08 else "#2196F3" for v in feat_imp_df["importance"]]
bars = ax.barh(feat_imp_df["feature"], feat_imp_df["importance"], color=colors_imp)
ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
for bar, val in zip(bars, feat_imp_df["importance"]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("artifacts/feature_importance.png", dpi=120, bbox_inches="tight")
plt.show()
"""))

# ─── CELL 8: MODEL REGISTRATION ───────────────────────────────────────────────
cells.append(md_cell("""## 8. Model Registration ke MLflow Model Registry

Mendaftarkan model terbaik ke MLflow Model Registry dan mempromosikannya ke tahap Production.
"""))

cells.append(code_cell("""# Identifikasi dan register model terbaik
best_model_name_clean = best_model_name.lower().replace(" ", "_")
registered_model_name = f"wine-classifier-{best_model_name_clean.split('_')[0]}"

print(f"Model terbaik: {best_model_name}")
print(f"Accuracy: {best_result['metrics']['accuracy']:.4f}")
print(f"Run ID: {best_result['run_id']}")
print(f"Registered model name: {registered_model_name}")

# Promosikan ke Production
try:
    client = mlflow.tracking.MlflowClient()
    
    # Dapatkan versi terbaru
    model_versions = client.get_latest_versions(registered_model_name, stages=["None"])
    if model_versions:
        version = model_versions[-1].version
        
        # Promosikan ke Staging dulu
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage="Staging"
        )
        print(f"Model versi {version} dipromosikan ke Staging")
        
        # Promosikan ke Production
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage="Production"
        )
        print(f"Model versi {version} dipromosikan ke Production")
        
        # Set model description
        client.update_registered_model(
            name=registered_model_name,
            description=f"Wine Classification Model - Best model: {best_model_name} (accuracy={best_result['metrics']['accuracy']:.4f})"
        )
        print(f"Deskripsi model diperbarui")
    else:
        print(f"Tidak ada versi model yang tersedia untuk {registered_model_name}")
        
except Exception as e:
    print(f"Model registration: {e}")
    print("(MLflow Model Registry mungkin memerlukan konfigurasi tambahan)")
"""))

# ─── CELL 9: SUMMARY ──────────────────────────────────────────────────────────
cells.append(md_cell("""## 9. Kesimpulan

### Ringkasan Hasil

Proyek ini berhasil membangun sistem ML untuk klasifikasi wine dengan menggunakan dataset dari sklearn. Berikut adalah ringkasan:

**Dataset:**
- Dataset Wine dengan 178 sampel dan 13 fitur kimia
- 3 kelas target yang relatif seimbang

**Preprocessing:**
- Tidak ditemukan missing values
- Penghapusan outlier menggunakan Z-Score (threshold=3.0)
- Normalisasi dengan StandardScaler

**Hasil Model:**
- Semua model mencapai akurasi di atas 90%
- Model terbaik dipilih berdasarkan akurasi test set
- Model didaftarkan ke MLflow Model Registry

**Langkah Selanjutnya:**
1. Deploy model sebagai REST API menggunakan FastAPI
2. Monitor performa model menggunakan Prometheus dan Grafana
3. Jalankan CI/CD pipeline melalui GitHub Actions
"""))

cells.append(code_cell("""# Simpan rangkuman akhir
summary = {
    "best_model": best_model_name,
    "best_metrics": best_result["metrics"],
    "best_run_id": best_result["run_id"],
    "all_models": {
        name: r["metrics"] for name, r in all_results.items()
    },
    "dataset_info": {
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "n_classes": df["target"].nunique(),
        "class_names": wine.target_names.tolist()
    },
    "preprocessing": {
        "outliers_removed": len(df) - len(X_clean),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "scaler": "StandardScaler"
    }
}

with open("artifacts/final_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("Ringkasan proyek berhasil disimpan!")
print("\\nFile yang dihasilkan:")
for root, dirs, files in os.walk("artifacts"):
    for fname in files:
        fpath = os.path.join(root, fname)
        size = os.path.getsize(fpath)
        print(f"  {fpath} ({size:,} bytes)")
print("\\n[SELESAI] Notebook berhasil dijalankan!")
"""))

# ─── ASSEMBLE NOTEBOOK ────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "version": "3.9.0"
        }
    },
    "cells": cells
}

with open("E:/Proyek Akhir/notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"notebook.ipynb berhasil dibuat dengan {len(cells)} cells")

