"""Model tuning (Validation), held-out evaluation (Test), error analysis, and plotting."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestCentroid, NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from src.data_loader import CATEGORY_NAMES, load_and_cache_data
from src.preprocessing import MODEL_FEATURES, fit_transform_splits, log_transform, split_data

sns.set_theme(style="whitegrid", context="notebook", palette="deep")

PERSONA_PALETTE = {
    "Champions (VIP)": "#1b9e77",
    "Loyal Regulars": "#386cb0",
    "Recent Emerging": "#f0a202",
    "At-Risk / Hibernating": "#d95f02",
}

PERSONA_ACTIONS: Dict[str, Dict[str, str]] = {
    "Champions (VIP)": {
        "summary": "Recent, high-frequency, high-spend shoppers with broad product exploration.",
        "action": "Invite to VIP loyalty tier, offer early access to new arrivals, and provide bundle perks instead of heavy discounting.",
    },
    "Loyal Regulars": {
        "summary": "Established repeat buyers with solid basket value across multiple product categories.",
        "action": "Send cross-category recommendations (e.g., matching KitchenDining & HomeDecor items) and free-shipping basket incentives.",
    },
    "Recent Emerging": {
        "summary": "Recently active shoppers with modest purchase frequency and smaller initial baskets.",
        "action": "Run 30-day second-order follow-ups highlighting their top product category and starter bundles.",
    },
    "At-Risk / Hibernating": {
        "summary": "Lapsed customers with many days since last order, low frequency, and lower lifetime spend.",
        "action": "Trigger targeted digital win-back offers on bestsellers in their historical category preference.",
    },
}


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz, handling DBSCAN noise (-1)."""
    noise_mask = labels == -1
    noise_pct = float(noise_mask.mean() * 100.0)
    valid_mask = ~noise_mask
    valid_labels = labels[valid_mask]
    n_clusters = len(set(valid_labels.tolist()))

    if n_clusters < 2 or valid_mask.sum() < 10:
        return {
            "n_clusters": int(n_clusters),
            "noise_pct": round(noise_pct, 2),
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }

    X_valid = X[valid_mask]
    return {
        "n_clusters": int(n_clusters),
        "noise_pct": round(noise_pct, 2),
        "silhouette": round(float(silhouette_score(X_valid, valid_labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_valid, valid_labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_valid, valid_labels)), 2),
    }


def predict_dbscan_out_of_sample(
    X_train: np.ndarray,
    train_labels: np.ndarray,
    X_eval: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Assign unseen points to nearest Train core point within eps, otherwise -1 noise."""
    core_mask = train_labels != -1
    if not core_mask.any():
        return np.full(len(X_eval), -1, dtype=int)
    nn = NearestNeighbors(n_neighbors=1).fit(X_train[core_mask])
    dists, idxs = nn.kneighbors(X_eval)
    y_core = train_labels[core_mask]
    return np.where(dists[:, 0] <= eps, y_core[idxs[:, 0]], -1)


def assign_semantic_personas(df: pd.DataFrame, cluster_col: str) -> Dict[int, str]:
    """Deterministically map numeric cluster IDs to retail persona labels."""
    summary = df.groupby(cluster_col).agg(
        med_recency=("RecencyDays", "median"),
        med_freq=("Frequency", "median"),
        med_monetary=("Monetary", "median"),
    )
    ranked = summary.sort_values(
        by=["med_monetary", "med_freq", "med_recency"],
        ascending=[False, False, True],
    )
    ordered_ids = ranked.index.tolist()
    mapping: Dict[int, str] = {}
    if len(ordered_ids) == 4:
        mapping[int(ordered_ids[0])] = "Champions (VIP)"
        mapping[int(ordered_ids[1])] = "Loyal Regulars"
        c2, c3 = int(ordered_ids[2]), int(ordered_ids[3])
        if summary.loc[c2, "med_recency"] <= summary.loc[c3, "med_recency"]:
            mapping[c2] = "Recent Emerging"
            mapping[c3] = "At-Risk / Hibernating"
        else:
            mapping[c3] = "Recent Emerging"
            mapping[c2] = "At-Risk / Hibernating"
    else:
        for idx, cid in enumerate(ordered_ids):
            mapping[int(cid)] = f"Segment {idx + 1}"
    return mapping


def save_plots(
    results: Dict[str, Any],
    features_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    """Generate and save all comparison, validation, and error-analysis plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    seg_df: pd.DataFrame = results["segmented_df"]
    km_val_df: pd.DataFrame = results["kmeans_val_sweep_df"]
    hc_val_df: pd.DataFrame = results["hier_val_sweep_df"]
    comp_df: pd.DataFrame = results["comparison_df"]
    err: Dict[str, Any] = results["error_analysis"]
    pca_var = results["pca_explained_variance_ratio"]

    # 1. EDA Distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for idx, col in enumerate(["RecencyDays", "Frequency", "Monetary", "UniqueProducts"]):
        ax = axes[idx // 3, idx % 3]
        sns.histplot(np.log1p(features_df[col]), bins=35, kde=True, ax=ax, color="#386cb0")
        ax.set_title(f"log1p({col}) Distribution")
        ax.set_xlabel(f"log1p({col})")

    cat_counts = features_df["DominantCategory"].value_counts()
    sns.barplot(x=cat_counts.values, y=cat_counts.index, ax=axes[1, 1], color="#1b9e77")
    axes[1, 1].set_title("Customers by Dominant Product Category")
    axes[1, 1].set_xlabel("Customer Count")
    axes[1, 1].set_ylabel("")

    sns.histplot(features_df["CategoryEntropy"], bins=30, kde=True, ax=axes[1, 2], color="#d95f02")
    axes[1, 2].set_title("Normalized Product Category Entropy")
    axes[1, 2].set_xlabel("Entropy (0 = Single Category, 1 = Even Mix)")

    fig.suptitle("Exploratory Analysis: Customer RFM & Product Preferences (N=4,334)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "01_eda_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. Validation Hyperparameter Selection (k=2..10)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(km_val_df["k"], km_val_df["train_inertia"], marker="o", linewidth=2, color="#386cb0")
    axes[0].axvline(4, color="#d95f02", linestyle="--", label="Selected k=4")
    axes[0].set_title("Elbow Method (60% Train Inertia)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Within-Cluster SSE")
    axes[0].legend()

    axes[1].plot(km_val_df["k"], km_val_df["val_silhouette"], marker="o", linewidth=2, color="#1b9e77", label="K-Means (Val)")
    axes[1].plot(hc_val_df["k"], hc_val_df["val_silhouette"], marker="s", linewidth=2, linestyle="--", color="#7570b3", label="Hierarchical (Val)")
    axes[1].axvline(4, color="#d95f02", linestyle="--")
    axes[1].set_title("Validation Silhouette (Higher is Better)")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    axes[2].plot(km_val_df["k"], km_val_df["val_davies_bouldin"], marker="o", linewidth=2, color="#e7298a", label="K-Means (Val)")
    axes[2].plot(hc_val_df["k"], hc_val_df["val_davies_bouldin"], marker="s", linewidth=2, linestyle="--", color="#66a61e", label="Hierarchical (Val)")
    axes[2].axvline(4, color="#d95f02", linestyle="--")
    axes[2].set_title("Validation Davies-Bouldin (Lower is Better)")
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("Davies-Bouldin Index")
    axes[2].legend()

    fig.suptitle("Hyperparameter Tuning on 20% Validation Split", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "02_validation_tuning.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3. Hierarchical Ward Dendrogram
    fig, ax = plt.subplots(figsize=(12, 5.5))
    dendrogram(
        results["ward_linkage"],
        truncate_mode="lastp",
        p=30,
        leaf_rotation=45,
        leaf_font_size=9,
        show_contracted=True,
        ax=ax,
        color_threshold=34,
    )
    ax.axhline(34, color="#d95f02", linestyle="--", linewidth=1.5, label="4-Cluster Cut Threshold")
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage, 60% Training Set)")
    ax.set_xlabel("Merged Cluster Leaf (Sample Count in Parentheses)")
    ax.set_ylabel("Ward Linkage Distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "03_hierarchical_dendrogram.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4. DBSCAN k-Distance Knee & Validation Grid
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    dists = results["sorted_k_dists"]
    axes[0].plot(np.arange(len(dists)), dists, linewidth=2, color="#386cb0")
    axes[0].axhline(0.55, color="#d95f02", linestyle="--", label="Selected eps=0.55")
    axes[0].set_title("8-NN Sorted Distance Graph (60% Train)")
    axes[0].set_xlabel("Training Customers (Sorted)")
    axes[0].set_ylabel("8th Nearest Neighbor Distance")
    axes[0].legend()

    db_df = results["dbscan_val_grid_df"]
    piv = db_df.pivot(index="min_samples", columns="eps", values="val_noise_pct")
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1])
    axes[1].set_title("DBSCAN Validation Noise Rate (%)")
    fig.tight_layout()
    fig.savefig(plots_dir / "04_dbscan_tuning.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 5. PCA Algorithm Comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.scatterplot(data=seg_df, x="PCA1", y="PCA2", hue="SegmentPersona", palette=PERSONA_PALETTE, s=18, alpha=0.75, ax=axes[0])
    axes[0].set_title("K-Means (k=4) Personas")
    axes[0].legend(fontsize=8, loc="best")

    sns.scatterplot(data=seg_df, x="PCA1", y="PCA2", hue="Cluster_Hierarchical", palette="Set2", s=18, alpha=0.75, ax=axes[1])
    axes[1].set_title("Hierarchical Ward (k=4)")
    axes[1].legend(title="Cluster", fontsize=8, loc="best")

    db_plot = seg_df.copy()
    db_plot["DBSCAN_Group"] = db_plot["Cluster_DBSCAN"].apply(lambda c: "Noise (-1)" if c == -1 else f"Cluster {c}")
    sns.scatterplot(data=db_plot, x="PCA1", y="PCA2", hue="DBSCAN_Group", palette="tab10", s=18, alpha=0.75, ax=axes[2])
    axes[2].set_title("DBSCAN (eps=0.55, min_samples=8)")
    axes[2].legend(fontsize=8, loc="best")

    for ax in axes:
        ax.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}% var)")
    fig.suptitle("2D PCA Projection Comparison Across Three Algorithms", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "05_model_comparison_pca.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 6. Segment Snake Plot & Category Preferences
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    z_cols = ["RecencyDays", "Frequency", "Monetary", "UniqueProducts", "AvgOrderValue", "CategoryEntropy"]
    z_df = seg_df[z_cols].copy()
    for c in ["RecencyDays", "Frequency", "Monetary", "UniqueProducts", "AvgOrderValue"]:
        z_df[c] = np.log1p(z_df[c])
    z_df = (z_df - z_df.mean()) / z_df.std()
    z_df["SegmentPersona"] = seg_df["SegmentPersona"]
    melted = z_df.melt(id_vars="SegmentPersona", var_name="Feature", value_name="Standardized_Value")
    sns.lineplot(data=melted, x="Feature", y="Standardized_Value", hue="SegmentPersona", palette=PERSONA_PALETTE, marker="o", linewidth=2.2, ax=axes[0])
    axes[0].axhline(0, color="gray", linestyle=":", linewidth=1)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Cluster Snake Plot (Standardized Behavioral Profile)")
    axes[0].set_ylabel("Mean Z-Score")

    share_cols = ["Share_HomeDecor", "Share_KitchenDining", "Share_BagsStorage", "Share_SeasonalParty", "Share_CraftStationeryToys", "Share_Other"]
    cat_heat = seg_df.groupby("SegmentPersona")[share_cols].mean() * 100.0
    cat_heat.columns = [c.replace("Share_", "") for c in cat_heat.columns]
    sns.heatmap(cat_heat, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Mean Spend Share (%) by Product Category")
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(plots_dir / "06_segment_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 7. Train/Val/Test Generalization, Error Analysis & Bootstrap Stability
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    y_lower = 10
    for persona, color in PERSONA_PALETTE.items():
        vals = np.sort(seg_df.loc[seg_df["SegmentPersona"] == persona, "Silhouette_KMeans"].to_numpy())
        if len(vals) == 0:
            continue
        y_upper = y_lower + len(vals)
        axes[0].fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, alpha=0.8, label=persona)
        y_lower = y_upper + 30
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.2, label="Boundary (0.0)")
    axes[0].set_title(f"Per-Sample Silhouette ({err['negative_silhouette_pct']}% < 0)")
    axes[0].set_xlabel("Silhouette Coefficient")
    axes[0].set_yticks([])
    axes[0].legend(fontsize=7, loc="upper left")

    comp_melt = comp_df.melt(
        id_vars="algorithm",
        value_vars=["train_silhouette", "val_silhouette", "test_silhouette"],
        var_name="Split",
        value_name="Silhouette",
    )
    comp_melt["Split"] = comp_melt["Split"].map(
        {"train_silhouette": "Train (60%)", "val_silhouette": "Val (20%)", "test_silhouette": "Test (20%)"}
    )
    sns.barplot(data=comp_melt, x="algorithm", y="Silhouette", hue="Split", ax=axes[1], palette="Set2")
    axes[1].set_ylim(0, 0.38)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_title("Train / Val / Test Silhouette")
    axes[1].set_xlabel("")
    axes[1].legend(loc="upper right", fontsize=8)

    ari = results["ari_scores"]
    axes[2].bar(range(1, len(ari) + 1), ari, color="#1b9e77", alpha=0.85)
    axes[2].axhline(np.mean(ari), color="#d95f02", linestyle="--", label=f"Mean ARI = {np.mean(ari):.3f}")
    axes[2].set_ylim(0, 1.15)
    axes[2].set_title("Bootstrap Stability (10 x 80% Resamples)")
    axes[2].set_xlabel("Bootstrap Iteration")
    axes[2].set_ylabel("Adjusted Rand Index (ARI)")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.suptitle("60/20/20 Generalization, Boundary Errors & Stability", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "07_error_stability_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_training_and_evaluation(
    features_df: pd.DataFrame,
    project_root: Path,
    selected_k: int = 4,
    dbscan_eps: float = 0.55,
    dbscan_min_samples: int = 8,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Tune on 20% Validation, evaluate on 20% Test, run error & fairness audits, and save artifacts."""
    artifacts_dir = project_root / "artifacts"
    plots_dir = artifacts_dir / "plots"
    data_dir = project_root / "data"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Leak-free 60% Train / 20% Validation / 20% Test split
    train_df, val_df, test_df = split_data(features_df, random_state=random_state)
    _, X_train, X_val, X_test = fit_transform_splits(train_df, val_df, test_df)

    # 2. Feature-space ablation on Train/Test
    ablation_specs = {
        "RFM Only (3 features)": ["RecencyDays", "Frequency", "Monetary"],
        "RFM + Product Breadth (4 features)": MODEL_FEATURES,
        "RFM + Breadth + Category Shares (11 features)": [
            "RecencyDays",
            "Frequency",
            "Monetary",
            "UniqueProducts",
            "CategoryEntropy",
            "Share_HomeDecor",
            "Share_KitchenDining",
            "Share_BagsStorage",
            "Share_SeasonalParty",
            "Share_CraftStationeryToys",
            "Share_Other",
        ],
    }
    ablation_rows = []
    for spec_name, cols in ablation_specs.items():
        tr_sub = train_df[cols].copy()
        te_sub = test_df[cols].copy()
        for c in cols:
            if c in {"RecencyDays", "Frequency", "Monetary", "UniqueProducts"}:
                tr_sub[c] = np.log1p(tr_sub[c])
                te_sub[c] = np.log1p(te_sub[c])
        ab_scaler = StandardScaler().fit(tr_sub.to_numpy(dtype=float))
        km_ab = KMeans(n_clusters=selected_k, n_init=20, random_state=random_state).fit(
            ab_scaler.transform(tr_sub.to_numpy(dtype=float))
        )
        m = compute_metrics(ab_scaler.transform(te_sub.to_numpy(dtype=float)), km_ab.predict(ab_scaler.transform(te_sub.to_numpy(dtype=float))))
        m["feature_set"] = spec_name
        ablation_rows.append(m)
    ablation_df = pd.DataFrame(ablation_rows)

    # 3. Hyperparameter tuning on Validation split (k = 2..10, eps/min_samples grid)
    kmeans_val_sweep = []
    hier_val_sweep = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit(X_train)
        km_val_m = compute_metrics(X_val, km.predict(X_val))
        kmeans_val_sweep.append(
            {
                "k": k,
                "train_inertia": round(float(km.inertia_), 2),
                "val_silhouette": km_val_m["silhouette"],
                "val_davies_bouldin": km_val_m["davies_bouldin"],
                "val_calinski_harabasz": km_val_m["calinski_harabasz"],
            }
        )

        hc_tr_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_train)
        hc_nc = NearestCentroid().fit(X_train, hc_tr_labels)
        hc_val_m = compute_metrics(X_val, hc_nc.predict(X_val))
        hier_val_sweep.append(
            {
                "k": k,
                "val_silhouette": hc_val_m["silhouette"],
                "val_davies_bouldin": hc_val_m["davies_bouldin"],
                "val_calinski_harabasz": hc_val_m["calinski_harabasz"],
            }
        )

    ward_linkage = linkage(X_train, method="ward")

    nn_k = NearestNeighbors(n_neighbors=dbscan_min_samples).fit(X_train)
    k_dists, _ = nn_k.kneighbors(X_train)
    sorted_k_dists = np.sort(k_dists[:, -1])

    dbscan_val_grid = []
    for eps in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for ms in [5, 8, 12, 16]:
            db_tr_lbl = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_train)
            db_val_lbl = predict_dbscan_out_of_sample(X_train, db_tr_lbl, X_val, eps=eps)
            db_val_m = compute_metrics(X_val, db_val_lbl)
            dbscan_val_grid.append(
                {
                    "eps": eps,
                    "min_samples": ms,
                    "val_clusters": db_val_m["n_clusters"],
                    "val_noise_pct": db_val_m["noise_pct"],
                    "val_silhouette": db_val_m["silhouette"],
                }
            )

    # 4. Final Tuned Model Comparison across Train (60%), Validation (20%), Held-Out Test (20%)
    km_model = KMeans(n_clusters=selected_k, n_init=30, random_state=random_state).fit(X_train)
    km_tr, km_va, km_te = km_model.labels_, km_model.predict(X_val), km_model.predict(X_test)

    hc_tr = AgglomerativeClustering(n_clusters=selected_k, linkage="ward").fit_predict(X_train)
    hc_nc = NearestCentroid().fit(X_train, hc_tr)
    hc_va, hc_te = hc_nc.predict(X_val), hc_nc.predict(X_test)

    db_tr = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X_train)
    db_va = predict_dbscan_out_of_sample(X_train, db_tr, X_val, eps=dbscan_eps)
    db_te = predict_dbscan_out_of_sample(X_train, db_tr, X_test, eps=dbscan_eps)

    comparison_rows = []
    for algo_name, tr_l, va_l, te_l in [
        (f"K-Means (k={selected_k})", km_tr, km_va, km_te),
        (f"Hierarchical Ward (k={selected_k})", hc_tr, hc_va, hc_te),
        (f"DBSCAN (eps={dbscan_eps}, min_samples={dbscan_min_samples})", db_tr, db_va, db_te),
    ]:
        m_tr = compute_metrics(X_train, tr_l)
        m_va = compute_metrics(X_val, va_l)
        m_te = compute_metrics(X_test, te_l)
        comparison_rows.append(
            {
                "algorithm": algo_name,
                "train_clusters": m_tr["n_clusters"],
                "train_noise_pct": m_tr["noise_pct"],
                "train_silhouette": m_tr["silhouette"],
                "val_silhouette": m_va["silhouette"],
                "test_silhouette": m_te["silhouette"],
                "test_davies_bouldin": m_te["davies_bouldin"],
                "test_calinski_harabasz": m_te["calinski_harabasz"],
                "test_noise_pct": m_te["noise_pct"],
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    # 5. Bootstrap Stability (10 x 80% of Train)
    rng = np.random.default_rng(random_state)
    ari_scores: List[float] = []
    sample_size = int(0.8 * len(X_train))
    for b in range(10):
        idx = rng.choice(len(X_train), size=sample_size, replace=False)
        km_boot = KMeans(n_clusters=selected_k, n_init=20, random_state=random_state + b + 1).fit(X_train[idx])
        ari_scores.append(float(adjusted_rand_score(km_tr, km_boot.predict(X_train))))

    # 6. Full-base production fit + Persona profiling + Error & Geographic Fairness audit
    full_scaler = StandardScaler().fit(log_transform(features_df))
    X_full = full_scaler.transform(log_transform(features_df))
    final_km = KMeans(n_clusters=selected_k, n_init=30, random_state=random_state).fit(X_full)
    full_km_labels = final_km.labels_
    full_hc_labels = AgglomerativeClustering(n_clusters=selected_k, linkage="ward").fit_predict(X_full)
    full_db_labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X_full)

    pca_2d = PCA(n_components=2, random_state=random_state).fit(X_full)
    X_pca = pca_2d.transform(X_full)

    segmented_df = features_df.copy()
    segmented_df["Cluster_KMeans"] = full_km_labels
    segmented_df["Cluster_Hierarchical"] = full_hc_labels
    segmented_df["Cluster_DBSCAN"] = full_db_labels
    segmented_df["PCA1"] = X_pca[:, 0]
    segmented_df["PCA2"] = X_pca[:, 1]
    segmented_df["Silhouette_KMeans"] = silhouette_samples(X_full, full_km_labels)

    persona_map = assign_semantic_personas(segmented_df, "Cluster_KMeans")
    segmented_df["SegmentPersona"] = segmented_df["Cluster_KMeans"].map(persona_map)

    neg_sil_mask = segmented_df["Silhouette_KMeans"] < 0
    db_noise_mask = segmented_df["Cluster_DBSCAN"] == -1

    # Geographic fairness audit (UK vs non-UK kept out of training features)
    fairness_audit = {
        "uk_overall_pct": round(float(segmented_df["IsUK"].mean() * 100.0), 2),
        "uk_boundary_error_pct": round(float(neg_sil_mask[segmented_df["IsUK"] == 1].mean() * 100.0), 2),
        "non_uk_boundary_error_pct": round(float(neg_sil_mask[segmented_df["IsUK"] == 0].mean() * 100.0), 2),
        "uk_vip_rate_pct": round(float((segmented_df.loc[segmented_df["IsUK"] == 1, "SegmentPersona"] == "Champions (VIP)").mean() * 100.0), 2),
        "non_uk_vip_rate_pct": round(float((segmented_df.loc[segmented_df["IsUK"] == 0, "SegmentPersona"] == "Champions (VIP)").mean() * 100.0), 2),
    }

    error_analysis = {
        "total_customers": int(len(segmented_df)),
        "split_sizes": {"train_60": len(train_df), "val_20": len(val_df), "test_20": len(test_df)},
        "negative_silhouette_count": int(neg_sil_mask.sum()),
        "negative_silhouette_pct": round(float(neg_sil_mask.mean() * 100.0), 2),
        "boundary_customer_medians": segmented_df.loc[neg_sil_mask, MODEL_FEATURES].median().round(2).to_dict(),
        "core_customer_medians": segmented_df.loc[~neg_sil_mask, MODEL_FEATURES].median().round(2).to_dict(),
        "dbscan_noise_count": int(db_noise_mask.sum()),
        "dbscan_noise_pct": round(float(db_noise_mask.mean() * 100.0), 2),
        "dbscan_noise_medians": segmented_df.loc[db_noise_mask, MODEL_FEATURES].median().round(2).to_dict(),
        "bootstrap_ari_mean": round(float(np.mean(ari_scores)), 4),
        "bootstrap_ari_std": round(float(np.std(ari_scores)), 4),
        "bootstrap_ari_scores": [round(s, 4) for s in ari_scores],
        "fairness_audit": fairness_audit,
    }

    profile_table = (
        segmented_df.groupby("SegmentPersona")
        .agg(
            Customers=("CustomerID", "count"),
            SharePct=("CustomerID", lambda s: round(len(s) / len(segmented_df) * 100.0, 2)),
            MedianRecency=("RecencyDays", "median"),
            MedianFrequency=("Frequency", "median"),
            MedianMonetary=("Monetary", "median"),
            TotalRevenue=("Monetary", "sum"),
            RevenueSharePct=("Monetary", lambda s: round(s.sum() / segmented_df["Monetary"].sum() * 100.0, 2)),
            MedianUniqueProducts=("UniqueProducts", "median"),
            MeanCategoryEntropy=("CategoryEntropy", "mean"),
            UKSharePct=("IsUK", lambda s: round(s.mean() * 100.0, 2)),
        )
        .round(2)
        .reset_index()
    )

    segmented_df.to_csv(data_dir / "segmented_customers.csv", index=False)

    pipeline_bundle = {
        "feature_names": MODEL_FEATURES,
        "scaler": full_scaler,
        "kmeans_model": final_km,
        "pca_2d": pca_2d,
        "persona_map": persona_map,
        "persona_actions": PERSONA_ACTIONS,
    }
    with open(artifacts_dir / "segmentation_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline_bundle, f)

    metrics_payload = {
        "ablation": ablation_df.to_dict(orient="records"),
        "kmeans_val_sweep": kmeans_val_sweep,
        "hier_val_sweep": hier_val_sweep,
        "dbscan_val_grid": dbscan_val_grid,
        "train_val_test_comparison": comparison_df.to_dict(orient="records"),
        "error_analysis": error_analysis,
        "segment_profiles": profile_table.to_dict(orient="records"),
        "pca_explained_variance_ratio": [round(float(v), 4) for v in pca_2d.explained_variance_ratio_],
    }
    with open(artifacts_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, allow_nan=False)

    results = {
        "segmented_df": segmented_df,
        "ablation_df": ablation_df,
        "kmeans_val_sweep_df": pd.DataFrame(kmeans_val_sweep),
        "hier_val_sweep_df": pd.DataFrame(hier_val_sweep),
        "dbscan_val_grid_df": pd.DataFrame(dbscan_val_grid),
        "comparison_df": comparison_df,
        "profile_table": profile_table,
        "error_analysis": error_analysis,
        "ward_linkage": ward_linkage,
        "sorted_k_dists": sorted_k_dists,
        "pca_explained_variance_ratio": pca_2d.explained_variance_ratio_,
        "ari_scores": ari_scores,
        "pipeline_bundle": pipeline_bundle,
    }
    save_plots(results, features_df, plots_dir)
    return results


def predict_new_customer(
    pipeline_bundle: Dict[str, Any],
    recency_days: float,
    frequency: float,
    monetary: float,
    unique_products: float,
    dominant_category: str = "HomeDecor",
) -> Dict[str, Any]:
    """Classify a customer profile into a retail segment with match confidence and tips."""
    raw = np.array([[recency_days, frequency, monetary, unique_products]], dtype=float)
    log_raw = np.log1p(np.clip(raw, a_min=0.0, a_max=None))
    scaled = pipeline_bundle["scaler"].transform(log_raw)

    km: KMeans = pipeline_bundle["kmeans_model"]
    cluster_id = int(km.predict(scaled)[0])
    dists = km.transform(scaled)[0]
    inv_d = 1.0 / (dists + 1e-6)
    probs = inv_d / inv_d.sum()

    persona = pipeline_bundle["persona_map"][cluster_id]
    action_info = pipeline_bundle["persona_actions"].get(persona, {})
    pca_coords = pipeline_bundle["pca_2d"].transform(scaled)[0]

    category_tips = {
        "HomeDecor": "Feature seasonal lighting, wall plaques, and candle holders in email hero banners.",
        "KitchenDining": "Highlight ceramic tea sets, pantry tins, and baking accessories as bundle add-ons.",
        "BagsStorage": "Promote reusable jumbo shoppers, lunch bags, and coordinated storage sets.",
        "SeasonalParty": "Time outreach 3-4 weeks ahead of peak gifting/party periods with bunting & gift wrap.",
        "CraftStationeryToys": "Cross-sell giftable notebooks, craft kits, and retro games.",
        "Other": "Surface storewide bestsellers to discover stronger category affinity.",
    }

    return {
        "cluster_id": cluster_id,
        "segment_persona": persona,
        "confidence_pct": round(float(probs[cluster_id] * 100.0), 1),
        "centroid_distance": round(float(dists[cluster_id]), 3),
        "pca_coords": (round(float(pca_coords[0]), 3), round(float(pca_coords[1]), 3)),
        "persona_summary": action_info.get("summary", ""),
        "recommended_strategy": action_info.get("action", ""),
        "category_merchandising_tip": category_tips.get(dominant_category, category_tips["Other"]),
        "all_segment_affinities_pct": {
            pipeline_bundle["persona_map"][i]: round(float(probs[i] * 100.0), 1)
            for i in range(len(probs))
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    clean_df, features_df, audit = load_and_cache_data(project_root, force_rebuild=False)
    print("Data Audit:", audit)
    print(f"Loaded {len(clean_df):,} clean transactions across {len(features_df):,} unique customers.")

    results = run_training_and_evaluation(features_df, project_root)
    print("\n=== 60/20/20 Train / Validation / Test Comparison ===")
    print(results["comparison_df"].to_string(index=False))
    print("\n=== Segment Profiles ===")
    print(results["profile_table"].to_string(index=False))
    print("\n=== Error & Fairness Audit ===")
    for k, v in results["error_analysis"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
