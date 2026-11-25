"""Streamlit Decision-Support App for Retail Customer Segmentation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CATEGORY_NAMES
from src.train_evaluate import PERSONA_PALETTE, predict_new_customer

st.set_page_config(
    page_title="Retail Customer Segmentation Studio",
    page_icon="🛍️",
    layout="wide",
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODEL_PATH = ARTIFACTS_DIR / "segmentation_pipeline.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics_summary.json"
SEGMENTED_CSV = PROJECT_ROOT / "data" / "segmented_customers.csv"

PRESETS = {
    "Custom Input": None,
    "Boutique VIP Gift Shop": dict(recency=7, frequency=14, monetary=4850, unique_products=145, category="HomeDecor"),
    "Repeat Kitchen Enthusiast": dict(recency=38, frequency=4, monetary=920, unique_products=42, category="KitchenDining"),
    "First-Time Seasonal Buyer": dict(recency=15, frequency=1, monetary=165, unique_products=12, category="SeasonalParty"),
    "Lapsed Single-Order Customer": dict(recency=220, frequency=1, monetary=110, unique_products=8, category="BagsStorage"),
}


@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    seg_df = pd.read_csv(SEGMENTED_CSV)
    return bundle, metrics, seg_df


if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not SEGMENTED_CSV.exists():
    st.error("Model artifacts not found. Run `python -m src.train_evaluate` first.")
    st.stop()

bundle, metrics, seg_df = load_artifacts()
err = metrics["error_analysis"]

# --- SIDEBAR ---
st.sidebar.title("Customer Profile Inputs")
preset_choice = st.sidebar.selectbox("Sample Presets", list(PRESETS.keys()), index=2)
preset_vals = PRESETS[preset_choice]

default_r = preset_vals["recency"] if preset_vals else 25
default_f = preset_vals["frequency"] if preset_vals else 5
default_m = preset_vals["monetary"] if preset_vals else 1500
default_p = preset_vals["unique_products"] if preset_vals else 48
default_c = preset_vals["category"] if preset_vals else "KitchenDining"

recency_days = st.sidebar.slider("Recency (Days Since Last Order)", min_value=1, max_value=374, value=default_r)
frequency = st.sidebar.slider("Purchase Frequency (Orders)", min_value=1, max_value=60, value=default_f)
monetary = st.sidebar.number_input("Total Spending (GBP £)", min_value=1.0, max_value=300000.0, value=float(default_m), step=25.0)
unique_products = st.sidebar.slider("Unique Products Purchased", min_value=1, max_value=350, value=default_p)
dominant_category = st.sidebar.selectbox(
    "Preferred Product Category",
    CATEGORY_NAMES,
    index=CATEGORY_NAMES.index(default_c) if default_c in CATEGORY_NAMES else 0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Decision-Support Disclaimer:** This tool groups customers using historical transaction patterns "
    "(2010–2011 UCI Online Retail data) to assist marketing and merchandising teams. Outputs are "
    "advisory recommendations—not automated pricing or credit decisions—and require human review."
)

# --- HEADER & KPIs ---
st.title("Retail Customer Segmentation Studio")
st.caption("K-Means, Hierarchical Ward & DBSCAN compared on a 60% / 20% / 20% Train / Validation / Test split (4,334 UCI customers).")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cleaned Customers", f"{err['total_customers']:,}", "391,283 valid orders")
k2.metric("VIP Revenue Share", "67.84%", "18.71% of customer base")
k3.metric("Test Silhouette (K-Means)", "0.2799", "60/20/20 held-out test")
k4.metric("Bootstrap Stability (ARI)", f"{err['bootstrap_ari_mean']:.3f}", f"std = {err['bootstrap_ari_std']:.4f}")

# --- PREDICTION ---
pred = predict_new_customer(
    bundle,
    recency_days=recency_days,
    frequency=frequency,
    monetary=monetary,
    unique_products=unique_products,
    dominant_category=dominant_category,
)

tab_pred, tab_exp, tab_eval, tab_plots = st.tabs(
    ["Segment Predictor", "Customer Explorer", "Model Evaluation & Error Analysis", "Diagnostic Plots"]
)

with tab_pred:
    left, right = st.columns([1.05, 1.25])

    with left:
        st.subheader(f"Assigned Segment: {pred['segment_persona']}")
        st.write(pred["persona_summary"])
        m1, m2 = st.columns(2)
        m1.metric("Match Confidence", f"{pred['confidence_pct']}%")
        m2.metric("Centroid Distance", f"{pred['centroid_distance']}")

        st.success(f"**Marketing Action:** {pred['recommended_strategy']}")
        st.info(f"**Category Merchandising Tip ({dominant_category}):** {pred['category_merchandising_tip']}")

        st.markdown("**Segment Affinity Breakdown**")
        aff_df = pd.DataFrame(
            sorted(pred["all_segment_affinities_pct"].items(), key=lambda x: -x[1]),
            columns=["Segment", "Affinity (%)"],
        )
        st.dataframe(aff_df, hide_index=True, width="stretch")

    with right:
        st.subheader("Live 2D PCA Cluster Map")
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for persona, color in PERSONA_PALETTE.items():
            sub = seg_df[seg_df["SegmentPersona"] == persona]
            ax.scatter(sub["PCA1"], sub["PCA2"], s=14, alpha=0.45, color=color, label=persona)
        qx, qy = pred["pca_coords"]
        ax.scatter([qx], [qy], s=260, marker="*", color="#0f172a", edgecolor="white", linewidth=1.5, zorder=10, label=f"Input ({qx}, {qy})")
        pca_var = metrics["pca_explained_variance_ratio"]
        ax.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}% variance)")
        ax.set_title("Customer Space with Queried Profile Star")
        ax.legend(fontsize=8, loc="best")
        st.pyplot(fig)
        plt.close(fig)

with tab_exp:
    st.subheader("Explore 4,334 Segmented UCI Customers")
    f1, f2, f3 = st.columns(3)
    seg_filter = f1.selectbox("Filter by Segment", ["All"] + list(PERSONA_PALETTE.keys()))
    cat_filter = f2.selectbox("Filter by Top Category", ["All"] + CATEGORY_NAMES)
    noise_only = f3.checkbox("Show DBSCAN Wholesale/Outliers Only (-1)")

    view_df = seg_df.copy()
    if seg_filter != "All":
        view_df = view_df[view_df["SegmentPersona"] == seg_filter]
    if cat_filter != "All":
        view_df = view_df[view_df["DominantCategory"] == cat_filter]
    if noise_only:
        view_df = view_df[view_df["Cluster_DBSCAN"] == -1]

    display_cols = [
        "CustomerID", "PrimaryCountry", "SegmentPersona", "RecencyDays",
        "Frequency", "Monetary", "UniqueProducts", "DominantCategory", "Silhouette_KMeans",
    ]
    st.write(f"Showing {min(len(view_df), 200)} of {len(view_df):,} matching customers.")
    st.dataframe(view_df[display_cols].head(200), hide_index=True, width="stretch")

with tab_eval:
    st.subheader("Segment Profiles & Revenue Share")
    st.dataframe(pd.DataFrame(metrics["segment_profiles"]), hide_index=True, width="stretch")

    st.subheader("Algorithm Comparison (60% Train / 20% Validation / 20% Test)")
    st.dataframe(pd.DataFrame(metrics["train_val_test_comparison"]), hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Feature-Space Ablation (Held-Out Test)")
        st.dataframe(pd.DataFrame(metrics["ablation"]), hide_index=True, width="stretch")
    with c2:
        st.subheader("Error & Geographic Fairness Audit")
        fair = err["fairness_audit"]
        st.write(
            f"- **Boundary Customers (Silhouette < 0):** {err['negative_silhouette_count']} "
            f"({err['negative_silhouette_pct']}%) — median Spend £{err['boundary_customer_medians']['Monetary']:,.2f}, "
            f"Frequency {err['boundary_customer_medians']['Frequency']}."
        )
        st.write(
            f"- **DBSCAN Noise Outliers (-1):** {err['dbscan_noise_count']} ({err['dbscan_noise_pct']}%) — "
            f"median Spend £{err['dbscan_noise_medians']['Monetary']:,.2f} across "
            f"{err['dbscan_noise_medians']['UniqueProducts']} unique products (wholesale/bulk buyers)."
        )
        st.write(
            f"- **Geographic Audit (UK vs. Non-UK, excluded from model inputs):** "
            f"UK share = {fair['uk_overall_pct']}%; boundary error rate = {fair['uk_boundary_error_pct']}% (UK) vs. "
            f"{fair['non_uk_boundary_error_pct']}% (non-UK)."
        )

with tab_plots:
    st.subheader("Saved Comparison & Diagnostic Plots (`artifacts/plots/`)")
    plot_files = sorted(PLOTS_DIR.glob("*.png"))
    for i in range(0, len(plot_files), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(plot_files):
                p = plot_files[i + j]
                col.image(str(p), caption=p.name, width="stretch")
