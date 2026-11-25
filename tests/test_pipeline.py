"""Automated pytest suite verifying data loading, leak-free splitting, training, and inference."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import CATEGORY_NAMES, build_customer_features, clean_transactions, load_and_cache_data
from src.preprocessing import AUDIT_COLUMNS, MODEL_FEATURES, fit_transform_splits, split_data
from src.train_evaluate import predict_new_customer, run_training_and_evaluation


def test_clean_transactions_filters_invalid_and_non_product():
    raw = pd.DataFrame(
        [
            {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "Quantity": 6, "InvoiceDate": "2010-12-01 08:26", "UnitPrice": 2.55, "CustomerID": 17850.0, "Country": "United Kingdom"},
            {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HANGING HEART T-LIGHT HOLDER", "Quantity": 6, "InvoiceDate": "2010-12-01 08:26", "UnitPrice": 2.55, "CustomerID": 17850.0, "Country": "United Kingdom"},  # duplicate
            {"InvoiceNo": "C536379", "StockCode": "D", "Description": "Discount", "Quantity": -1, "InvoiceDate": "2010-12-01 09:41", "UnitPrice": 27.5, "CustomerID": 14527.0, "Country": "United Kingdom"},  # cancelled
            {"InvoiceNo": "536400", "StockCode": "POST", "Description": "POSTAGE", "Quantity": 1, "InvoiceDate": "2010-12-01 10:00", "UnitPrice": 18.0, "CustomerID": 12583.0, "Country": "France"},  # non-product
            {"InvoiceNo": "536401", "StockCode": "22423", "Description": "REGENCY CAKESTAND 3 TIER", "Quantity": 2, "InvoiceDate": "2010-12-01 10:15", "UnitPrice": 12.75, "CustomerID": None, "Country": "United Kingdom"},  # missing CustomerID
        ]
    )
    clean_df, audit = clean_transactions(raw)
    assert len(clean_df) == 1
    assert audit["raw_rows"] == 5
    assert clean_df.iloc[0]["ProductCategory"] == "HomeDecor"
    assert np.isclose(clean_df.iloc[0]["TotalAmount"], 15.30)

    feat_df = build_customer_features(clean_df)
    assert len(feat_df) == 1
    assert feat_df.iloc[0]["DominantCategory"] == "HomeDecor"
    assert feat_df.iloc[0]["IsUK"] == 1


def test_split_proportions_and_leak_free_scaling():
    _, features_df, _ = load_and_cache_data(PROJECT_ROOT, force_rebuild=False)
    train_df, val_df, test_df = split_data(features_df, random_state=42)

    total = len(features_df)
    assert len(train_df) + len(val_df) + len(test_df) == total
    assert abs(len(train_df) / total - 0.60) < 0.01
    assert abs(len(val_df) / total - 0.20) < 0.01
    assert abs(len(test_df) / total - 0.20) < 0.01

    # Sensitive/ID columns must never be inside MODEL_FEATURES
    for audit_col in AUDIT_COLUMNS:
        assert audit_col not in MODEL_FEATURES

    scaler, X_train, X_val, X_test = fit_transform_splits(train_df, val_df, test_df)
    assert X_train.shape == (len(train_df), len(MODEL_FEATURES))
    assert np.allclose(X_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X_train.std(axis=0), 1.0, atol=1e-6)
    assert np.isfinite(X_val).all() and np.isfinite(X_test).all()
    assert scaler.n_features_in_ == len(MODEL_FEATURES)


def test_end_to_end_training_and_prediction(tmp_path: Path):
    _, features_df, _ = load_and_cache_data(PROJECT_ROOT, force_rebuild=False)
    # Run full pipeline into tmp_path to avoid mutating repo artifacts during unit test
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    results = run_training_and_evaluation(features_df, tmp_path, selected_k=4)

    comp = results["comparison_df"]
    assert len(comp) == 3
    assert set(comp["algorithm"]) == {
        "K-Means (k=4)",
        "Hierarchical Ward (k=4)",
        "DBSCAN (eps=0.55, min_samples=8)",
    }
    assert (tmp_path / "artifacts" / "segmentation_pipeline.pkl").exists()
    assert (tmp_path / "artifacts" / "metrics_summary.json").exists()
    assert len(list((tmp_path / "artifacts" / "plots").glob("*.png"))) == 7

    pred = predict_new_customer(
        results["pipeline_bundle"],
        recency_days=10,
        frequency=12,
        monetary=4500.0,
        unique_products=130,
        dominant_category="HomeDecor",
    )
    assert pred["segment_persona"] == "Champions (VIP)"
    assert 0.0 < pred["confidence_pct"] <= 100.0
    assert abs(sum(pred["all_segment_affinities_pct"].values()) - 100.0) < 0.5
    assert set(CATEGORY_NAMES) >= {"HomeDecor", "KitchenDining", "BagsStorage", "SeasonalParty", "CraftStationeryToys", "Other"}
