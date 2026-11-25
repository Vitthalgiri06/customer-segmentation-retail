"""Dataset download, local caching, transaction cleaning, and customer feature aggregation."""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

UCI_ONLINE_RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

NON_PRODUCT_CODES = {
    "POST",
    "DOT",
    "M",
    "D",
    "S",
    "AMAZONFEE",
    "CRUK",
    "B",
    "m",
    "PADS",
    "BANK CHARGES",
}

CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "HomeDecor": (
        "LIGHT",
        "CANDLE",
        "LANTERN",
        "HOLDER",
        "CHALKBOARD",
        "FRAME",
        "CLOCK",
        "SIGN",
        "MIRROR",
        "ORNAMENT",
        "VASE",
        "WICKER",
        "CUSHION",
        "DOORMAT",
        "WALL",
        "HANGING",
        "LAMP",
        "TRINKET",
        "HEART",
    ),
    "KitchenDining": (
        "CAKE",
        "PANTRY",
        "MUG",
        "CUP",
        "TEA",
        "COFFEE",
        "JAM",
        "BAKING",
        "RECIPE",
        "BOWL",
        "PLATE",
        "JAR",
        "GLASS",
        "SPOON",
        "CUTLERY",
        "TIN",
        "TRAY",
        "DOILY",
        "KITCHEN",
        "APRON",
        "BOTTLE",
        "FLASK",
        "COASTER",
        "NAPKIN",
        "MILK",
        "SUGAR",
    ),
    "BagsStorage": (
        "BAG",
        "SHOPPER",
        "LUNCH",
        "POUCH",
        "PURSE",
        "TOTE",
        "BOX",
        "CASE",
        "BASKET",
        "STORAGE",
        "SATCHEL",
        "RUCKSACK",
        "WALLET",
        "SUITCASE",
    ),
    "SeasonalParty": (
        "CHRISTMAS",
        "XMAS",
        "PARTY",
        "BUNTING",
        "GIFT",
        "RIBBON",
        "WRAP",
        "TAG",
        "BALLOON",
        "BIRTHDAY",
        "TREE",
        "EASTER",
        "HALLOWEEN",
        "ADVENT",
        "SNOWMAN",
        "REINDEER",
        "SANTA",
        "GARLAND",
        "CRACKER",
    ),
    "CraftStationeryToys": (
        "NOTEBOOK",
        "PEN",
        "PENCIL",
        "PAPER",
        "STICKER",
        "TAPE",
        "CRAFT",
        "SEWING",
        "KNITTING",
        "TOY",
        "GAME",
        "PUZZLE",
        "DOLL",
        "KEYRING",
        "CARD",
        "BOOK",
        "DRAWING",
        "PAINT",
        "SCISSORS",
        "BEAD",
        "BADGE",
    ),
}

CATEGORY_NAMES = list(CATEGORY_KEYWORDS.keys()) + ["Other"]


def categorize_description(description: object) -> str:
    """Map a free-text item description to a retail product category."""
    if not isinstance(description, str) or not description.strip():
        return "Other"
    tokens = re.sub(r"[^A-Z0-9]+", " ", description.upper()).strip()
    padded = f" {tokens} "
    word_set = set(tokens.split())

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in word_set for kw in keywords):
            return category
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in padded for kw in keywords):
            return category
    return "Other"


def ensure_raw_dataset(data_dir: Path) -> Path:
    """Ensure the UCI Online Retail Excel file is cached in data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_xlsx = data_dir / "Online Retail.xlsx"
    if raw_xlsx.exists():
        return raw_xlsx

    zip_path = data_dir / "online_retail.zip"
    if not zip_path.exists():
        with urllib.request.urlopen(UCI_ONLINE_RETAIL_URL, timeout=60) as resp:
            zip_bytes = resp.read()
        zip_path.write_bytes(zip_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        zf.extractall(data_dir)
    return raw_xlsx


def clean_transactions(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean raw UCI Online Retail line items and return audit counts."""
    audit: Dict[str, int] = {"raw_rows": int(len(raw_df))}

    df = raw_df.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()

    # Remove unattributed guest transactions and exact duplicates
    df = df.dropna(subset=["CustomerID"]).copy()
    audit["after_dropna_customer"] = int(len(df))

    df = df.drop_duplicates().copy()
    audit["after_dedup"] = int(len(df))

    # Remove cancellations ('C' prefix) and non-positive quantities/unit prices
    df = df[~df["InvoiceNo"].str.startswith("C")].copy()
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    audit["after_valid_qty_price"] = int(len(df))

    # Remove non-product accounting/postage codes
    df = df[~df["StockCode"].isin(NON_PRODUCT_CODES)].copy()
    audit["after_product_filter"] = int(len(df))

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df["ProductCategory"] = df["Description"].map(categorize_description)

    audit["unique_customers"] = int(df["CustomerID"].nunique())
    audit["unique_invoices"] = int(df["InvoiceNo"].nunique())
    audit["unique_products"] = int(df["StockCode"].nunique())
    return df, audit


def build_customer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions into customer-level RFM and product preference records."""
    snapshot_date = clean_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        clean_df.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalAmount", "sum"),
            TotalQuantity=("Quantity", "sum"),
            UniqueProducts=("StockCode", "nunique"),
            PrimaryCountry=("Country", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )

    rfm["RecencyDays"] = (snapshot_date - rfm["LastPurchase"]).dt.days.astype(int)
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["AvgItemsPerOrder"] = rfm["TotalQuantity"] / rfm["Frequency"]
    # Retained strictly for geographic/fairness auditing; excluded from clustering features
    rfm["IsUK"] = (rfm["PrimaryCountry"] == "United Kingdom").astype(int)

    cat_spend = (
        clean_df.groupby(["CustomerID", "ProductCategory"])["TotalAmount"]
        .sum()
        .unstack(fill_value=0.0)
    )
    for cat in CATEGORY_NAMES:
        if cat not in cat_spend.columns:
            cat_spend[cat] = 0.0
    cat_spend = cat_spend[CATEGORY_NAMES]

    cat_shares = cat_spend.div(cat_spend.sum(axis=1), axis=0).fillna(0.0)
    eps = 1e-12
    probs = cat_shares.to_numpy()
    raw_entropy = -np.sum(np.where(probs > 0, probs * np.log(probs + eps), 0.0), axis=1)
    norm_entropy = raw_entropy / np.log(len(CATEGORY_NAMES))

    cat_shares_renamed = cat_shares.rename(
        columns={cat: f"Share_{cat}" for cat in CATEGORY_NAMES}
    )
    cat_shares_renamed["CategoryEntropy"] = norm_entropy
    cat_shares_renamed["DominantCategory"] = cat_shares.idxmax(axis=1)

    features = rfm.merge(cat_shares_renamed, on="CustomerID", how="left")
    features = features.drop(columns=["LastPurchase", "TotalQuantity"])
    return features


def load_and_cache_data(
    project_root: Path, force_rebuild: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Load locally cached cleaned data and customer features, or build from raw Excel."""
    data_dir = project_root / "data"
    clean_csv = data_dir / "cleaned_transactions.csv"
    features_csv = data_dir / "customer_features.csv"

    if not force_rebuild and clean_csv.exists() and features_csv.exists():
        clean_df = pd.read_csv(clean_csv, parse_dates=["InvoiceDate"])
        features_df = pd.read_csv(features_csv)
        audit = {
            "after_product_filter": int(len(clean_df)),
            "unique_customers": int(features_df["CustomerID"].nunique()),
        }
        return clean_df, features_df, audit

    raw_xlsx = ensure_raw_dataset(data_dir)
    raw_df = pd.read_excel(raw_xlsx)
    clean_df, audit = clean_transactions(raw_df)
    features_df = build_customer_features(clean_df)

    clean_df.to_csv(clean_csv, index=False)
    features_df.to_csv(features_csv, index=False)
    return clean_df, features_df, audit
