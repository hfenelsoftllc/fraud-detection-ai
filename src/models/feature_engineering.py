"""
Feature engineering for the fraud detection model.
Transforms raw transaction dicts/DataFrames into model-ready feature matrices.
"""

import numpy as np
import pandas as pd
from typing import Union

HIGH_RISK_MERCHANTS = frozenset(["QuickCash", "PayDay", "LoanShark", "ShadyMerchants"])
HIGH_RISK_COUNTRIES = frozenset(
    [
        "Nigeria",
        "Somalia",
        "Yemen",
        "Venezuela",
        "Afghanistan",
        "Syria",
        "North Korea",
        "Iran",
        "Iraq",
        "Libya",
        "Sudan",
    ]
)

TRANSACTION_TYPE_MAP = {
    "purchase": 0,
    "cash_withdrawal": 1,
    "card_testing": 2,
    "chargeback": 3,
    "refund": 4,
}

FEATURE_COLUMNS = [
    "amount",
    "log_amount",
    "is_high_amount",
    "txn_type_encoded",
    "is_cash_withdrawal",
    "is_card_testing",
    "is_chargeback",
    "is_refund",
    "is_high_risk_merchant",
    "is_high_risk_country",
    "user_segment",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]


def extract_features_from_dict(transaction: dict) -> np.ndarray:
    """Extract a feature vector from a single transaction dict."""
    amount = float(transaction.get("amount", 0.0))
    txn_type = transaction.get("transaction_type", "purchase")
    merchant = transaction.get("merchant", "")
    merchant_country = transaction.get("merchant_country", "")
    user_id = int(transaction.get("user_id", 0))
    event_time = transaction.get("event_time", "")

    try:
        dt = pd.to_datetime(event_time)
        hour = dt.hour
        dow = dt.dayofweek
        is_weekend = 1 if dow >= 5 else 0
    except Exception:
        hour, dow, is_weekend = 12, 0, 0

    features = [
        amount,
        np.log1p(amount),
        1.0 if amount > 1000 else 0.0,
        float(TRANSACTION_TYPE_MAP.get(txn_type, 0)),
        1.0 if txn_type == "cash_withdrawal" else 0.0,
        1.0 if txn_type == "card_testing" else 0.0,
        1.0 if txn_type == "chargeback" else 0.0,
        1.0 if txn_type == "refund" else 0.0,
        1.0 if merchant in HIGH_RISK_MERCHANTS else 0.0,
        1.0 if merchant_country in HIGH_RISK_COUNTRIES else 0.0,
        float(user_id % 100),
        float(hour),
        float(dow),
        float(is_weekend),
    ]
    return np.array(features, dtype=np.float32)


def extract_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Extract feature matrix from a transaction DataFrame."""
    out = pd.DataFrame()

    out["amount"] = df["amount"].astype(float)
    out["log_amount"] = np.log1p(out["amount"])
    out["is_high_amount"] = (out["amount"] > 1000).astype(int)

    txn_type = df["transaction_type"].fillna("purchase")
    out["txn_type_encoded"] = txn_type.map(TRANSACTION_TYPE_MAP).fillna(0).astype(int)
    out["is_cash_withdrawal"] = (txn_type == "cash_withdrawal").astype(int)
    out["is_card_testing"] = (txn_type == "card_testing").astype(int)
    out["is_chargeback"] = (txn_type == "chargeback").astype(int)
    out["is_refund"] = (txn_type == "refund").astype(int)

    out["is_high_risk_merchant"] = df["merchant"].isin(HIGH_RISK_MERCHANTS).astype(int)
    out["is_high_risk_country"] = (
        df["merchant_country"].isin(HIGH_RISK_COUNTRIES).astype(int)
    )
    out["user_segment"] = df["user_id"].astype(int) % 100

    try:
        dt = pd.to_datetime(df["event_time"])
        out["hour_of_day"] = dt.dt.hour
        out["day_of_week"] = dt.dt.dayofweek
        out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    except Exception:
        out["hour_of_day"] = 12
        out["day_of_week"] = 0
        out["is_weekend"] = 0

    return out[FEATURE_COLUMNS]
