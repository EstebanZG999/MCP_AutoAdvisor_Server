# MCP_AutoAdvisor_Server - Tools (Updated_Car_Sales_Data)
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

REQUIRED_COLUMNS = [
    "Car Make","Car Model","Year","Mileage","Price","Fuel Type",
    "Color","Transmission","Options/Features","Condition","Accident"
]

def _check_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Data types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Minimal string normalization
    for c in ["Car Make","Car Model","Fuel Type","Color","Transmission","Condition","Accident"]:
        df[c] = df[c].astype(str).str.strip()

    # Plausible filters
    df = df.dropna(subset=["Year","Mileage","Price","Fuel Type","Transmission","Condition","Accident"])
    df = df[(df["Year"] >= 2010) & (df["Year"] <= 2025)]
    df = df[(df["Mileage"] >= 0) & (df["Mileage"] <= 300_000)]
    df = df[(df["Price"] > 0) & (df["Price"] <= 400_000)]
    return df

def init_data_and_model(csv_path: Path, state: Dict[str, Any]) -> None:
    df = pd.read_csv(csv_path)
    df = _check_columns(df)
    df = _clean_df(df)

    # Features for the model
    feature_cols_numeric = ["Year","Mileage"]
    feature_cols_categ = ["Fuel Type","Transmission","Condition","Accident","Car Make","Car Model"]

    X = df[feature_cols_numeric + feature_cols_categ]
    y = df["Price"]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_categ),
        ]
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("reg", LinearRegression())
        ]
    )
    model.fit(X, y)

    state["df"] = df
    state["model"] = model
    state["feature_columns"] = feature_cols_numeric + feature_cols_categ

# ---------- Filtering helpers ----------
def _apply_filters(df: pd.DataFrame, args: Dict[str, Any]) -> pd.DataFrame:
    q = df
    def _norm(s): return str(s).strip().lower()

    # Exact filters
    for col in ["Car Make","Car Model","Fuel Type","Transmission","Condition","Accident"]:
        if col in args and args[col]:
            q = q[q[col].astype(str).str.strip().str.lower() == _norm(args[col])]

    # Ranges
    if args.get("Year_min") is not None:
        q = q[q["Year"] >= int(args["Year_min"])]
    if args.get("Year_max") is not None:
        q = q[q["Year"] <= int(args["Year_max"])]
    if args.get("Price_max") is not None:
        q = q[q["Price"] <= float(args["Price_max"])]
    if args.get("Mileage_max") is not None:
        q = q[q["Mileage"] <= float(args["Mileage_max"])]

    return q

# ---------- Tools ----------
def tool_filter_cars(df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    q = _apply_filters(df, args)
    limit = int(args.get("limit", 20))
    cols = ["Car Make","Car Model","Year","Mileage","Price","Fuel Type","Transmission","Condition","Accident","Color"]
    out = q[cols].sort_values(by=["Price","Year"], ascending=[True, False]).head(limit).to_dict(orient="records")
    return {"count": len(out), "results": out}

def tool_recommend(df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    if "budget_max" not in args:
        raise ValueError("budget_max is required")
    args_local = dict(args)  # copy
    args_local["Price_max"] = args["budget_max"]
    q = _apply_filters(df, args_local)
    cols = ["Car Make","Car Model","Year","Mileage","Price","Fuel Type","Transmission","Condition","Accident"]
    out = q.sort_values(by=["Price","Year"], ascending=[True, False])[cols].head(int(args.get("limit", 10))).to_dict(orient="records")
    return {"budget_max": float(args["budget_max"]), "count": len(out), "recommendations": out}

def tool_estimate_price(model, feature_columns: List[str], args: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "Car Make": str(args.get("Car Make", "")),
        "Car Model": str(args.get("Car Model", "")),
        "Year": int(args["Year"]),
        "Mileage": float(args["Mileage"]),
        "Fuel Type": str(args["Fuel Type"]),
        "Transmission": str(args["Transmission"]),
        "Condition": str(args.get("Condition", "Used")),
        "Accident": str(args.get("Accident", "No")),
    }
    X_pred = pd.DataFrame([payload])[feature_columns]
    pred = float(model.predict(X_pred)[0])
    return {"input": payload, "estimated_price": pred}

def tool_average_price(df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    q = _apply_filters(df, args)
    avg = None if q.empty else float(q["Price"].mean())
    n = int(len(q))
    return {"filters": args, "average_price": avg, "samples": n}

def tool_top_cars(df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    q = _apply_filters(df, args)
    n = int(args.get("n", 10))
    order = args.get("sort_order", "cheap")
    asc = True if order == "cheap" else False
    cols = ["Car Make","Car Model","Year","Mileage","Price","Fuel Type","Transmission","Condition","Accident"]
    out = q.sort_values(by="Price", ascending=asc)[cols].head(n).to_dict(orient="records")
    return {"order": order, "results": out}
