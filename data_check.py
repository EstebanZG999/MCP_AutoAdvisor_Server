# data_check.py — Simple review of Updated_Car_Sales_Data.csv
import argparse
import pandas as pd

EXPECTED_COLUMNS = [
    "Car Make","Car Model","Year","Mileage","Price","Fuel Type",
    "Color","Transmission","Options/Features","Condition","Accident"
]

def main(csv_path: str):
    print(f"[INFO] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # Column verification
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing expected columns: {missing}")
    else:
        print("[OK] All expected columns are present.")

    print("\n=== General summary ===")
    print(df.info())

    print("\n=== First rows ===")
    print(df.head())

    print("\n=== Null values per column ===")
    print(df.isna().sum())

    print("\n=== Cardinality of categorical columns ===")
    cat_cols = ["Car Make","Car Model","Fuel Type","Color","Transmission","Condition","Accident"]
    for col in cat_cols:
        if col in df.columns:
            print(f"- {col}: {df[col].nunique()} unique values → {df[col].unique()[:10]}...")

    print("\n=== Numerical statistics ===")
    num_cols = ["Year","Mileage","Price"]
    print(df[num_cols].describe(percentiles=[.01, .05, .95, .99]))

    print("\n=== Possible outliers ===")
    for col in ["Year","Mileage","Price"]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            p1, p99 = s.quantile(0.01), s.quantile(0.99)
            print(f"- {col}: expected range ~ [{p1}, {p99}], min={s.min()}, max={s.max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple data check for Updated_Car_Sales_Data.csv")
    parser.add_argument("--csv", required=True, help="Path to the car sales CSV file")
    args = parser.parse_args()
    main(args.csv)
