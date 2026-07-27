import numpy as np
import pandas as pd


def normalize_min_max(df, numeric_cols):
    df_norm = df.copy()
    for col in numeric_cols:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val != min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    return df_norm


def main():
    file_path = "Lab Session Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    df = df.replace("?", np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df_normalized = normalize_min_max(df, numeric_cols)

    print("--- A9: MIN-MAX NORMALIZED DATA SAMPLE ---")
    print(df_normalized[numeric_cols].head())


if __name__ == "__main__":
    main()