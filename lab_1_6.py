import numpy as np
import pandas as pd


def compute_cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 > 0 and norm_v2 > 0:
        return dot_product / (norm_v1 * norm_v2)
    return 0.0


def main():
    file_path = "Lab Session Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    df = df.replace("?", np.nan)

    # Convert non-numeric categorical columns to numbers for continuous vector comparison
    df_numeric = df.copy()
    for col in df_numeric.columns:
        df_numeric[col] = df_numeric[col].astype("category").cat.codes

    v1 = df_numeric.iloc[0].to_numpy()
    v2 = df_numeric.iloc[1].to_numpy()

    cos_sim = compute_cosine_similarity(v1, v2)

    print("--- A6: COSINE SIMILARITY ---")
    print(f"Cosine Similarity between Observation 1 & 2: {cos_sim:.4f}")


if __name__ == "__main__":
    main()