import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_binary_similarity(v1, v2):
    f11 = sum((v1 == 1) & (v2 == 1))
    f00 = sum((v1 == 0) & (v2 == 0))
    f01 = sum((v1 == 0) & (v2 == 1))
    f10 = sum((v1 == 1) & (v2 == 0))

    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) > 0 else 0.0
    smc = (f11 + f00) / (f00 + f01 + f10 + f11) if len(v1) > 0 else 0.0
    return jc, smc


def compute_cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return (
        dot_product / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0.0
    )


def main():
    file_path = "Lab Session Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    df = df.replace("?", np.nan)

    # Process first 20 vectors
    n_samples = min(20, len(df))

    # Encoding binary features for JC/SMC
    df_bin = df.copy()
    for col in df_bin.columns:
        df_bin[col] = df_bin[col].replace(
            {"t": 1, "f": 0, "M": 1, "F": 0, "y": 1, "n": 0}
        )
    binary_cols = [
        c
        for c in df_bin.columns
        if set(df_bin[c].dropna().unique()).issubset({0, 1})
    ]
    df_20_bin = df_bin[binary_cols].iloc[:n_samples]

    # Encoding all features for Cosine Similarity
    df_num = df.copy()
    for col in df_num.columns:
        df_num[col] = df_num[col].astype("category").cat.codes
    df_20_num = df_num.iloc[:n_samples]

    jc_matrix = np.zeros((n_samples, n_samples))
    smc_matrix = np.zeros((n_samples, n_samples))
    cos_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            j_val, s_val = compute_binary_similarity(
                df_20_bin.iloc[i], df_20_bin.iloc[j]
            )
            jc_matrix[i, j] = j_val
            smc_matrix[i, j] = s_val
            cos_matrix[i, j] = compute_cosine_similarity(
                df_20_num.iloc[i].to_numpy(), df_20_num.iloc[j].to_numpy()
            )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(jc_matrix, annot=False, cmap="Blues", ax=axes[0])
    axes[0].set_title("Jaccard Coefficient (JC)")

    sns.heatmap(smc_matrix, annot=False, cmap="Greens", ax=axes[1])
    axes[1].set_title("Simple Matching Coefficient (SMC)")

    sns.heatmap(cos_matrix, annot=False, cmap="Reds", ax=axes[2])
    axes[2].set_title("Cosine Similarity")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()