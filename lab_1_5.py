import pandas as pd


def compute_binary_similarity(v1, v2):
    f11 = sum((v1 == 1) & (v2 == 1))
    f00 = sum((v1 == 0) & (v2 == 0))
    f01 = sum((v1 == 0) & (v2 == 1))
    f10 = sum((v1 == 1) & (v2 == 0))

    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) > 0 else 0.0
    smc = (f11 + f00) / (f00 + f01 + f10 + f11) if len(v1) > 0 else 0.0
    return jc, smc, f11, f00, f01, f10


def main():
    file_path = "Lab Session Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

    df_binary = df.copy()
    for col in df_binary.columns:
        df_binary[col] = df_binary[col].replace(
            {"t": 1, "f": 0, "M": 1, "F": 0, "y": 1, "n": 0}
        )

    binary_cols = [
        c
        for c in df_binary.columns
        if set(df_binary[c].dropna().unique()).issubset({0, 1})
    ]

    v1 = df_binary[binary_cols].iloc[0]
    v2 = df_binary[binary_cols].iloc[1]

    jc, smc, f11, f00, f01, f10 = compute_binary_similarity(v1, v2)

    print("--- A5: BINARY SIMILARITY MEASURES ---")
    print(
        f"f11: {f11} | f00: {f00} | f01: {f01} | f10: {f10}\n"
        f"Jaccard Coefficient (JC):             {jc:.4f}\n"
        f"Simple Matching Coefficient (SMC):    {smc:.4f}"
    )


if __name__ == "__main__":
    main()