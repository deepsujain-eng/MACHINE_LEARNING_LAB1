import numpy as np
import pandas as pd


def impute_missing_values(df, numeric_cols, categorical_cols):
    df_imputed = df.copy()

    # Categorical: Mode
    for col in categorical_cols:
        if not df_imputed[col].mode().empty:
            df_imputed[col] = df_imputed[col].fillna(
                df_imputed[col].mode()[0]
            )

    # Numeric: Median if outliers exist, Mean if clean
    for col in numeric_cols:
        q1 = df_imputed[col].quantile(0.25)
        q3 = df_imputed[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df_imputed[
            (df_imputed[col] < q1 - 1.5 * iqr)
            | (df_imputed[col] > q3 + 1.5 * iqr)
        ]

        if len(outliers) > 0:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        else:
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

    return df_imputed


def main():
    file_path = "Lab Session Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    df = df.replace("?", np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print("Missing values BEFORE imputation:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    df_clean = impute_missing_values(df, numeric_cols, categorical_cols)

    print("\nMissing values AFTER imputation:")
    print(df_clean.isnull().sum().sum(), "missing values remaining.")


if __name__ == "__main__":
    main()