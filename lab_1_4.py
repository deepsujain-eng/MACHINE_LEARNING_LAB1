import numpy as np
import pandas as pd


file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
df = df.replace("?", np.nan)


categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("--- A4: DATA EXPLORATION ---")
print(f"Numeric Attributes ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical Attributes ({len(categorical_cols)}): {categorical_cols}\n")

print("Suggested Encodings:")
for col in categorical_cols:
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) <= 2:
        print(f" - {col}: Binary/Nominal -> Label or One-Hot Encoding")
    else:
        print(f" - {col}: Nominal -> One-Hot Encoding")


print("\n--- NUMERIC DATA RANGES ---")
for col in numeric_cols:
    c_min, c_max = df[col].min(), df[col].max()
    print(f" - {col}: Min = {c_min}, Max = {c_max}, Range = {c_max - c_min}")


print("\n--- MISSING VALUES ---")
missing_sum = df.isnull().sum()
print(missing_sum[missing_sum > 0])


print("\n--- OUTLIERS (IQR Method) ---")
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)][col]
    print(f" - {col}: {len(outliers)} outliers found")


print("\n--- MEAN, VARIANCE & STD DEV ---")
for col in numeric_cols:
    print(
        f" - {col}: Mean = {df[col].mean():.4f}, Var = {df[col].var():.4f}, Std = {df[col].std():.4f}"
    )