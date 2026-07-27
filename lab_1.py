import pandas as pd
import numpy as np



df = pd.read_excel('Lab Session Data.xlsx', sheet_name='Purchase data')
cols = ['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']
df_clean = df[cols].dropna()

X = df_clean[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
y = df_clean['Payment (Rs)'].values

# --- A1 Question 4: Rank of Matrix ---
rank_X = np.linalg.matrix_rank(X)
print("Rank of feature matrix X:", rank_X)

# --- A1 Question 5: Pseudo-Inverse Cost Calculation ---
X_pinv = np.linalg.pinv(X)
costs = X_pinv @ y

print(f"Cost per Candy: ₹{costs[0]:.2f}")
print(f"Cost per Kg of Mangoes: ₹{costs[1]:.2f}")
print(f"Cost per Milk Packet: ₹{costs[2]:.2f}")




