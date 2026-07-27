import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")


df["Date"] = pd.to_datetime(df["Date"])
price_data = df["Price"].dropna()


np_mean = np.mean(price_data)
np_var = np.var(price_data)



def my_mean(data):
    return sum(data) / len(data)


def my_var(data):
    m = my_mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


custom_mean_val = my_mean(price_data)
custom_var_val = my_var(price_data)

print(f"Numpy Mean: {np_mean:.2f} | Custom Mean: {custom_mean_val:.2f}")
print(f"Numpy Var:  {np_var:.2f}  | Custom Var:  {custom_var_val:.2f}")


pop_mean = np_mean


wed_prices = df[df["Day"] == "Wed"]["Price"]
wed_mean = np.mean(wed_prices)

apr_prices = df[df["Date"].dt.month == 4]["Price"]
apr_mean = np.mean(apr_prices)

print(f"\nPopulation Mean:       {pop_mean:.2f}")
print(f"Wednesday Sample Mean: {wed_mean:.2f}")
print(f"April Sample Mean:     {apr_mean:.2f}")


if df["Chg%"].dtype == "O":
    df["Chg%"] = df["Chg%"].str.rstrip("%").astype(float)

chg_data = df["Chg%"].dropna()


is_loss = lambda x: x < 0
loss_count = sum(1 for x in chg_data if is_loss(x))
prob_loss = loss_count / len(chg_data)

wed_df = df[df["Day"] == "Wed"]
wed_profit_count = len(wed_df[wed_df["Chg%"] > 0])


prob_profit_wednesday = wed_profit_count / len(df)


cond_prob_profit_wed = wed_profit_count / len(wed_df)

print(f"\nProbability of Loss: {prob_loss:.4f}")
print(f"Probability of Profit on Wednesday: {prob_profit_wednesday:.4f}")
print(f"Conditional Prob P(Profit | Wednesday): {cond_prob_profit_wed:.4f}")



def main():
    plt.scatter(df["Day"], df["Chg%"], color="blue", alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Chg% vs Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.show()


if __name__ == "__main__":
    main()


