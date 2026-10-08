
import pandas as pd
import matplotlib.pyplot as plt

ols1 = pd.read_csv("var1_model_comparison.csv")
ols2 = pd.read_csv("var2_model_comparison.csv")

ridge1 = pd.read_csv("var1_ridge_comparison.csv")
ridge2 = pd.read_csv("var2_ridge_comparison.csv")

# Lowest MSE among feature subsets for each OLS degree
ols1_mse = ols1.groupby("Degree")["MSE"].min()
ols2_mse = ols2.groupby("Degree")["MSE"].min()

# Lowest MSE among tested alpha values for each Ridge degree
ridge1_mse = ridge1.groupby("Degree")["MSE"].min()
ridge2_mse = ridge2.groupby("Degree")["MSE"].min()

for name, ols, ridge, max_degree in [
    ("var1", ols1_mse, ridge1_mse, 10),
    ("var2", ols2_mse, ridge2_mse, 20)
]:
    plt.figure(figsize=(9, 5))

    plt.plot(ols.index, ols.values, marker="o", label="OLS")
    plt.plot(ridge.index, ridge.values, marker="s", label="Ridge")

    plt.yscale("symlog")
    plt.xticks(range(1, max_degree + 1))
    plt.xlabel("Polynomial Degree")
    plt.ylabel("5-Fold CV MSE")
    plt.title(f"{name.upper()}: OLS vs Ridge Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{name}_final_comparison.png", dpi=300)
    plt.close()

    print(f"{name}_final_comparison.png saved")
