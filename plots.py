
import pandas as pd
import matplotlib.pyplot as plt

# Use the saved results from our joint searches
var1 = pd.read_csv("var1_model_comparison.csv")
var2 = pd.read_csv("var2_model_comparison.csv")

# For each degree, find the best feature subset
var1_best = var1.groupby("Degree")["MSE"].min().sort_index()
var2_best = var2.groupby("Degree")["MSE"].min().sort_index()

# Plot Var1
plt.figure(figsize=(8, 5))
plt.plot(var1_best.index, var1_best.values, marker="o")
plt.yscale("symlog")
plt.xlabel("Polynomial Degree")
plt.ylabel("Best 5-Fold CV MSE")
plt.title("Var1: Polynomial Degree vs Validation MSE")
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("var1_degree_comparison.png", dpi=300)
plt.close()

# Plot Var2
plt.figure(figsize=(8, 5))
plt.plot(var2_best.index, var2_best.values, marker="o")
plt.yscale("symlog")
plt.xlabel("Polynomial Degree")
plt.ylabel("Best 5-Fold CV MSE")
plt.title("Var2: Polynomial Degree vs Validation MSE")
plt.xticks(range(1, 21))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("var2_degree_comparison.png", dpi=300)
plt.close()

print("Both graphs saved successfully.")
