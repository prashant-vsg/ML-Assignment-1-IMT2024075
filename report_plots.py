
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

experiments = [
    {
        "name": "var1",
        "file": "IMT2024075_train_var1.csv",
        "degrees": range(1, 8),
        "ridge_alpha": 15,
        "final_degree": 5
    },
    {
        "name": "var2",
        "file": "IMT2024075_train_var2.csv",
        "degrees": range(5, 14),
        "ridge_alpha": 0.6,
        "final_degree": 10
    }
]

for exp in experiments:

    df = pd.read_csv(exp["file"])
    X = df.drop(columns=["y"])
    y = df["y"]

    ols_mse = []
    ridge_mse = []

    for degree in exp["degrees"]:

        ols = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            LinearRegression()
        )

        ridge = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            StandardScaler(),
            Ridge(alpha=exp["ridge_alpha"])
        )

        ols_score = -cross_val_score(
            ols, X, y, cv=kf,
            scoring="neg_mean_squared_error"
        ).mean()

        ridge_score = -cross_val_score(
            ridge, X, y, cv=kf,
            scoring="neg_mean_squared_error"
        ).mean()

        ols_mse.append(ols_score)
        ridge_mse.append(ridge_score)

    plt.figure(figsize=(8, 4.5))

    plt.plot(
        exp["degrees"], ols_mse,
        marker="o", label="OLS"
    )

    plt.plot(
        exp["degrees"], ridge_mse,
        marker="s",
        label=f"Ridge (alpha={exp['ridge_alpha']})"
    )

    plt.axvline(
        exp["final_degree"],
        linestyle="--",
        alpha=0.6,
        label="Selected degree"
    )

    plt.yscale("log")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("5-Fold CV MSE")
    plt.title(f"{exp['name'].upper()}: OLS vs Ridge")
    plt.xticks(list(exp["degrees"]))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{exp['name']}_report_plot.png",
        dpi=300
    )

    plt.close()

    print(f"{exp['name']} report graph saved")
