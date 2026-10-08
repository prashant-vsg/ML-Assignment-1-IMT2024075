from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

# Load datasets
train1 = pd.read_csv((PROJECT_ROOT / "data/IMT2024075_train_var1.csv"))
train2 = pd.read_csv((PROJECT_ROOT / "data/IMT2024075_train_var2.csv"))

X1, y1 = train1.drop(columns="y"), train1["y"]
X2, y2 = train2.drop(columns="y"), train2["y"]

models = {
    "Var1 OLS (d=4)": (
        X1, y1,
        make_pipeline(
            PolynomialFeatures(4, include_bias=False),
            LinearRegression()
        )
    ),

    "Var1 Ridge (d=5, alpha=15)": (
        X1, y1,
        make_pipeline(
            PolynomialFeatures(5, include_bias=False),
            StandardScaler(),
            Ridge(alpha=15)
        )
    ),

    "Var2 OLS (d=8)": (
        X2, y2,
        make_pipeline(
            PolynomialFeatures(8, include_bias=False),
            LinearRegression()
        )
    ),

    "Var2 Ridge (d=10, alpha=0.6)": (
        X2, y2,
        make_pipeline(
            PolynomialFeatures(10, include_bias=False),
            StandardScaler(),
            Ridge(alpha=0.6)
        )
    ),

    "Var2 Ridge (d=12, alpha=3)": (
        X2, y2,
        make_pipeline(
            PolynomialFeatures(12, include_bias=False),
            StandardScaler(),
            Ridge(alpha=3)
        )
    )
}

seeds = [7, 21, 123]

print("\nMODEL STABILITY COMPARISON\n")

for name, (X, y, model) in models.items():

    scores = []

    for seed in seeds:
        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        mse = -cross_val_score(
            model, X, y,
            cv=kf,
            scoring="neg_mean_squared_error"
        ).mean()

        scores.append(mse)

    print(name)
    print("MSE per seed:", np.round(scores, 6))
    print("Mean MSE:", round(np.mean(scores), 6))
    print()
