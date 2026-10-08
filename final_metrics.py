
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_validate

configs = [
    ("Var1", "IMT2024075_train_var1.csv", 5, 15),
    ("Var2", "IMT2024075_train_var2.csv", 10, 0.6)
]

for name, filename, degree, alpha in configs:

    df = pd.read_csv(filename)
    X = df.drop(columns=["y"])
    y = df["y"]

    model = make_pipeline(
        PolynomialFeatures(degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    print(f"\n{name} — FINAL RIDGE MODEL")

    # Original validation split
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=["neg_mean_squared_error", "r2"]
    )

    print("Seed 42:")
    print("MSE:", -scores["test_neg_mean_squared_error"].mean())
    print("R2:", scores["test_r2"].mean())

    # Additional stability checks
    all_mse = []
    all_r2 = []

    for seed in [7, 21, 123]:
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)

        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["neg_mean_squared_error", "r2"]
        )

        all_mse.append(-scores["test_neg_mean_squared_error"].mean())
        all_r2.append(scores["test_r2"].mean())

    print("Three additional seeds:")
    print("Mean MSE:", np.mean(all_mse))
    print("Mean R2:", np.mean(all_r2))
