from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent


import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

# Load Var1
df = pd.read_csv((PROJECT_ROOT / "data/IMT2024075_train_var1.csv"))

X = df.drop(columns=["y"])
y = df["y"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Our existing baseline
baseline = make_pipeline(
    PolynomialFeatures(degree=4, include_bias=False),
    LinearRegression()
)

baseline_mse = -cross_val_score(
    baseline, X, y, cv=kf,
    scoring="neg_mean_squared_error"
).mean()

print(f"OLS Baseline MSE: {baseline_mse:.6f}")

# Test different regularization strengths
alphas = [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

print("\nRIDGE REGRESSION — VAR1")

for alpha in alphas:

    model = make_pipeline(
        PolynomialFeatures(degree=4, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    mse = -cross_val_score(
        model, X, y, cv=kf,
        scoring="neg_mean_squared_error"
    ).mean()

    print(f"Alpha = {alpha:<10} MSE = {mse:.6f}")

results = []

alphas = [0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 300]

for degree in range(1, 11):
    print(f"Testing degree {degree}...", flush=True)

    for alpha in alphas:

        model = make_pipeline(
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            ),
            StandardScaler(),
            Ridge(alpha=alpha)
        )

        mse = -cross_val_score(
            model,
            X,
            y,
            cv=kf,
            scoring="neg_mean_squared_error"
        ).mean()

        results.append({
            "Degree": degree,
            "Alpha": alpha,
            "MSE": mse
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MSE")

print("\nTOP 10 RIDGE MODELS — VAR1")
print(results_df.head(10).to_string(index=False))

results_df.to_csv((PROJECT_ROOT / "results/var1_ridge_comparison.csv"), index=False)

print("\nFINE-TUNING RIDGE — VAR1")

fine_alphas = [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25]

fine_results = []

for alpha in fine_alphas:

    model = make_pipeline(
        PolynomialFeatures(degree=5, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    mse = -cross_val_score(
        model, X, y,
        cv=kf,
        scoring="neg_mean_squared_error"
    ).mean()

    fine_results.append({
        "Alpha": alpha,
        "MSE": mse
    })

fine_df = pd.DataFrame(fine_results).sort_values("MSE")

print(fine_df.to_string(index=False))

from itertools import combinations

print("\nRIDGE FEATURE SELECTION — VAR1")

features = list(X.columns)
feature_results = []

for r in range(1, len(features) + 1):
    for subset in combinations(features, r):

        X_subset = X[list(subset)]

        model = make_pipeline(
            PolynomialFeatures(degree=5, include_bias=False),
            StandardScaler(),
            Ridge(alpha=15)
        )

        mse = -cross_val_score(
            model,
            X_subset,
            y,
            cv=kf,
            scoring="neg_mean_squared_error"
        ).mean()

        feature_results.append({
            "Features": subset,
            "MSE": mse
        })

feature_df = pd.DataFrame(feature_results).sort_values("MSE")

print(feature_df.head(10).to_string(index=False))
