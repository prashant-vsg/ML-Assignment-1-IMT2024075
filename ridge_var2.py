
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score

# Load Var2 dataset
df = pd.read_csv("IMT2024075_train_var2.csv")

X = df.drop(columns=["y"])
y = df["y"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Original model
baseline = make_pipeline(
    PolynomialFeatures(degree=8, include_bias=False),
    LinearRegression()
)

baseline_mse = -cross_val_score(
    baseline, X, y, cv=kf,
    scoring="neg_mean_squared_error"
).mean()

print(f"OLS Baseline MSE: {baseline_mse:.6f}")

# Ridge models
alphas = [
    0.000001, 0.0001, 0.001,
    0.01, 0.1, 1, 10, 100
]

print("\nRIDGE REGRESSION — VAR2")

for alpha in alphas:

    model = make_pipeline(
        PolynomialFeatures(degree=8, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha)
    )

    mse = -cross_val_score(
        model, X, y,
        cv=kf,
        scoring="neg_mean_squared_error"
    ).mean()

    print(f"Alpha = {alpha:<10} MSE = {mse:.6f}")

    
# =====================================
# VAR2 — JOINT RIDGE DEGREE + ALPHA
# =====================================

results = []

alphas = [
    0.00001, 0.0001, 0.001,
    0.01, 0.1, 1, 10, 100
]

for degree in range(1, 21):

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
            model, X, y,
            cv=kf,
            scoring="neg_mean_squared_error"
        ).mean()

        results.append({
            "Degree": degree,
            "Alpha": alpha,
            "MSE": mse
        })

results_df = pd.DataFrame(results).sort_values("MSE")

print("\nTOP 10 RIDGE MODELS — VAR2")
print(results_df.head(10).to_string(index=False))

results_df.to_csv("var2_ridge_comparison.csv", index=False)


# =====================================
# VAR2 — RIDGE FINE-TUNING
# =====================================

print("\nFINE-TUNING RIDGE — VAR2")

fine_results = []

degrees = [9, 10, 11, 12]
fine_alphas = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 2, 3, 5]

for degree in degrees:
    for alpha in fine_alphas:

        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
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

        fine_results.append({
            "Degree": degree,
            "Alpha": alpha,
            "MSE": mse
        })

fine_df = pd.DataFrame(fine_results).sort_values("MSE")

print(fine_df.head(15).to_string(index=False))


# =====================================
# VAR2 — RIDGE FEATURE SELECTION
# =====================================

from itertools import combinations

print("\nRIDGE FEATURE SELECTION — VAR2")

features = list(X.columns)
feature_results = []

for r in range(1, len(features) + 1):
    for subset in combinations(features, r):

        X_subset = X[list(subset)]

        model = make_pipeline(
            PolynomialFeatures(degree=12, include_bias=False),
            StandardScaler(),
            Ridge(alpha=3)
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

print(feature_df.to_string(index=False))
