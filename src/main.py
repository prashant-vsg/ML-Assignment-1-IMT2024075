from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline

# STEP 1: Load the dataset
train_var1 = pd.read_csv((PROJECT_ROOT / "data/IMT2024075_train_var1.csv"))

# STEP 2: Separate inputs and target
X1 = train_var1.drop(columns=["y"])
y1 = train_var1["y"]

print("Dataset shape:", train_var1.shape)
print("Features:", list(X1.columns))

# STEP 3: Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# STEP 4: Evaluate degrees 1 through 10
print("\nPOLYNOMIAL DEGREE COMPARISON")

for degree in range(1, 11):

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    mse_scores = -cross_val_score(
        model, X1, y1,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    r2_scores = cross_val_score(
        model, X1, y1,
        cv=kf,
        scoring="r2"
    )

    print(
        f"Degree {degree}: "
        f"Mean MSE = {mse_scores.mean():.4f}, "
        f"Mean R2 = {r2_scores.mean():.4f}"
    )

    from itertools import combinations

print("\nFEATURE SELECTION — VAR1")

features = list(X1.columns)
results = []

# Try every non-empty combination of the 6 features
for r in range(1, len(features) + 1):
    for subset in combinations(features, r):

        X_subset = X1[list(subset)]

        model = make_pipeline(
            PolynomialFeatures(degree=4, include_bias=False),
            LinearRegression()
        )

        mse_scores = -cross_val_score(
            model,
            X_subset,
            y1,
            cv=kf,
            scoring="neg_mean_squared_error"
        )

        results.append({
            "Features": subset,
            "MSE": mse_scores.mean()
        })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MSE")

print(results_df.head(10).to_string(index=False))

# ==========================================
# VAR2 — POLYNOMIAL DEGREE SELECTION
# ==========================================

train_var2 = pd.read_csv((PROJECT_ROOT / "data/IMT2024075_train_var2.csv"))

X2 = train_var2.drop(columns=["y"])
y2 = train_var2["y"]

print("\nVAR2 DATASET")
print("Dataset shape:", train_var2.shape)
print("Features:", list(X2.columns))

print("\nPOLYNOMIAL DEGREE COMPARISON — VAR2")

for degree in range(1, 21):

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    mse_scores = -cross_val_score(
        model,
        X2,
        y2,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    r2_scores = cross_val_score(
        model,
        X2,
        y2,
        cv=kf,
        scoring="r2"
    )

    print(
        f"Degree {degree}: "
        f"Mean MSE = {mse_scores.mean():.4f}, "
        f"Mean R2 = {r2_scores.mean():.4f}"
    )

    # ==========================================
# VAR2 — FEATURE SELECTION
# ==========================================

print("\nFEATURE SELECTION — VAR2")

features2 = list(X2.columns)
results2 = []

for r in range(1, len(features2) + 1):
    for subset in combinations(features2, r):

        X_subset = X2[list(subset)]

        model = make_pipeline(
            PolynomialFeatures(degree=8, include_bias=False),
            LinearRegression()
        )

        mse_scores = -cross_val_score(
            model,
            X_subset,
            y2,
            cv=kf,
            scoring="neg_mean_squared_error"
        )

        r2_scores = cross_val_score(
            model,
            X_subset,
            y2,
            cv=kf,
            scoring="r2"
        )

        results2.append({
            "Features": subset,
            "MSE": mse_scores.mean(),
            "R2": r2_scores.mean()
        })

results2_df = pd.DataFrame(results2)
results2_df = results2_df.sort_values("MSE")

print(results2_df.to_string(index=False))

# ==========================================
# VAR2 — JOINT DEGREE + FEATURE SELECTION
# ==========================================

print("\nJOINT MODEL SELECTION — VAR2")

joint_results2 = []

for r in range(1, len(features2) + 1):
    for subset in combinations(features2, r):

        X_subset = X2[list(subset)]

        for degree in range(1, 21):

            model = make_pipeline(
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                ),
                LinearRegression()
            )

            mse_scores = -cross_val_score(
                model,
                X_subset,
                y2,
                cv=kf,
                scoring="neg_mean_squared_error"
            )

            joint_results2.append({
                "Degree": degree,
                "Features": subset,
                "MSE": mse_scores.mean()
            })

        print(f"Completed feature subset: {subset}", flush=True)

joint_df2 = pd.DataFrame(joint_results2)
joint_df2 = joint_df2.sort_values("MSE")

print("\nTOP 10 VAR2 MODELS")
print(joint_df2.head(10).to_string(index=False))

joint_df2.to_csv((PROJECT_ROOT / "results/var2_model_comparison.csv"), index=False)

# ==========================================
# VAR1 — JOINT DEGREE + FEATURE SELECTION
# ==========================================

print("\nJOINT MODEL SELECTION — VAR1")

joint_results1 = []

for r in range(1, len(features) + 1):
    for subset in combinations(features, r):

        X_subset = X1[list(subset)]

        for degree in range(1, 11):

            model = make_pipeline(
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False
                ),
                LinearRegression()
            )

            mse_scores = -cross_val_score(
                model,
                X_subset,
                y1,
                cv=kf,
                scoring="neg_mean_squared_error"
            )

            joint_results1.append({
                "Degree": degree,
                "Features": subset,
                "MSE": mse_scores.mean()
            })

        print(f"Completed subset: {subset}", flush=True)

joint_df1 = pd.DataFrame(joint_results1)
joint_df1 = joint_df1.sort_values("MSE")

print("\nTOP 10 VAR1 MODELS")
print(joint_df1.head(10).to_string(index=False))

joint_df1.to_csv((PROJECT_ROOT / "results/var1_model_comparison.csv"), index=False)