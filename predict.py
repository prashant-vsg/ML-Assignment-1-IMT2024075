
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

# ======================================
# VAR1 — FINAL RIDGE MODEL
# ======================================

train1 = pd.read_csv("IMT2024075_train_var1.csv")
test1 = pd.read_csv("IMT2024075_test_var1.csv")

features1 = ["x1", "x2", "x3", "x4", "x5", "x6"]

model1 = make_pipeline(
    PolynomialFeatures(degree=5, include_bias=False),
    StandardScaler(),
    Ridge(alpha=15)
)

model1.fit(train1[features1], train1["y"])

pred1 = model1.predict(test1[features1])

pd.DataFrame({"y": pred1}).to_csv(
    "IMT2024075_pred_var1.csv",
    index=False
)

# ======================================
# VAR2 — FINAL RIDGE MODEL
# ======================================

train2 = pd.read_csv("IMT2024075_train_var2.csv")
test2 = pd.read_csv("IMT2024075_test_var2.csv")

features2 = ["x1", "x2", "x3"]

model2 = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    StandardScaler(),
    Ridge(alpha=0.6)
)

model2.fit(train2[features2], train2["y"])

pred2 = model2.predict(test2[features2])

pd.DataFrame({"y": pred2}).to_csv(
    "IMT2024075_pred_var2.csv",
    index=False
)

# ======================================
# VERIFY PREDICTIONS
# ======================================

for filename in [
    "IMT2024075_pred_var1.csv",
    "IMT2024075_pred_var2.csv"
]:
    df = pd.read_csv(filename)

    assert df.shape == (1000, 1)
    assert list(df.columns) == ["y"]
    assert np.isfinite(df["y"].to_numpy()).all()

    print(f"{filename} — VERIFIED")
    print("Min:", df["y"].min())
    print("Max:", df["y"].max())
    print("Mean:", df["y"].mean())
    print()
