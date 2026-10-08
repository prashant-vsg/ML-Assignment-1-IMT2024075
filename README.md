
# ML Assignment 1 — Polynomial Regression

**Roll Number:** IMT2024075

## Overview

Polynomial regression models were developed for two datasets:

- **Var1:** Steam Turbine Optimization
- **Var2:** Subterranean Thermal Reservoir Mapping

Ordinary Least Squares (OLS) and Ridge regression were compared using 5-fold cross-validation. Polynomial degrees, feature subsets, and regularization strengths were tested to select the final models.

## Final Results

| Parameter | Var1 | Var2 |
|---|---|---|
| Model | Ridge | Ridge |
| Polynomial Degree | 5 | 10 |
| Alpha | 15 | 0.6 |
| Features | All 6 | All 3 |
| Mean CV MSE | 0.523408 | 0.236946 |
| Mean CV R² | 0.947907 | 0.995131 |

*Metrics are averaged across 5-fold CV runs with random seeds 7, 21, and 123.*

## Requirements

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

## Running the Code

Run the final prediction script:

```bash
python predict.py
```

This generates:

- `IMT2024075_pred_var1.csv`
- `IMT2024075_pred_var2.csv`

Other scripts contain the polynomial degree comparisons, feature selection, Ridge experiments, cross-validation, and plotting code.

## Report

The assignment report is available in `IMT2024075.pdf`.
