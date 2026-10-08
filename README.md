
# ML Assignment 1 — Polynomial Regression

**Roll Number:** IMT2024075

## Overview

This project implements polynomial regression for two datasets:

- **Var1:** Steam Turbine Optimization
- **Var2:** Subterranean Thermal Reservoir Mapping

Ordinary Least Squares (OLS) and Ridge regression were compared using polynomial degree selection, feature subset experiments, and 5-fold cross-validation.

## Final Models and Results

| Parameter | Var1 | Var2 |
|---|---|---|
| Model | Ridge | Ridge |
| Polynomial Degree | 5 | 10 |
| Alpha | 15 | 0.6 |
| Features | All 6 | All 3 |
| Mean CV MSE | 0.523408 | 0.236946 |
| Mean CV R² | 0.947907 | 0.995131 |

*The reported metrics are averages of 5-fold cross-validation results using random seeds 7, 21, and 123.*

## Repository Structure

| Folder | Contents |
|---|---|
| `src/` | Training, evaluation, prediction, and plotting scripts |
| `data/` | Training datasets, test datasets, and sample submission |
| `predictions/` | Final prediction CSV files |
| `results/` | Model comparison results |
| `figures/` | Generated graphs and visualizations |
| `reports/` | Assignment question paper and final report |

## Requirements

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install dependencies from the project root:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

## Running the Code

Run the following commands from the repository's root directory.

Generate final predictions:

```bash
python src/predict.py
```

Reproduce the final validation metrics:

```bash
python src/final_metrics.py
```

The generated prediction files are saved in `predictions/`.

## Submission

- **Report:** `reports/IMT2024075.pdf`
- **Var1 predictions:** `predictions/IMT2024075_pred_var1.csv`
- **Var2 predictions:** `predictions/IMT2024075_pred_var2.csv`
