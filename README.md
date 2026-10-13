# samd-model-evaluation
A comprehensive Python evaluation library for ML models (especially for software as a medical device). It calculates probability and classification metrics (e.g., Brier Skill Score, O:E Ratio, adjusted PPV/NPV) and generates automated radar charts to prevent model overestimation by providing a holistic "cockpit" overview.

🌱 Continuous Development: This repository is a living project. I am continuously iterating on the codebase to add new evaluation capabilities and refine the existing pipeline as my work evolves.

# A Comprehensive ML Model Evaluation Dashboard

Evaluating a machine learning model is like reading an airplane cockpit—relying on a single metric (like accuracy or AUC) often leads to overestimating capabilities or missing critical flaws. 

This repository provides a custom Python evaluation library built from scratch to maintain a holistic overview of a model's true predictive power, its calibration, and its edge cases.

## 🚀 Features

This library calculates a wide dashboard of metrics and provides automated visualizations:

*   **Probability & Calibration Metrics:** ROC-AUC, PR-AUC, partial AUC, Brier Score, Brier Skill Score, and O:E Ratio (Observed to Expected).
*   **Hard Classification Metrics:** True/False Positives & Negatives, Sensitivity, Specificity, Positive/Negative Likelihood Ratios (LR+, LR-).
*   **Adjusted Predictive Values:** Calculates PPV and NPV adjusted to theoretical prevalences (e.g., 10%, 30%, 50%, 80%) to simulate real-world population shifts.
*   **Radar Chart Visualization:** Automatically generates a spider/radar chart for a quick visual inspection of the model's balance across key metrics (AUC, PR_AUC, 1-Brier, Sens, Spec, PPV, NPV).

## 📦 Requirements

*   `numpy`
*   `matplotlib`
*   `scikit-learn`

## 💡 Usage Example

You can easily compute all metrics and plot the radar chart by passing your true labels and predicted probabilities:

```python
import numpy as np
from metrics import calculate_metrics, plot_spider_chart

# Example data
y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])
y_prob = np.array([0.1, 0.8, 0.4, 0.9, 0.7, 0.2, 0.3, 0.6])
threshold = 0.5

# 1. Calculate all combined metrics
# Calculates probability metrics, classification metrics, and prevalence-adjusted PPV/NPV
results = calculate_metrics(
    y_true=y_true, 
    y_prob=y_prob, 
    threshold=threshold,
    adj_prevalences=(0.1, 0.3, 0.5)
)

print(f"Brier Skill Score: {results['Brier_Skill_Score']:.4f}")
print(f"O:E Ratio: {results['O:E_Ratio']:.4f}")

# 2. Visualize model performance as a Radar Chart
plot_spider_chart(
    metrics_dict=results,
    title="Holistic Model Evaluation",
    save_path="./model_evaluation_cockpit.png"
)
