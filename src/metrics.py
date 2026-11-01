import logging
from typing import Optional, dict, tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

def get_probability_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        max_fpr: Optional[float] = None
        ) -> dict[str, float]:
    """
    Calculates metrics that require prediction probabilities.
    """
    metrics = {}
    
    if len(np.unique(y_true)) > 1:
        metrics['AUC'] = float(roc_auc_score(y_true, y_prob))
        metrics['PR_AUC'] = float(average_precision_score(y_true, y_prob))

        if max_fpr is not None:
            metrics[f'pAUC_{max_fpr}'] = float(
                roc_auc_score(y_true, y_prob, max_fpr=max_fpr)
                )
    else:
        metrics['AUC'] = np.nan
        metrics['PR_AUC'] = np.nan

        if max_fpr is not None:
            metrics[f'pAUC_{max_fpr}'] = np.nan

    if len(y_true) > 0:
        brier = float(brier_score_loss(y_true, y_prob))
        metrics['Brier'] = brier
        metrics['1-Brier'] = 1.0 - brier
        
        prevalence = np.mean(y_true)
        brier_baseline = prevalence * (1.0 - prevalence)
        bss = 1.0 - (brier / brier_baseline) if brier_baseline > 0 else np.nan
    else:
        brier, brier_baseline, bss = np.nan, np.nan, np.nan
        metrics['Brier'] = brier
        metrics['1-Brier'] = np.nan
        
    metrics['Brier_Baseline'] = float(brier_baseline)
    metrics['Brier_Skill_Score'] = float(bss)
    
    obs, exp = np.sum(y_true), np.sum(y_prob)
    metrics['O:E_Ratio'] = float(obs / exp) if exp > 0 else np.nan
    
    return metrics

def get_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculates metrics that require hard classifications (0 or 1).
    """
    metrics = {}
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics['TP'] = int(tp)
    metrics['FP'] = int(fp)
    metrics['TN'] = int(tn)
    metrics['FN'] = int(fn)

    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    metrics['Sens'] = sens
    metrics['Spec'] = spec

    metrics['LR_pos'] = float(sens / (1.0 - spec)) if (1.0 - spec) > 0 else np.nan
    metrics['LR_neg'] = float((1.0 - sens) / spec) if spec > 0 else np.nan
    
    metrics['PPV'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics['NPV'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    
    return metrics

def get_adjusted_predictive_values(
        sens: float,
        spec: float,
        prevalences: tuple[float, ...]
        ) -> dict[str, float]:
    """
    Adjust the PPV and NPV to reflect theoretical prevalence rates.
    """
    adj_metrics = {}
    
    for prev in prevalences:
        ppv_denom = (sens * prev) + ((1 - spec) * (1 - prev))
        adj_metrics[f'PPV_adj_{prev}'] = (sens * prev) / ppv_denom if ppv_denom > 0 else 0.0
        
        npv_denom = (spec * (1 - prev)) + ((1 - sens) * prev)
        adj_metrics[f'NPV_adj_{prev}'] = (spec * (1 - prev)) / npv_denom if npv_denom > 0 else 0.0
        
    return adj_metrics

def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """
    Ensures that the inputs meet the formal requirements for a SaMD evaluation.
    """
    
    # Check Dimensions and Length
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true.shape}, "
            f"but y_prob has {y_prob.shape}."
            )
        
    if len(y_true) == 0:
        raise ValueError("The arrays y_true and y_prob must not be empty.")

    # Check the valid range of the labels (only 0 and 1 are allowed)
    unique_labels = np.unique(y_true)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"y_true may only contain 0 and 1. Found: {unique_labels}")

    # Warning if only one class is present (e.g., the data set contains only zeros)
    if len(unique_labels) == 1:
        logger.warning(
            f"The dataset contains only one class ({unique_labels[0]}). "
            "Some metrics (such as AUC) cannot be calculated."
            )

    # Check the range of probabilities (0.0 to 1.0)
    if not np.all((y_prob >= 0.0) & (y_prob <= 1.0)):
        raise ValueError(
            "All values in y_prob must be valid probabilities between "
            "0.0 and 1.0."
            )
        
    # Exclude NaNs
    if np.isnan(y_true).any() or np.isnan(y_prob).any():
        raise ValueError(
            "The inputs contain NaN values, which is not allowed."
            )

def calculate_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        adj_prevalences: tuple[float, ...] = (0.1, 0.3, 0.5, 0.8),
        max_fpr: Optional[float] = None,
        suffix: str = ""        
) -> dict[str, float]:
    """
    Combines all metrics for the standard workflow.
    Compatible with your existing setup in `classify.py`.
    """

    _validate_inputs(y_true, y_prob)

    y_pred = (np.array(y_prob) >= threshold).astype(int)
    
    # Independent Module Calls
    prob_metrics = get_probability_metrics(y_true, y_prob, max_fpr=max_fpr)
    pred_metrics = get_prediction_metrics(y_true, y_pred)
    adj_metrics = get_adjusted_predictive_values(
        pred_metrics['Sens'], 
        pred_metrics['Spec'], 
        adj_prevalences
    )
    
    # Merge results and add a prefix
    combined_metrics = {**prob_metrics, **pred_metrics, **adj_metrics}
    
    return {f"{k}{suffix}": v for k, v in combined_metrics.items()}

