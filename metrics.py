import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes 
from typing import Tuple, Dict, Optional, cast
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix, 
    brier_score_loss
)

def get_probability_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        max_fpr: Optional[float] = None
        ) -> Dict[str, float]:
    """Berechnet Metriken, die Vorhersagewahrscheinlichkeiten erfordern."""
    metrics = {}
    
    if len(np.unique(y_true)) > 1:
        metrics['AUC'] = float(roc_auc_score(y_true, y_prob))
        metrics['PR_AUC'] = float(average_precision_score(y_true, y_prob))

        if max_fpr is not None:
            metrics[f'pAUC_{max_fpr}'] = float(roc_auc_score(y_true, y_prob, max_fpr=max_fpr))
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

def get_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Berechnet Metriken, die harte Klassifikationen (0 oder 1) erfordern."""
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
        prevalences: Tuple[float, ...]
        ) -> Dict[str, float]:
    """Passt den PPV und NPV an theoretische Prävalenzen an."""
    adj_metrics = {}
    
    for prev in prevalences:
        ppv_denom = (sens * prev) + ((1 - spec) * (1 - prev))
        adj_metrics[f'PPV_adj_{prev}'] = (sens * prev) / ppv_denom if ppv_denom > 0 else 0.0
        
        npv_denom = (spec * (1 - prev)) + ((1 - sens) * prev)
        adj_metrics[f'NPV_adj_{prev}'] = (spec * (1 - prev)) / npv_denom if npv_denom > 0 else 0.0
        
    return adj_metrics

def calculate_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        adj_prevalences: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.8),
        max_fpr: Optional[float] = None,
        suffix: str = ""        
) -> Dict[str, float]:
    """
    Kombiniert alle Metriken für den Standard-Workflow.
    Kompatibel mit deinem bisherigen Setup in `classify.py`.
    """
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    
    # Unabhängige Modul-Aufrufe
    prob_metrics = get_probability_metrics(y_true, y_prob, max_fpr=max_fpr)
    pred_metrics = get_prediction_metrics(y_true, y_pred)
    adj_metrics = get_adjusted_predictive_values(
        pred_metrics['Sens'], 
        pred_metrics['Spec'], 
        adj_prevalences
    )
    
    # Ergebnisse zusammenführen und Präfix hinzufügen
    combined_metrics = {**prob_metrics, **pred_metrics, **adj_metrics}
    
    return {f"{k}{suffix}": v for k, v in combined_metrics.items()}

def plot_spider_chart(
        metrics_dict: Dict[str, float],
        suffix: str = "",
        title: str = "Modell Metriken",
        categories: Tuple[str, ...] = ('AUC',
                                       'PR_AUC',
                                       '1-Brier',                                       
                                       'Sens',
                                       'Spec',
                                       'PPV',
                                       'NPV'
                                       ),
        save_path: Optional[str] = None
        ) -> None:
    """
    Erstellt ein Spinnendiagramm (Radar Chart) für die wichtigsten Klassifikationsmetriken.
    Achtung: Metriken wie O:E Ratio oder Brier Score haben unterschiedliche Skalen.
    """    
    # Werte extrahieren (verwende 0.0 als Fallback, falls eine Metrik fehlt oder NaN ist)
    values = []
    for cat in categories:
        val = metrics_dict.get(f"{cat}{suffix}", np.nan)        
        values.append(val if not np.isnan(val) else 0.0)

    # Ersten Wert anfügen, um den Kreis im Plot zu schließen
    values += values[:1]
    
    # Winkel für die Achsen berechnen
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax_generic = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax = cast(PolarAxes, ax_generic) 
    
    # Erster Item oben (Norden)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Achsenbeschriftungen setzen
    plt.xticks(angles[:-1], categories, size=10)

    # Y-Labels formatieren (Ringe)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.0)

    # Daten plotten
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='Modell Performance')
    ax.fill(angles, values, alpha=0.25)

    bss = metrics_dict.get(f"Brier_Skill_Score{suffix}", np.nan)
    oe = metrics_dict.get(f"O:E_Ratio{suffix}", np.nan)

    full_title = title
    if not np.isnan(bss) or not np.isnan(oe):
        full_title += "\n"
        subtitle_parts = []
        if not np.isnan(bss):
            subtitle_parts.append(f"Brier Skill Score: {bss:.4f}")
        if not np.isnan(oe):
            subtitle_parts.append(f"O:E Ratio: {oe:.2f}")
        full_title += " | ".join(subtitle_parts)

    plt.title(full_title, size=14, y=1.1, linespacing=1.5)
    plt.tight_layout()

    if save_path:
        # Erstelle die Ordnerstruktur, falls sie nicht existiert
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Plot speichern (dpi=300 sorgt für scharfe Druckqualität)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # WICHTIG: Gibt den Arbeitsspeicher wieder frei!
    else:
        plt.show()