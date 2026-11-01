import os
from typing import Optional, cast, dict, tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections.polar import PolarAxes


def plot_spider_chart(
        metrics_dict: dict[str, float],
        suffix: str = "",
        title: str = "Modell Metriken",
        categories: tuple[str, ...] = ('AUC',
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